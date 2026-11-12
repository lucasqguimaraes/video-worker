# main.py
import os
import shutil
import threading
from typing import Optional

import psycopg2
import psycopg2.extras
import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from slicer import (
    SHARED_MEDIA_ROOT,
    SlicerError,
    is_valid_chapter_folder,
    slice_chapter,
)
from ocr_vision import (
    OcrError,
    extract_chapter_ocr,
)
from script_generator import (
    ScriptError,
    generate_chapter_script,
)

app = FastAPI(title="Video Machine - Slicer & OCR API", version="1.4.0")

SHARED_MEDIA_MOUNT = os.path.dirname(SHARED_MEDIA_ROOT)

REDIS_HOST = os.getenv("REDIS_HOST", "n8n_redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = 2
DB_HOST = os.getenv("DB_HOST", "n8n_postgres")
DB_NAME = os.getenv("DB_NAME", "video_machine")
DB_USER = os.getenv("DB_USER", "n8n")
DB_PASS = os.getenv("DB_PASS", "")
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY", "")

slice_lock = threading.Lock()
ocr_lock = threading.Lock()
script_lock = threading.Lock()


class SliceChapterRequest(BaseModel):
    chapter_id: int
    chapter_folder_name: str


class ExtractOcrRequest(BaseModel):
    chapter_id: int
    openai_api_key: Optional[str] = None


class PrepareChapterRequest(BaseModel):
    chapter_folder_name: str


class GenerateScriptRequest(BaseModel):
    chapter_id: int
    openai_api_key: Optional[str] = None


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD or None,
        db=REDIS_DB,
        socket_connect_timeout=3,
        socket_timeout=3,
    )


def check_shared_volume_writable() -> bool:
    try:
        os.makedirs(SHARED_MEDIA_ROOT, exist_ok=True)
        probe = os.path.join(SHARED_MEDIA_ROOT, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def check_ffmpeg_installed() -> bool:
    return shutil.which("ffmpeg") is not None


def check_redis_db2() -> bool:
    try:
        return bool(get_redis_client().ping())
    except Exception:
        return False


def check_postgres() -> bool:
    try:
        conn = get_db_connection()
        conn.close()
        return True
    except Exception:
        return False


@app.get("/health")
def health():
    shared_ok = check_shared_volume_writable()
    ffmpeg_ok = check_ffmpeg_installed()
    redis_ok = check_redis_db2()
    postgres_ok = check_postgres()
    all_ok = shared_ok and ffmpeg_ok and redis_ok and postgres_ok
    return {
        "status": "online" if all_ok else "degraded",
        "shared_volume_writable": shared_ok,
        "ffmpeg_installed": ffmpeg_ok,
        "redis_db2_connected": redis_ok,
        "postgres_video_machine_connected": postgres_ok,
    }


def _mark_chapter_error(chapter_id: int, message: str) -> None:
    conn = None
    try:
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chapters
                       SET status = 'Erro',
                           error_log = %s,
                           updated_at = CURRENT_TIMESTAMP
                     WHERE id = %s
                    """,
                    (message, chapter_id),
                )
    except Exception:
        pass
    finally:
        if conn is not None:
            conn.close()


@app.post("/prepare-chapter")
def prepare_chapter_endpoint(req: PrepareChapterRequest):
    if not is_valid_chapter_folder(req.chapter_folder_name):
        raise HTTPException(status_code=400, detail="chapter_folder_name inválido")
    chapter_dir = os.path.join(SHARED_MEDIA_ROOT, req.chapter_folder_name)
    raw_dir = os.path.join(chapter_dir, "raw")
    try:
        for path in (SHARED_MEDIA_MOUNT, SHARED_MEDIA_ROOT, chapter_dir, raw_dir):
            os.makedirs(path, exist_ok=True)
            os.chmod(path, 0o777)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao preparar pastas do capítulo: {e}")
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "chapter_folder_name": req.chapter_folder_name,
            "chapter_dir": chapter_dir,
            "raw_dir": raw_dir,
        },
    )


@app.post("/slice-chapter")
def slice_chapter_endpoint(req: SliceChapterRequest):
    if not is_valid_chapter_folder(req.chapter_folder_name):
        raise HTTPException(
            status_code=400,
            detail="chapter_folder_name inválido",
        )
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao conectar no PostgreSQL: {e}",
        )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM chapters WHERE id = %s", (req.chapter_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Capítulo {req.chapter_id} não encontrado",
        )
    with slice_lock:
        try:
            result = slice_chapter(req.chapter_folder_name)
        except FileNotFoundError as e:
            msg = str(e)
            _mark_chapter_error(req.chapter_id, msg)
            raise HTTPException(status_code=404, detail=msg)
        except SlicerError as e:
            msg = str(e)
            _mark_chapter_error(req.chapter_id, msg)
            raise HTTPException(status_code=400, detail=msg)
        except Exception as e:
            msg = f"Erro inesperado no corte: {e}"
            _mark_chapter_error(req.chapter_id, msg)
            raise HTTPException(status_code=500, detail=msg)
        panels = result.get("panels", [])
        chapter_dir = result["chapter_dir"]
        conn = None
        try:
            conn = get_db_connection()
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM scenes WHERE chapter_id = %s",
                        (req.chapter_id,),
                    )
                    if panels:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO scenes (chapter_id, scene_order, panel_image_path)
                            VALUES %s
                            """,
                            [
                                (
                                    req.chapter_id,
                                    p["scene_order"],
                                    p["frame_16_9_path"],
                                )
                                for p in panels
                            ],
                        )
                    cur.execute(
                        """
                        UPDATE chapters
                           SET status = 'Extração Concluída',
                               local_work_dir = %s,
                               error_log = NULL,
                               updated_at = CURRENT_TIMESTAMP
                         WHERE id = %s
                        """,
                        (chapter_dir, req.chapter_id),
                    )
        except Exception as e:
            msg = f"Falha ao gravar o corte no banco: {e}"
            _mark_chapter_error(req.chapter_id, msg)
            raise HTTPException(status_code=500, detail=msg)
        finally:
            if conn is not None:
                conn.close()
    return JSONResponse(
        status_code=200,
        content={
            "chapter_id": req.chapter_id,
            "chapter_folder_name": req.chapter_folder_name,
            "status": "Extração Concluída",
            "chapter_dir": chapter_dir,
            "total_frames": result["total_frames"],
            "total_panels": len(panels),
            "panels": [
                {
                    "scene_order": p["scene_order"],
                    "y_start": p["y_start"],
                    "y_end": p["y_end"],
                    "height": p["height"],
                    "width": p["width"],
                    "clean_path": p["clean_path"],
                    "frame_16_9_path": p["frame_16_9_path"],
                }
                for p in panels
            ],
        },
    )


@app.post("/extract-ocr")
def extract_ocr_endpoint(req: ExtractOcrRequest):
    api_key = (req.openai_api_key or OPENAI_API_KEY_ENV).strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY não configurada (env ou openai_api_key no body)",
        )
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao conectar no PostgreSQL: {e}",
        )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, local_work_dir FROM chapters WHERE id = %s",
                (req.chapter_id,),
            )
            chapter_row = cur.fetchone()
            if chapter_row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Capítulo {req.chapter_id} não encontrado",
                )
            cur.execute(
                """
                SELECT scene_order, panel_image_path
                  FROM scenes
                 WHERE chapter_id = %s
                 ORDER BY scene_order
                """,
                (req.chapter_id,),
            )
            scene_rows = cur.fetchall()
    finally:
        conn.close()
    chapter_dir = chapter_row[1]
    if not chapter_dir:
        raise HTTPException(
            status_code=400,
            detail="Capítulo sem local_work_dir. Execute /slice-chapter antes.",
        )
    if not scene_rows:
        raise HTTPException(
            status_code=400,
            detail="Nenhuma cena encontrada para o capítulo. Execute /slice-chapter antes.",
        )
    scenes = [
        {"scene_order": row[0], "frame_16_9_path": row[1]}
        for row in scene_rows
    ]
    with ocr_lock:
        try:
            result = extract_chapter_ocr(
                chapter_id=req.chapter_id,
                chapter_dir=chapter_dir,
                scenes=scenes,
                api_key=api_key,
            )
        except OcrError as e:
            msg = str(e)
            _mark_chapter_error(req.chapter_id, msg)
            raise HTTPException(status_code=502, detail=msg)
        except Exception as e:
            msg = f"Erro inesperado no OCR/Vision: {e}"
            _mark_chapter_error(req.chapter_id, msg)
            raise HTTPException(status_code=500, detail=msg)
        frames = result["frames"]
        conn = None
        try:
            conn = get_db_connection()
            with conn:
                with conn.cursor() as cur:
                    for frame in frames:
                        cur.execute(
                            """
                            UPDATE scenes
                               SET extracted_ocr_text = %s
                             WHERE chapter_id = %s
                               AND scene_order = %s
                            """,
                            (
                                frame["extracted_ocr_text"],
                                req.chapter_id,
                                frame["scene_order"],
                            ),
                        )
                    cur.execute(
                        """
                        UPDATE chapters
                           SET status = CASE
                                   WHEN status = 'Erro' THEN 'Extração Concluída'
                                   ELSE status
                               END,
                               error_log = NULL,
                               updated_at = CURRENT_TIMESTAMP
                         WHERE id = %s
                        """,
                        (req.chapter_id,),
                    )
        except Exception as e:
            msg = f"Falha ao gravar o OCR no banco: {e}"
            _mark_chapter_error(req.chapter_id, msg)
            raise HTTPException(status_code=500, detail=msg)
        finally:
            if conn is not None:
                conn.close()
    return JSONResponse(
        status_code=200,
        content={
            "chapter_id": req.chapter_id,
            "status": "Extração Concluída",
            "model": result["model"],
            "total_frames": result["total_frames"],
            "extracao_json_path": os.path.join(chapter_dir, "extracao_cap.json"),
            "frames": [
                {
                    "scene_order": f["scene_order"],
                    "clean_path": f["clean_path"],
                    "frame_16_9_path": f["frame_16_9_path"],
                    "dialogues": f["dialogues"],
                    "visual_action": f["visual_action"],
                    "extracted_ocr_text": f["extracted_ocr_text"],
                }
                for f in frames
            ],
        },
    )


@app.post("/generate-script")
def generate_script_endpoint(req: GenerateScriptRequest):
    api_key = (req.openai_api_key or OPENAI_API_KEY_ENV).strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY não configurada (env ou openai_api_key no body)",
        )
    try:
        conn = get_db_connection()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao conectar no PostgreSQL: {e}",
        )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, local_work_dir FROM chapters WHERE id = %s",
                (req.chapter_id,),
            )
            chapter_row = cur.fetchone()
            if chapter_row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Capítulo {req.chapter_id} não encontrado",
                )
            cur.execute(
                """
                SELECT scene_order, extracted_ocr_text
                  FROM scenes
                 WHERE chapter_id = %s
                 ORDER BY scene_order
                """,
                (req.chapter_id,),
            )
            scene_rows = cur.fetchall()
    finally:
        conn.close()
    chapter_dir = chapter_row[1]
    if not chapter_dir:
        raise HTTPException(
            status_code=400,
            detail="Capítulo sem local_work_dir. Execute /slice-chapter antes.",
        )
    if not scene_rows:
        raise HTTPException(
            status_code=400,
            detail="Nenhuma cena encontrada para o capítulo. Execute /slice-chapter antes.",
        )
    scenes = [
        {"scene_order": row[0], "extracted_ocr_text": row[1]}
        for row in scene_rows
    ]
    with script_lock:
        try:
            result = generate_chapter_script(
                chapter_id=req.chapter_id,
                chapter_dir=chapter_dir,
                scenes=scenes,
                api_key=api_key,
            )
        except ScriptError as e:
            msg = str(e)
            _mark_chapter_error(req.chapter_id, msg)
            raise HTTPException(status_code=502, detail=msg)
        except Exception as e:
            msg = f"Erro inesperado na geração de roteiro: {e}"
            _mark_chapter_error(req.chapter_id, msg)
            raise HTTPException(status_code=500, detail=msg)
        conn = None
        try:
            conn = get_db_connection()
            with conn:
                with conn.cursor() as cur:
                    for scene in result["scenes"]:
                        cur.execute(
                            """
                            UPDATE scenes
                               SET narration_script = %s,
                                   camera_effect = %s
                             WHERE chapter_id = %s
                               AND scene_order = %s
                            """,
                            (
                                scene["narration_script"],
                                scene["camera_effect"],
                                req.chapter_id,
                                scene["scene_order"],
                            ),
                        )
                    cur.execute(
                        """
                        UPDATE chapters
                           SET status = 'Roteiro Gerado',
                               error_log = NULL,
                               updated_at = CURRENT_TIMESTAMP
                         WHERE id = %s
                        """,
                        (req.chapter_id,),
                    )
        except Exception as e:
            msg = f"Falha ao gravar o roteiro no banco: {e}"
            _mark_chapter_error(req.chapter_id, msg)
            raise HTTPException(status_code=500, detail=msg)
        finally:
            if conn is not None:
                conn.close()
    return JSONResponse(
        status_code=200,
        content={
            "chapter_id": req.chapter_id,
            "status": "Roteiro Gerado",
            "model": result["model"],
            "total_scenes": result["total_scenes"],
            "roteiro_json_path": result["roteiro_json_path"],
            "full_script_markdown": result["full_script_markdown"],
            "scenes": result["scenes"],
        },
    )
