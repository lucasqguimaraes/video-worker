import os
import shutil
import threading

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

app = FastAPI(title="Video Machine - Slicer API", version="1.2.0")

REDIS_HOST = os.getenv("REDIS_HOST", "n8n_redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = 2

DB_HOST = os.getenv("DB_HOST", "n8n_postgres")
DB_NAME = os.getenv("DB_NAME", "video_machine")
DB_USER = os.getenv("DB_USER", "n8n")
DB_PASS = os.getenv("DB_PASS", "")

# Impede dois fatiamentos pesados simultâneos (RAM limitada na VPS KVM2)
slice_lock = threading.Lock()


class SliceChapterRequest(BaseModel):
    chapter_id: int
    chapter_folder_name: str


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


@app.post("/slice-chapter")
def slice_chapter_endpoint(req: SliceChapterRequest):
    # Validação do nome da pasta antes de tocar em disco/banco
    if not is_valid_chapter_folder(req.chapter_folder_name):
        raise HTTPException(
            status_code=400,
            detail="chapter_folder_name inválido",
        )

    # Verifica existência do capítulo
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

    # Garante um fatiamento por vez
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
