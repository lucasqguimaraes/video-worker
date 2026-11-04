# ocr_vision.py
import base64
import io
import json
import os
import re
import time
from typing import Any, Dict, List

import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"
MAX_WIDTH = 800
JPEG_QUALITY = 85
REQUEST_TIMEOUT = 120
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "Você analisa UM quadro de quadrinho/manhwa para um pipeline de vídeo. "
    "Responda APENAS com um JSON válido, sem markdown, com exatamente estas chaves:\n"
    '{"dialogues": ["fala 1", "fala 2"], "visual_action": "descrição curta"}\n\n'
    "Regras:\n"
    "- dialogues: textos de balões de fala, pensamento, narração e caixas de legenda do quadro, "
    "transcritos literalmente, ordenados de cima para baixo (e da esquerda para a direita dentro "
    "da mesma faixa horizontal). Cada balão é um item separado. Se não houver texto, use [].\n"
    "- Ignore marcas d'água, logotipos, nomes de sites, créditos de tradutor/scan e numeração de página.\n"
    "- Não invente texto ilegível; omita balões que não consiga ler com segurança.\n"
    "- visual_action: 1 a 2 frases curtas em português descrevendo o que acontece visualmente "
    "(personagens, expressões, golpes, movimento, cenário). Mesmo sem falas (cenas de luta mudas), "
    "descreva a ação.\n"
    "- Não escreva nenhum texto fora do JSON."
)


class OcrError(Exception):
    """Erro controlado na extração de OCR/Vision."""


def panel_clean_path(chapter_dir: str, scene_order: int) -> str:
    return os.path.join(chapter_dir, "clean", f"panel_{int(scene_order):03d}.jpg")


def image_to_base64_jpeg(path: str) -> str:
    """Redimensiona para largura máxima de 800px e converte para JPEG q85 em memória."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.width > MAX_WIDTH:
            ratio = MAX_WIDTH / float(img.width)
            new_height = max(1, int(img.height * ratio))
            img = img.resize((MAX_WIDTH, new_height), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)

    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _normalize_result(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise OcrError("Resposta do modelo não é um objeto JSON")

    raw_dialogues = data.get("dialogues", [])
    if isinstance(raw_dialogues, str):
        raw_dialogues = [raw_dialogues]
    if not isinstance(raw_dialogues, list):
        raw_dialogues = []

    dialogues: List[str] = []
    for item in raw_dialogues:
        text = str(item).strip()
        if text:
            dialogues.append(text)

    raw_visual = data.get("visual_action", "")
    visual_action = str(raw_visual).strip() if raw_visual is not None else ""

    return {"dialogues": dialogues, "visual_action": visual_action}


def _parse_model_content(content: str) -> Dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise OcrError("Conteúdo do modelo sem JSON válido")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise OcrError(f"JSON inválido do modelo: {e}")

    return _normalize_result(data)


def call_vision(api_key: str, image_b64: str) -> Dict[str, Any]:
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.1,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analise este quadro e retorne apenas o JSON."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = "erro desconhecido"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(
                OPENAI_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            last_error = f"falha de rede: {e}"
        else:
            status = resp.status_code

            if status == 200:
                try:
                    content = resp.json()["choices"][0]["message"]["content"]
                    return _parse_model_content(content)
                except (KeyError, IndexError, TypeError, ValueError, OcrError) as e:
                    last_error = f"resposta inválida: {e}"

            elif status in (401, 403):
                raise OcrError("OpenAI recusou a chave de API (HTTP %d)" % status)

            elif status in (429, 500, 502, 503, 504):
                last_error = f"HTTP {status}"

            else:
                raise OcrError(f"Erro HTTP {status} da OpenAI: {resp.text[:300]}")

        if attempt < MAX_ATTEMPTS:
            time.sleep(2 ** attempt)

    raise OcrError(f"Falha após {MAX_ATTEMPTS} tentativas: {last_error}")


def build_consolidated_text(visual_action: str, dialogues: List[str]) -> str:
    action_text = visual_action.strip() if visual_action else "(não identificada)"
    falas_text = " / ".join(dialogues) if dialogues else "(sem falas)"
    return f"Ação: {action_text} | Falas: {falas_text}"


def _write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def extract_chapter_ocr(
    chapter_id: int,
    chapter_dir: str,
    scenes: List[Dict[str, Any]],
    api_key: str,
) -> Dict[str, Any]:
    """
    scenes: lista de dicts com 'scene_order' e 'frame_16_9_path' (ou 'panel_image_path').
    Retorna o payload completo também salvo em {chapter_dir}/extracao_cap.json.
    """
    if not os.path.isdir(chapter_dir):
        raise OcrError(f"Diretório do capítulo não encontrado: {chapter_dir}")

    frames: List[Dict[str, Any]] = []

    for scene in sorted(scenes, key=lambda s: int(s["scene_order"])):
        scene_order = int(scene["scene_order"])
        clean_path = panel_clean_path(chapter_dir, scene_order)

        if not os.path.isfile(clean_path):
            raise OcrError(
                f"Arquivo ausente: clean/panel_{scene_order:03d}.jpg"
            )

        image_b64 = image_to_base64_jpeg(clean_path)
        result = call_vision(api_key, image_b64)

        dialogues = result["dialogues"]
        visual_action = result["visual_action"]
        frame_16_9 = scene.get("frame_16_9_path") or scene.get("panel_image_path")

        frames.append(
            {
                "scene_order": scene_order,
                "clean_path": clean_path,
                "frame_16_9_path": frame_16_9,
                "dialogues": dialogues,
                "visual_action": visual_action,
                "extracted_ocr_text": build_consolidated_text(visual_action, dialogues),
            }
        )

    payload = {
        "chapter_id": chapter_id,
        "model": OPENAI_MODEL,
        "total_frames": len(frames),
        "frames": frames,
    }

    _write_json_atomic(os.path.join(chapter_dir, "extracao_cap.json"), payload)

    return payload
