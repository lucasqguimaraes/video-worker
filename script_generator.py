# script_generator.py
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("video_machine.script_generator")

BATCH_SIZE = 25
CONTEXT_WINDOW = 3
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
REQUEST_TIMEOUT = 180
VALID_CAMERA_EFFECTS = {"zoom_in", "zoom_out", "pan_down", "pan_up"}
DEFAULT_CAMERA_EFFECT = "pan_down"
EXTRACAO_FILENAME = "extracao_cap.json"
ROTEIRO_FILENAME = "roteiro_cap.json"


class ScriptError(Exception):
    pass


SYSTEM_PROMPT = (
    "Você é o narrador em 3ª pessoa de um canal brasileiro de Manhwa/Webtoon no YouTube. "
    "Estilo: envolvente e dinâmico, com humor inteligente e sarcasmo leve diante de vilões "
    "arrogantes, e empolgação real nas lutas e nas reviravoltas.\n"
    "Regras de narração:\n"
    "1. Os balões no OCR vêm em inglês e a ação visual em português: traduza e adapte o sentido "
    "para o português do Brasil dentro da própria narração, de forma fluida. NÃO leia como "
    "audiolivro seco e não use 'ele diz:'.\n"
    "2. NÃO narre avisos de 'Warning', disclaimers, créditos de scan, marcas d'água ou links. "
    "Se o quadro for só disso, escreva uma transição neutra curta.\n"
    "3. O quadro 1 do capítulo é a abertura do vídeo: crie um hook épico e instigante nele.\n"
    "4. O último quadro do capítulo é o encerramento: crie um gancho instigante para o próximo "
    "episódio, sem citar a scan.\n"
    "5. Quadros de ação rápida/transição: 1 frase curta e ritmada (6 a 14 palavras). Quadros com "
    "diálogo importante ou momento-chave: 2 a 3 frases (15 a 40 palavras). Nunca exceda 40 palavras.\n"
    "6. Texto 100% otimizado para TTS (ElevenLabs) em português do Brasil: sem emojis, sem "
    "markdown, sem siglas estranhas e sem onomatopeias em inglês (GASP, SPLAT, THUD).\n"
    "7. Mantenha coerência com o contexto de continuidade recebido: tom, tempo verbal, nomes e "
    "linha da história. Não repita informação já narrada.\n"
    "Responda EXCLUSIVAMENTE com JSON válido no formato: "
    '{"scenes": [{"scene_order": <int>, "camera_effect": "<zoom_in|zoom_out|pan_down|pan_up>", '
    '"narration_script": "<texto PT-BR para TTS>"}]} '
    "com UMA entrada para CADA scene_order recebido, sem pular nenhum e sem inventar outros; "
    "narration_script nunca pode ser vazio."
)


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _normalize_scenes(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scenes: Dict[int, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            order = int(item.get("scene_order"))
        except (TypeError, ValueError):
            continue
        scenes[order] = _clean_text(item.get("extracted_ocr_text"))
    return [{"scene_order": o, "extracted_ocr_text": scenes[o]} for o in sorted(scenes)]


def _load_scenes_from_file(chapter_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(chapter_dir, EXTRACAO_FILENAME)
    if not os.path.exists(path):
        raise ScriptError(f"Sem cenas recebidas e {EXTRACAO_FILENAME} não encontrado em {chapter_dir}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ScriptError(f"Falha ao ler {path}: {exc}") from exc
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, list) or not frames:
        raise ScriptError(f"Chave 'frames' ausente ou vazia em {path}")
    return _normalize_scenes(frames)


def _scene_line(scene: Dict[str, Any]) -> str:
    ocr = scene["extracted_ocr_text"]
    if ocr:
        return f'- scene_order {scene["scene_order"]} | OCR: "{ocr}"'
    return (
        f'- scene_order {scene["scene_order"]} | OCR: '
        "(quadro sem texto legível; descreva o impacto visual em 1 frase curta)"
    )


def _build_user_prompt(
    batch: List[Dict[str, Any]],
    batch_index: int,
    total_batches: int,
    total_scenes: int,
    continuity: List[str],
) -> str:
    lines = [
        f"LOTE {batch_index}/{total_batches} de um episódio com {total_scenes} quadros.",
        f"Quadros deste lote: scene_order {batch[0]['scene_order']} a {batch[-1]['scene_order']}.",
        "",
    ]
    if continuity:
        lines.append(
            "CONTEXTO DE CONTINUIDADE — últimas narrações geradas no lote anterior "
            "(mantenha tom, tempo verbal e nomes; não repita o que já foi dito):"
        )
        for idx, narration in enumerate(continuity, start=1):
            lines.append(f'  {idx}. "{narration}"')
        lines.append("")
    lines.append("CENAS DESTE LOTE — gere uma narração para CADA scene_order abaixo, sem pular nenhum:")
    lines.extend(_scene_line(scene) for scene in batch)
    lines.append("")
    if batch_index == 1:
        lines.append(
            f"ATENÇÃO: o quadro {batch[0]['scene_order']} é a abertura do vídeo; "
            "crie um hook épico e instigante nele."
        )
    if batch_index == total_batches:
        lines.append(
            f"ATENÇÃO: o quadro {batch[-1]['scene_order']} é o último do episódio (créditos/fim); "
            "nele faça um encerramento instigante para o próximo episódio."
        )
    lines.append("")
    lines.append(
        "Responda EXCLUSIVAMENTE com JSON válido: "
        '{"scenes": [{"scene_order": <int>, "camera_effect": "<zoom_in|zoom_out|pan_down|pan_up>", '
        '"narration_script": "<texto PT-BR para TTS>"}]}'
    )
    return "\n".join(lines)


def _call_openai(api_key: str, user_prompt: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.65,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    last_error: Any = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                try:
                    return resp.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, ValueError) as exc:
                    raise ScriptError(f"Resposta inesperada da OpenAI: {exc}") from exc
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_error = ScriptError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            else:
                raise ScriptError(f"HTTP {resp.status_code} (sem retry): {resp.text[:500]}")
        except requests.RequestException as exc:
            last_error = exc
        logger.warning("OpenAI tentativa %d/%d falhou: %s", attempt, MAX_RETRIES, last_error)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise ScriptError(f"OpenAI falhou após {MAX_RETRIES} tentativas: {last_error}")


def _parse_batch(content: str) -> List[Dict[str, Any]]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ScriptError("Resposta da OpenAI não contém JSON válido.") from None
        data = json.loads(match.group(0))
    entries = data.get("scenes") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ScriptError("JSON da OpenAI não contém lista em 'scenes'.")
    return [entry for entry in entries if isinstance(entry, dict)]


_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE0F"
    "]+"
)


def _clean_narration(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = _EMOJI_RE.sub(" ", value)
    text = re.sub(r"[*#_`>]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip('"“” ').strip()


_FALLBACK_NARRATIONS = (
    "A cena avança sem dar trégua e o ritmo não deixa ninguém respirar.",
    "Sem tempo de hesitar: o próximo movimento já está em jogo.",
    "O quadro fala por si e a tensão no ar continua subindo.",
    "Um detalhe silencioso aqui pode mudar todo o rumo da história.",
    "Entre um golpe e outro, nada nesse episódio desacelera.",
    "E a pressão aumenta, porque aqui ninguém tem espaço para vacilar.",
)


def _fallback_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    order = scene["scene_order"]
    return {
        "scene_order": order,
        "camera_effect": DEFAULT_CAMERA_EFFECT,
        "narration_script": _FALLBACK_NARRATIONS[order % len(_FALLBACK_NARRATIONS)],
    }


def _validate_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        order = int(entry.get("scene_order"))
    except (TypeError, ValueError):
        return None
    narration = _clean_narration(entry.get("narration_script"))
    if not narration:
        return None
    camera = entry.get("camera_effect")
    if camera not in VALID_CAMERA_EFFECTS:
        camera = DEFAULT_CAMERA_EFFECT
    return {"scene_order": order, "camera_effect": camera, "narration_script": narration}


def _process_batch(
    api_key: str,
    batch: List[Dict[str, Any]],
    batch_index: int,
    total_batches: int,
    total_scenes: int,
    continuity: List[str],
) -> List[Dict[str, Any]]:
    prompt = _build_user_prompt(batch, batch_index, total_batches, total_scenes, continuity)
    content = _call_openai(api_key, prompt)
    entries = _parse_batch(content)
    by_order: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        valid = _validate_entry(entry)
        if valid is not None:
            by_order[valid["scene_order"]] = valid
    return [
        by_order.get(scene["scene_order"]) or _fallback_scene(scene)
        for scene in batch
    ]


def _build_markdown(scenes: List[Dict[str, Any]]) -> str:
    return "".join(
        f"[Cena {scene['scene_order']:03d} | Efeito: {scene['camera_effect']}]\n"
        f"{scene['narration_script']}\n\n"
        for scene in scenes
    )


def generate_chapter_script(
    chapter_id: int,
    chapter_dir: str,
    scenes: List[Dict[str, Any]],
    api_key: str,
) -> Dict[str, Any]:
    if not api_key:
        raise ScriptError("api_key da OpenAI ausente.")
    if not scenes:
        scenes = _load_scenes_from_file(chapter_dir)
    scenes = _normalize_scenes(scenes)
    if not scenes:
        raise ScriptError(f"Nenhuma cena válida para o capítulo {chapter_id}.")

    total_batches = (len(scenes) + BATCH_SIZE - 1) // BATCH_SIZE
    all_scenes: List[Dict[str, Any]] = []
    continuity: List[str] = []

    for index in range(total_batches):
        batch = scenes[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
        logger.info(
            "Capítulo %d: lote %d/%d (%d quadros)",
            chapter_id, index + 1, total_batches, len(batch),
        )
        batch_results = _process_batch(
            api_key, batch, index + 1, total_batches, len(scenes), continuity
        )
        all_scenes.extend(batch_results)
        continuity = [s["narration_script"] for s in batch_results[-CONTEXT_WINDOW:]]

    os.makedirs(chapter_dir, exist_ok=True)
    output_path = os.path.join(chapter_dir, ROTEIRO_FILENAME)
    payload = {
        "chapter_id": chapter_id,
        "model": OPENAI_MODEL,
        "total_scenes": len(all_scenes),
        "roteiro_json_path": output_path,
        "full_script_markdown": _build_markdown(all_scenes),
        "scenes": all_scenes,
    }

    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, output_path)
    except OSError as exc:
        raise ScriptError(f"Falha ao salvar {output_path}: {exc}") from exc

    logger.info(
        "Capítulo %d: roteiro salvo em %s (%d cenas)", chapter_id, output_path, len(all_scenes)
    )
    return payload
