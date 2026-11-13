# tts_generator.py
import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List

import requests

logger = logging.getLogger("video_machine.tts_generator")

AUDIO_DIR_NAME = "audio"
MANIFEST_FILENAME = "audio_manifest.json"
ROTEIRO_FILENAME = "roteiro_cap.json"
MIN_VALID_AUDIO_BYTES = 500
MIN_DURATION_MS = 100
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
REQUEST_TIMEOUT = 120
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
ELEVENLABS_DEFAULT_VOICE = "pNInz6obpgDQGcFmaJgB"
SUPPORTED_PROVIDERS = ("openai", "voicestudio", "elevenlabs")


class TtsError(Exception):
    pass


def _audio_path(chapter_dir: str, scene_order: int) -> str:
    return os.path.join(chapter_dir, AUDIO_DIR_NAME, f"audio_{scene_order:03d}.mp3")


def _load_scenes_from_roteiro(chapter_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(chapter_dir, ROTEIRO_FILENAME)
    if not os.path.exists(path):
        raise TtsError(f"Sem cenas recebidas e {ROTEIRO_FILENAME} não encontrado em {chapter_dir}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise TtsError(f"Falha ao ler {path}: {exc}") from exc
    scenes = data.get("scenes") if isinstance(data, dict) else None
    if not isinstance(scenes, list) or not scenes:
        raise TtsError(f"Chave 'scenes' ausente ou vazia em {path}")
    return scenes


def _build_request(
    provider: str, text: str, api_key: str, voice_id: str, base_url: str
) -> tuple:
    if provider == "openai":
        url = "https://api.openai.com/v1/audio/speech"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "tts-1",
            "voice": voice_id or "onyx",
            "input": text,
            "response_format": "mp3",
        }
    elif provider == "voicestudio":
        if not base_url:
            raise TtsError("base_url é obrigatória para o provedor voicestudio.")
        url = f"{base_url.rstrip('/')}/v1/audio/speech"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": "tts-1",
            "voice": voice_id or "default",
            "input": text,
            "response_format": "mp3",
        }
    elif provider == "elevenlabs":
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id or ELEVENLABS_DEFAULT_VOICE}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
    else:
        raise TtsError(f"Provedor de TTS não suportado: {provider}")
    return url, headers, payload


def _synthesize(
    provider: str, text: str, api_key: str, voice_id: str, base_url: str
) -> bytes:
    url, headers, payload = _build_request(provider, text, api_key, voice_id, base_url)
    last_error: Any = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and resp.content:
                return resp.content
            if resp.status_code in RETRYABLE_STATUS or resp.status_code == 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            else:
                raise TtsError(f"HTTP {resp.status_code} sem retry ({provider}): {resp.text[:500]}")
        except requests.RequestException as exc:
            last_error = exc
        logger.warning("TTS (%s) tentativa %d/%d falhou: %s", provider, attempt, MAX_RETRIES, last_error)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise TtsError(f"Provedor {provider} falhou após {MAX_RETRIES} tentativas: {last_error}")


def _write_atomic(path: str, data: bytes) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as fh:
        fh.write(data)
    os.replace(tmp_path, path)


def _get_audio_duration_ms(audio_path: str) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        seconds = float(proc.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        raise TtsError(f"ffprobe falhou em {audio_path}: {exc}") from exc
    return max(MIN_DURATION_MS, int(round(seconds * 1000)))


def _write_manifest(chapter_dir: str, manifest: Dict[str, Any]) -> str:
    path = os.path.join(chapter_dir, MANIFEST_FILENAME)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def generate_chapter_audio(
    chapter_id: int,
    chapter_dir: str,
    scenes: List[Dict[str, Any]],
    provider: str = "openai",
    api_key: str = "",
    voice_id: str = "",
    base_url: str = "",
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    provider = (provider or "").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise TtsError(f"Provedor de TTS não suportado: {provider}")

    source = scenes or _load_scenes_from_roteiro(chapter_dir)
    ordered = sorted(source, key=lambda s: int(s["scene_order"]))
    if not ordered:
        raise TtsError(f"Nenhuma cena disponível para o capítulo {chapter_id}.")

    audio_dir = os.path.join(chapter_dir, AUDIO_DIR_NAME)
    os.makedirs(audio_dir, exist_ok=True)

    timeline: List[Dict[str, Any]] = []
    current_time_ms = 0

    for scene in ordered:
        scene_order = int(scene["scene_order"])
        text = (scene.get("narration_script") or "").strip()
        path = _audio_path(chapter_dir, scene_order)

        reuse = (
            not force_regenerate
            and os.path.exists(path)
            and os.path.getsize(path) > MIN_VALID_AUDIO_BYTES
        )
        if reuse:
            logger.info("Cena %d: reaproveitando áudio existente.", scene_order)
        else:
            if not text:
                raise TtsError(f"Cena {scene_order} sem narration_script.")
            logger.info("Cena %d: sintetizando via %s.", scene_order, provider)
            audio_bytes = _synthesize(provider, text, api_key, voice_id, base_url)
            _write_atomic(path, audio_bytes)

        duration_ms = _get_audio_duration_ms(path)
        start_time_ms = current_time_ms
        end_time_ms = start_time_ms + duration_ms
        current_time_ms = end_time_ms

        timeline.append({
            "scene_order": scene_order,
            "audio_path": path,
            "duration_ms": duration_ms,
            "start_time_ms": start_time_ms,
            "end_time_ms": end_time_ms,
        })

    total_duration_ms = current_time_ms
    manifest = {
        "chapter_id": chapter_id,
        "provider": provider,
        "voice_id": voice_id,
        "total_scenes": len(timeline),
        "total_duration_ms": total_duration_ms,
        "total_duration_seconds": round(total_duration_ms / 1000, 3),
        "manifest_json_path": os.path.join(chapter_dir, MANIFEST_FILENAME),
        "scenes": timeline,
    }
    _write_manifest(chapter_dir, manifest)
    logger.info(
        "Capítulo %d: %d áudios, duração total %d ms.",
        chapter_id, len(timeline), total_duration_ms,
    )
    return manifest
