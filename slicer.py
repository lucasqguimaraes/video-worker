"""
slicer.py - Fatiador inteligente de Webtoon/Manhwa com geração de canvas 16:9.

Fluxo:
  1. Lê as imagens de {SHARED_MEDIA_ROOT}/video_machine/{chapter_folder_name}/raw
     em ordem natural (img_2 antes de img_10).
  2. Normaliza a largura (referência = primeira imagem) e faz o stitch vertical
     em um único array uint8 pré-alocado, sem cópias intermediárias.
  3. Localiza calhas (gutters) entre MIN_PANEL_H e MAX_PANEL_H, priorizando faixas
     de cor sólida e evitando regiões de alto gradiente (balões, traços, rostos).
  4. Descarta recortes com altura < MIN_KEEP_H ou praticamente vazios.
  5. Salva:
       clean/panel_NNN.jpg        -> quadro vertical puro (para OCR/Vision)
       frames_16_9/frame_NNN.jpg  -> canvas 1920x1080 com fundo desfocado
"""
from __future__ import annotations

import logging
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

# Remove a trava de decompressão do Pillow: webtoons podem ter dezenas de milhares de px de altura.
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger("slicer")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
SHARED_MEDIA_ROOT = os.getenv("SHARED_MEDIA_ROOT", "/shared_media")
PROJECT_DIR_NAME = "video_machine"
RAW_DIR_NAME = "raw"
CLEAN_DIR_NAME = "clean"
FRAMES_DIR_NAME = "frames_16_9"

MIN_PANEL_H = 750            # altura mínima entre cortes
MAX_PANEL_H = 1650           # altura máxima entre cortes
EDGE_IGNORE_RATIO = 0.05     # 5% de cada lateral ignorados (molduras)
MIN_KEEP_H = 250             # recortes menores são descartados
SOLID_STD_THRESHOLD = 6.0    # desvio padrão (0-255) abaixo do qual a linha é "cor sólida"
SMOOTH_WINDOW = 9            # suavização vertical do score
CENTER_PENALTY_WEIGHT = 0.15 # leve preferência por cortes no centro da janela
NON_SOLID_PENALTY = 1.0      # penalidade para linhas não sólidas
MAX_STRIP_BYTES = int(os.getenv("MAX_STRIP_BYTES", str(2_500 * 1024**2)))

CANVAS_W, CANVAS_H = 1920, 1080
FOREGROUND_RATIO = 0.90      # quadro ocupa 90% do canvas (5% de margem em cada lado)
BACKGROUND_BRIGHTNESS = 0.38 # ~35-40% de brilho no fundo
BG_PROBE_W, BG_PROBE_H = 480, 270
JPEG_QUALITY = 92
BLANK_RATIO = 0.995          # >=99.5% pixels brancos/pretos = quadro vazio

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_FOLDER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ \-]{0,199}")


class SlicerError(Exception):
    """Erro de processamento/regra de negócio do fatiador."""


@dataclass
class PanelResult:
    scene_order: int
    y_start: int
    y_end: int
    height: int
    width: int
    clean_path: str
    frame_16_9_path: str


# ---------------------------------------------------------------------------
# Utilidades de arquivo
# ---------------------------------------------------------------------------
def is_valid_chapter_folder(name: str) -> bool:
    """Impede path traversal e caracteres perigosos no nome da pasta."""
    return bool(name) and ".." not in name and _FOLDER_RE.fullmatch(name) is not None


def natural_sort_key(filename: str) -> List[Any]:
    """img_2.jpg vem antes de img_10.jpg. Partes pares são texto, ímpares são números."""
    parts = re.split(r"(\d+)", filename.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def list_raw_images(raw_dir: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(raw_dir):
        if name.startswith("."):
            continue
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(raw_dir, name))
    files.sort(key=lambda p: natural_sort_key(os.path.basename(p)))
    return files


def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


# ---------------------------------------------------------------------------
# Stitch
# ---------------------------------------------------------------------------
def build_strip(image_paths: List[str]) -> np.ndarray:
    """Normaliza a largura para a da primeira imagem e empilha verticalmente."""
    sizes: List[Tuple[int, int]] = []
    for p in image_paths:
        with Image.open(p) as im:
            sizes.append(im.size)  # (w, h), lido apenas do cabeçalho

    ref_w = sizes[0][0]
    if ref_w < 16:
        raise SlicerError(f"Largura de referência inválida: {ref_w}px")

    heights = [max(1, round(h * ref_w / w)) for w, h in sizes]
    total_h = sum(heights)

    needed = total_h * ref_w * 3
    if needed > MAX_STRIP_BYTES:
        raise SlicerError(
            f"Tira resultante muito grande ({needed / 1024**2:.0f} MB). "
            f"Limite: {MAX_STRIP_BYTES / 1024**2:.0f} MB. Divida o capítulo em partes."
        )

    strip = np.empty((total_h, ref_w, 3), dtype=np.uint8)
    y = 0
    for path, h in zip(image_paths, heights):
        img = _load_rgb(path)
        if img.shape[1] != ref_w or img.shape[0] != h:
            interp = cv2.INTER_AREA if img.shape[1] > ref_w else cv2.INTER_CUBIC
            img = cv2.resize(img, (ref_w, h), interpolation=interp)
        strip[y:y + h] = img
        y += h
        del img

    logger.info("Tira montada: %dx%d a partir de %d imagens", ref_w, total_h, len(image_paths))
    return strip


# ---------------------------------------------------------------------------
# Detecção de calhas
# ---------------------------------------------------------------------------
def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    return np.convolve(padded, np.ones(k, dtype=np.float32) / k, mode="valid")


def _find_cut(strip: np.ndarray, start: int, width: int) -> int:
    """
    Escolhe a linha de corte dentro da janela [start+MIN, start+MAX].
    Score combinado (menor = melhor):
      - desvio padrão horizontal (linhas sólidas têm std baixo)
      - gradiente horizontal médio (linhas sem bordas/traços)
      - penalidade para linhas não sólidas
      - leve preferência pelo centro da janela
    """
    height = strip.shape[0]
    lo = start + MIN_PANEL_H
    # Garante que o restante após o corte também tenha pelo menos MIN_PANEL_H.
    hi = min(start + MAX_PANEL_H, height - MIN_PANEL_H)
    if hi - lo < 10:
        # Sem janela válida: divide o restante ao meio (ambas as partes >= MIN_PANEL_H).
        return start + (height - start) // 2

    left = int(width * EDGE_IGNORE_RATIO)
    right = width - left
    region = np.ascontiguousarray(strip[lo:hi, left:right])
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(np.float32)

    row_std = gray.std(axis=1)
    row_grad = np.abs(np.diff(gray, axis=1)).mean(axis=1)

    std_n = row_std / (row_std.max() + 1e-6)
    grad_n = row_grad / (row_grad.max() + 1e-6)
    raw_score = 0.5 * std_n + 0.5 * grad_n

    score_s = _smooth(raw_score, SMOOTH_WINDOW)
    std_s = _smooth(row_std, SMOOTH_WINDOW)

    n = score_s.shape[0]
    idx = np.arange(n, dtype=np.float32)
    center = (n - 1) / 2.0
    center_penalty = CENTER_PENALTY_WEIGHT * np.abs(idx - center) / max(center, 1.0)
    solid_penalty = np.where(std_s > SOLID_STD_THRESHOLD, NON_SOLID_PENALTY, 0.0)

    total = score_s + center_penalty + solid_penalty
    best = int(np.argmin(total))
    return lo + best


def compute_segments(strip: np.ndarray, width: int) -> List[Tuple[int, int]]:
    """Percorre a tira e devolve os intervalos (y_start, y_end) de cada quadro."""
    height = strip.shape[0]
    segments: List[Tuple[int, int]] = []
    cur = 0
    while cur < height:
        remaining = height - cur
        if remaining <= MAX_PANEL_H:
            segments.append((cur, height))
            break
        cut = _find_cut(strip, cur, width)
        if not (cur < cut < height):  # salvaguarda contra corte inválido
            cut = cur + remaining // 2
        segments.append((cur, cut))
        cur = cut
    return segments


def _is_blank(gray: np.ndarray) -> bool:
    total = gray.size
    white = np.count_nonzero(gray >= 240) / total
    black = np.count_nonzero(gray <= 15) / total
    return white >= BLANK_RATIO or black >= BLANK_RATIO


# ---------------------------------------------------------------------------
# Canvas 16:9
# ---------------------------------------------------------------------------
def build_16_9_frame(panel: np.ndarray) -> np.ndarray:
    """
    Fundo: o próprio quadro em cover 1920x1080, desfocado e com brilho reduzido.
    Frente: o quadro original centralizado, com 5% de margem em cada lado.
    """
    ph, pw = panel.shape[:2]

    # Fundo calculado em resolução reduzida (barato em CPU) e depois ampliado.
    s = max(BG_PROBE_W / pw, BG_PROBE_H / ph)
    bw = max(BG_PROBE_W, math.ceil(pw * s))
    bh = max(BG_PROBE_H, math.ceil(ph * s))
    bg = cv2.resize(panel, (bw, bh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x0 = (bw - BG_PROBE_W) // 2
    y0 = (bh - BG_PROBE_H) // 2
    bg = np.ascontiguousarray(bg[y0:y0 + BG_PROBE_H, x0:x0 + BG_PROBE_W])
    bg = cv2.GaussianBlur(bg, (31, 31), 0)
    bg = cv2.resize(bg, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_LINEAR)
    canvas = np.clip(bg.astype(np.float32) * BACKGROUND_BRIGHTNESS, 0, 255).astype(np.uint8)

    # Frente: encaixa o quadro dentro de 90% do canvas, preservando proporção.
    max_w = int(CANVAS_W * FOREGROUND_RATIO)
    max_h = int(CANVAS_H * FOREGROUND_RATIO)
    scale = min(max_w / pw, max_h / ph)
    fw = max(1, round(pw * scale))
    fh = max(1, round(ph * scale))
    fg = cv2.resize(panel, (fw, fh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x0 = (CANVAS_W - fw) // 2
    y0 = (CANVAS_H - fh) // 2
    canvas[y0:y0 + fh, x0:x0 + fw] = fg
    return canvas


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def slice_chapter(chapter_folder_name: str, media_root: str = SHARED_MEDIA_ROOT) -> Dict[str, Any]:
    if not is_valid_chapter_folder(chapter_folder_name):
        raise SlicerError(f"Nome de pasta inválido: {chapter_folder_name!r}")

    chapter_dir = os.path.join(media_root, PROJECT_DIR_NAME, chapter_folder_name)
    raw_dir = os.path.join(chapter_dir, RAW_DIR_NAME)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Pasta raw não encontrada: {raw_dir}")

    image_paths = list_raw_images(raw_dir)
    if not image_paths:
        raise SlicerError(f"Nenhuma imagem suportada em {raw_dir}")

    clean_dir = os.path.join(chapter_dir, CLEAN_DIR_NAME)
    frames_dir = os.path.join(chapter_dir, FRAMES_DIR_NAME)
    for d in (clean_dir, frames_dir):  # remove saídas antigas para não misturar execuções
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    strip = build_strip(image_paths)
    strip_h, strip_w = strip.shape[:2]
    segments = compute_segments(strip, strip_w)

    panels: List[PanelResult] = []
    discarded = 0
    scene_order = 0

    for y0, y1 in segments:
        seg = strip[y0:y1]
        seg_h = y1 - y0
        if seg_h < MIN_KEEP_H or _is_blank(cv2.cvtColor(seg, cv2.COLOR_RGB2GRAY)):
            discarded += 1
            continue

        scene_order += 1
        clean_path = os.path.join(clean_dir, f"panel_{scene_order:03d}.jpg")
        frame_path = os.path.join(frames_dir, f"frame_{scene_order:03d}.jpg")

        Image.fromarray(seg).save(clean_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        frame = build_16_9_frame(seg)
        Image.fromarray(frame).save(frame_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        del frame

        panels.append(
            PanelResult(
                scene_order=scene_order,
                y_start=y0,
                y_end=y1,
                height=seg_h,
                width=strip_w,
                clean_path=clean_path,
                frame_16_9_path=frame_path,
            )
        )

    del strip

    logger.info(
        "Capítulo %s: %d quadros gerados, %d descartados (de %d segmentos)",
        chapter_folder_name, len(panels), discarded, len(segments),
    )

    return {
        "chapter_folder_name": chapter_folder_name,
        "chapter_dir": chapter_dir,
        "source_images": len(image_paths),
        "strip_size": {"width": strip_w, "height": strip_h},
        "segments_detected": len(segments),
        "discarded_segments": discarded,
        "total_frames": len(panels),
        "frame_resolution": f"{CANVAS_W}x{CANVAS_H}",
        "panels": [asdict(p) for p in panels],
    }
