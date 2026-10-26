"""Engine-native vision for DeepSeek-V4.1: pictures the dsv41 engine encodes itself.

The dsv41 engine (src/dsv41/ds4_vision.cu) carries the model's own vision tower, packed from
the mmproj GGUF (tools/dsv41/pack_gguf.py --mmproj).  A GENI request to it names a DSI1
pictures file - raw RGB and the prompt offset where each picture's span starts - and the
engine encodes the pictures itself.  The separate strata-vision encoder (llama.cpp mtmd) has
no deepseek41v projector, so this family does not use it at all.

A picture reaches the prompt as a run of its image token (<|deepseek_image|>, the
checkpoint's image_token_id): the tower cuts the picture into 14-pixel patches, merges 3 x 3
of them and writes one prompt token per merged patch plus one separator per row and two for
the block, rows = ceil(h/42) * (ceil(w/42) + 1) + 2, at most vision.max_image_tokens (1024).
"""
from __future__ import annotations

import base64
import io
import re
import struct
import tempfile
import urllib.request
from pathlib import Path

PATCH = 14
MERGE = 3
MAX_TOKENS = 1024
MARKER = "\x01strata-image\x01"          # where a picture sits in the prompt before tokenizing
IMAGE_TEXT = "<\uff5cdeepseek_image\uff5c>"   # the model's image token as text


def detect(cfg: dict) -> dict | None:
    """The pack's vision tower as the engine describes it (ds4.txt), or None when the pack has none."""
    args = cfg.get("args") or []
    if "--pack" not in args:
        return None
    try:
        pack = Path(args[args.index("--pack") + 1])
        text = (pack / "ds4.txt").read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    tok = re.search(r"^image_token_id\s+(\d+)\s*$", text, re.M)
    if not tok:
        return None
    geo = {k: int(v) for k, v in re.findall(r"^vision\.(\w+)\s+(\d+)\s*$", text, re.M)}
    return {"token": int(tok.group(1)),
            "patch": geo.get("patch_size") or PATCH,
            "merge": geo.get("downsample_ratio") or MERGE,
            "max_tokens": geo.get("max_image_tokens") or MAX_TOKENS}


def rows(width: int, height: int, geo: dict) -> int:
    """The prompt tokens one picture takes in this tower."""
    step = (geo.get("patch") or PATCH) * (geo.get("merge") or MERGE)
    return -(-height // step) * (-(-width // step) + 1) + 2


def fit(width: int, height: int, geo: dict) -> tuple[int, int]:
    """The size the tower takes: both sides on the patch grid, the aspect kept, the span inside max_tokens."""
    step = (geo.get("patch") or PATCH) * (geo.get("merge") or MERGE)
    limit = geo.get("max_tokens") or MAX_TOKENS
    w, h = max(step, int(width)), max(step, int(height))
    for _ in range(96):
        if rows(w, h, geo) <= limit:
            return (w // step) * step, (h // step) * step
        w, h = int(w * 0.9), int(h * 0.9)
    return max(step, (w // step) * step), max(step, (h // step) * step)


def read_source(source: str) -> bytes:
    """One picture's bytes: a data: URL, an http(s) URL or a local path."""
    if source.startswith("data:"):
        _, _, payload = source.partition(",")
        return base64.b64decode(payload)
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as r:
            return r.read(64 << 20)
    return Path(source).read_bytes()


def prepare(source: str, geo: dict) -> tuple[int, int, bytes]:
    """(width, height, RGB) at the size the tower takes."""
    from PIL import Image
    im = Image.open(io.BytesIO(read_source(source))).convert("RGB")
    w, h = fit(im.width, im.height, geo)
    if (im.width, im.height) != (w, h):
        im = im.resize((w, h), Image.LANCZOS)
    return w, h, im.tobytes()


def flatten(messages: list, geo: dict) -> tuple[list, list]:
    """Content lists -> text with one MARKER per picture (in order); returns (messages, pictures)."""
    pics: list = []
    out: list = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        text = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image" and part.get("source"):
                pics.append(prepare(part["source"], geo))
                text.append(MARKER)
            elif part.get("type") == "text":
                text.append(part.get("text") or "")
        msg = dict(m)
        msg["content"] = "".join(text)
        out.append(msg)
    return out, pics


def expand(prompt: str, row_counts: list) -> str:
    """Replace each MARKER with that picture's run of image tokens."""
    parts = prompt.split(MARKER)
    if len(parts) - 1 != len(row_counts):
        raise ValueError("the prompt and its pictures do not match (%d markers, %d pictures)"
                         % (len(parts) - 1, len(row_counts)))
    out = [parts[0]]
    for n, tail in zip(row_counts, parts[1:]):
        out.append(IMAGE_TEXT * n)
        out.append(tail)
    return "".join(out)


def starts(ids: list, token: int, row_counts: list) -> list:
    """The prompt offset of every picture's run of image tokens, in order."""
    out, i = [], 0
    for n in row_counts:
        while i < len(ids) and ids[i] != token:
            i += 1
        if i + n > len(ids):
            raise ValueError("the prompt holds fewer image tokens than its pictures need")
        out.append(i)
        i += n
    return out


def write_dsi1(pics: list, starts_: list, directory=None) -> Path:
    """A DSI1 pictures file: magic, count, then (start i64, width i32, height i32, RGB) per picture."""
    folder = Path(tempfile.mkdtemp(prefix="strata-dsv41-vision-", dir=directory))
    path = folder / "pictures.dsi1"
    with open(path, "wb") as f:
        f.write(b"DSI1")
        f.write(struct.pack("<i", len(pics)))
        for (w, h, rgb), start in zip(pics, starts_):
            f.write(struct.pack("<q", int(start)))
            f.write(struct.pack("<i", int(w)))
            f.write(struct.pack("<i", int(h)))
            f.write(rgb)
    return path
