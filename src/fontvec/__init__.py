import argparse
import json
import sys
from pathlib import Path

import torch
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModel

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
SYMBOLS = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
REQUIRED = LETTERS + DIGITS
CHARS = LETTERS + DIGITS + SYMBOLS

GRID = 10
CELL = 44
INPUT_SIZE = 448
FONT_SIZE = 30
BASELINE = 32

MODEL_NAME = "facebook/dinov2-small"


def check_font(path):
    font = TTFont(path)
    try:
        cmap = font.getBestCmap()
        if not cmap:
            raise ValueError("no usable cmap")
        missing_required = [c for c in REQUIRED if ord(c) not in cmap]
        if missing_required:
            raise ValueError("missing glyphs: " + " ".join(missing_required))
        missing_symbols = [c for c in SYMBOLS if ord(c) not in cmap]
        family = font["name"].getDebugName(1) or ""
        style = font["name"].getDebugName(2) or ""
        return family, style, set(cmap), missing_symbols
    finally:
        font.close()


def render(path, cmap):
    img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), "white")
    draw = ImageDraw.Draw(img)
    pil_font = ImageFont.truetype(str(path), FONT_SIZE)
    for i, ch in enumerate(CHARS):
        if ord(ch) not in cmap:
            continue  # never draw .notdef/tofu
        col, row = i % GRID, i // GRID
        advance = draw.textlength(ch, font=pil_font)
        x = col * CELL + (CELL - advance) / 2
        y = row * CELL + BASELINE
        draw.text((x, y), ch, font=pil_font, fill="black", anchor="ls")
    return img


def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    return processor, model


@torch.no_grad()
def embed(img, processor, model):
    inputs = processor(
        images=img, return_tensors="pt", do_resize=False, do_center_crop=False
    )
    cls = model(**inputs).last_hidden_state[:, 0].squeeze(0)
    cls = cls / cls.norm()  # L2-normalize for cosine similarity
    return cls.tolist()


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="fontvec",
        usage="fontvec [path/to/fonts] [--save-images DIR]",
    )
    ap.add_argument("folder", type=Path, help="folder containing .ttf files")
    ap.add_argument(
        "--save-images",
        type=Path,
        metavar="DIR",
        help="save each rendered grid as DIR/<fontfile>.png",
    )
    args = ap.parse_args()

    folder: Path = args.folder
    if not folder.is_dir():
        print(f"error: {folder} is not a directory", file=sys.stderr)
        return 1

    if args.save_images:
        args.save_images.mkdir(parents=True, exist_ok=True)

    fonts = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".ttf"
    )
    if not fonts:
        return 0

    processor, model = load_model()

    for path in fonts:
        try:
            family, style, cmap, missing_symbols = check_font(path)
            if missing_symbols:
                print(
                    f"warning {path.name}: missing symbols: {' '.join(missing_symbols)}",
                    file=sys.stderr,
                )
            img = render(path, cmap)
            if args.save_images:
                img.save(args.save_images / f"{path.name}.png")
            vec = embed(img, processor, model)
            print(
                json.dumps(
                    {
                        "path": str(path),
                        "family": family,
                        "style": style,
                        "vector": vec,
                    }
                ),
                flush=True,
            )
        except Exception as e:
            print(f"skipped {path.name}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
