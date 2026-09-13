"""Rebuild data/pokemon: the first-generation Pokemon classification dataset.

Layout: data/pokemon/<Species>/*.jpg — 128x128 JPEGs, one folder per class
(151 species, the National Dex #001-#151).

Sources (public Hugging Face mirrors of the Kaggle 'Pokemon Generation One'
dataset, plus a 151-species collection):
  - Dusduo/1stGen-Pokemon-Images : parquet dumps, ~10.5k images. Covers 142
    species; its 'Mr. Mime' and 'MrMime' labels are merged into one folder.
  - RogerKoala/gen1-pokemon-images : per-species folders (MIT license), used
    to fill the 9 species missing from Dusduo (Golem, Kabuto, Krabby, Muk,
    Nidoran-f, Nidoran-m, Onix, Paras, Persian).

Dusduo/1stGen-Pokemon-Images declares no license on Hugging Face; the dataset
is redistributed here with attribution only. See the Kaggle source page for
original credits.

Usage (from the repo root):
    uv run --with pyarrow,pillow python tools/build_dataset.py
"""

import io
import json
import os
import re
import sys
import urllib.request

import pyarrow.parquet as pq
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "data", "pokemon")
CACHE = os.path.join(REPO_ROOT, ".dataset_cache")

DUSDUO = "Dusduo/1stGen-Pokemon-Images"
DUSDUO_FILES = {
    "train-00000": 387980099,
    "train-00001": 386094113,
    "train-00002": 370156790,
    "train-00003": 409235801,
    "test": 375264771,
}
ROGERKOALA = "RogerKoala/gen1-pokemon-images"
ROGERKOALA_FILL = ["Golem", "Kabuto", "Krabby", "Muk", "Nidoran-f", "Nidoran-m", "Onix", "Paras", "Persian"]

SIZE = (128, 128)
JPEG_QUALITY = 90


def hf_resolve(repo, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def norm_label(name):
    name = name.strip()
    return "Mr. Mime" if name == "MrMime" else name


def save_image(img_bytes, species, stem):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Flatten transparent sprites onto white instead of black
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img).convert("RGB")
        else:
            img = img.convert("RGB")
    except Exception:
        return False
    img = img.resize(SIZE, Image.LANCZOS)
    out_dir = os.path.join(OUT, species)
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, f"{stem}.jpg"), "JPEG", quality=JPEG_QUALITY)
    return True


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def dusduo_readme():
    readme = os.path.join(CACHE, "dusduo_README.md")
    if not os.path.exists(readme):
        os.makedirs(CACHE, exist_ok=True)
        fetch(hf_resolve(DUSDUO, "README.md"), readme)
    text = open(readme, encoding="utf-8").read()
    pairs = re.findall(r"'(\d+)': ([A-Za-z.' ]+)", text)
    return {int(i): norm_label(n) for i, n in pairs}


def from_dusduo():
    label_names = dusduo_readme()
    n_ok = n_bad = 0
    for name, expected in DUSDUO_FILES.items():
        path = os.path.join(CACHE, f"{name}.parquet")
        if not (os.path.exists(path) and os.path.getsize(path) == expected):
            os.makedirs(CACHE, exist_ok=True)
            print(f"[dusduo] downloading {name}.parquet ...", flush=True)
            fetch(hf_resolve(DUSDUO, f"data/{name}.parquet"), path)
        pf = pq.ParquetFile(path)
        idx = 0
        for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
            for row in batch.to_pylist():
                img = row["image"]
                species = label_names[row["label"]]
                stem = os.path.splitext(os.path.basename(img.get("path") or ""))[0]
                stem = "d" + (stem if stem else f"{idx:06d}")
                if save_image(img["bytes"], species, stem):
                    n_ok += 1
                else:
                    n_bad += 1
                idx += 1
        print(f"[dusduo] {name}: cumulative ok={n_ok} bad={n_bad}", flush=True)


def from_rogerkoala():
    n_ok = 0
    for split in ("train", "test"):
        for species in ROGERKOALA_FILL:
            api = f"https://huggingface.co/api/datasets/{ROGERKOALA}/tree/main/dataset/{split}/{species}?limit=500"
            try:
                with urllib.request.urlopen(api, timeout=30) as r:
                    entries = json.load(r)
            except Exception as e:
                print(f"[rk] {split}/{species}: listing failed: {e}", flush=True)
                continue
            for e in entries:
                if e["type"] != "file":
                    continue
                try:
                    with urllib.request.urlopen(hf_resolve(ROGERKOALA, e["path"]), timeout=60) as r:
                        data = r.read()
                except Exception:
                    continue
                stem = os.path.splitext(os.path.basename(e["path"]))[0]
                if save_image(data, species, f"rk{stem}"):
                    n_ok += 1
            print(f"[rk] {split}/{species}: cumulative ok={n_ok}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    from_dusduo()
    from_rogerkoala()
    species = sorted(
        d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d))
    )
    total = sum(
        len(os.listdir(os.path.join(OUT, s))) for s in species
    )
    minsp = min(species, key=lambda s: len(os.listdir(os.path.join(OUT, s))))
    print(f"\nRESULT: {len(species)} species, {total} images "
          f"(rarest: {minsp} with {len(os.listdir(os.path.join(OUT, minsp)))})")
