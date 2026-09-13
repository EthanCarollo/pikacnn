"""Rebuild data/pokemon: the Pokemon classification dataset (generations I-IX).

Layout: data/pokemon/<Species>/*.jpg — 128x128 JPEGs, one folder per class
(the National Dex #001-#1025).

Sources (public Hugging Face datasets):
  - Dusduo/1stGen-Pokemon-Images : parquet dumps, ~10.5k rendered images.
    Covers 142 species; its 'Mr. Mime' and 'MrMime' labels are merged into one
    folder.
  - RogerKoala/gen1-pokemon-images : per-species folders (MIT license), used
    to fill the 9 species missing from Dusduo (Golem, Kabuto, Krabby, Muk,
    Nidoran-f, Nidoran-m, Onix, Paras, Persian).
  - JJMack/pokemon-classification-gen1-9 : 256px official sprites
    (CC-BY-NC-SA-4.0), used for generations II-IX only (shiny variants
    excluded). Generation I species keep their rendered images.

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
import unicodedata
import urllib.request

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
JJMACK = "JJMack/pokemon-classification-gen1-9"
# cache name -> (source file, expected bytes)
JJMACK_FILES = {
    "jj_train": ("train", 437175163),
    "jj_val": ("validation", 72568614),
    "jj_test": ("test", 89673734),
}

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
    import pyarrow.parquet as pq
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


def clean_species_name(name):
    """Normalize a display species name into a folder-safe name.

    Strips accents (Flabebe), gender symbols (Nidoran-f/-m), apostrophes
    (Farfetchd) and characters illegal in Windows paths (Type: Null ->
    Type Null)."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.replace("♀", "-f").replace("♂", "-m")
    # A few rows carry file names instead of species names ("Iron_Thorns.png")
    name = re.sub(r"(?i)\.(png|jpe?g|webp)$", "", name)
    name = name.replace("_", " ")
    # curly apostrophes ("Farfetch’d") into ASCII so they get stripped too
    name = name.replace("‘", "'").replace("’", "'")
    name = re.sub(r"[:?\"<>|*']", "", name)
    # Windows silently strips trailing dots/commas from folder names
    # ("Mime Jr.," -> "Mime Jr"), so mirror that here
    name = name.rstrip(" .,")
    return re.sub(r"\s+", " ", name).strip()


def pokeapi_species_names():
    """id -> English display name, from PokeAPI's pokemon_species_names.csv."""
    csv_path = os.path.join(CACHE, "species_names.csv")
    if not os.path.exists(csv_path):
        os.makedirs(CACHE, exist_ok=True)
        fetch(
            "https://raw.githubusercontent.com/PokeAPI/pokeapi/master/"
            "data/v2/csv/pokemon_species_names.csv", csv_path,
        )
    names = {}
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split(",")
            if len(parts) >= 3 and parts[1] == "9":  # local_language_id 9 = English
                names[int(parts[0])] = parts[2]
    return names


def species_from_file_name(file_name, names):
    """JJMack file names look like '0137-Porygon-SpriteBack-4.png'; the
    4-digit prefix is the National Dex number. The dataset's own 'name'
    column is corrupted for hyphenated species ('Ho-Oh' -> 'Ho')."""
    m = re.match(r"(\d{4})-", file_name)
    if m and int(m.group(1)) in names:
        return names[int(m.group(1))]
    return None


def from_jjmack():
    """Add generations II-IX from the JJMack sprite dataset.

    Skips shiny variants (wrong colors for species ID) and National Dex
    #001-#151 — generation I keeps its rendered images. Species come from the
    dex number embedded in each file name, resolved to the official English
    display name via PokeAPI."""
    import pyarrow.parquet as pq
    existing = {
        d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d))
    }
    names = pokeapi_species_names()
    n_ok = n_skip = 0
    for name, (source, expected) in JJMACK_FILES.items():
        path = os.path.join(CACHE, f"{name}.parquet")
        if not (os.path.exists(path) and os.path.getsize(path) == expected):
            os.makedirs(CACHE, exist_ok=True)
            print(f"[jjmack] downloading {name}.parquet ...", flush=True)
            fetch(hf_resolve(JJMACK, f"{source}.parquet"), path)
        pf = pq.ParquetFile(path)
        idx = 0
        for batch in pf.iter_batches(
            batch_size=256,
            columns=["image_data", "generation", "shiny", "file_name", "name"],
        ):
            for row in batch.to_pylist():
                idx += 1
                if row["shiny"] == "yes":
                    n_skip += 1
                    continue
                species = species_from_file_name(row["file_name"], names)
                species = clean_species_name(
                    species if species else row.get("name") or ""
                )
                if not species or species in existing:
                    # existing covers every National Dex #001-#151 species
                    # (rendered images already committed)
                    n_skip += 1
                    continue
                stem = os.path.splitext(os.path.basename(row["file_name"]))[0]
                if save_image(row["image_data"], species, f"j{name}{idx:06d}{stem[-4:]}"):
                    n_ok += 1
                else:
                    n_skip += 1
        print(f"[jjmack] {name}: cumulative ok={n_ok} skip={n_skip}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    from_dusduo()
    from_rogerkoala()
    from_jjmack()
    species = sorted(
        d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d))
    )
    total = sum(
        len(os.listdir(os.path.join(OUT, s))) for s in species
    )
    minsp = min(species, key=lambda s: len(os.listdir(os.path.join(OUT, s))))
    print(f"\nRESULT: {len(species)} species, {total} images "
          f"(rarest: {minsp} with {len(os.listdir(os.path.join(OUT, minsp)))})")
