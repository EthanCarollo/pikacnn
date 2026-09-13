"""Content audit for data/pokemon: does every image look like its species?

For each species, official reference images are fetched from the PokeAPI
sprites repository (pixel sprite + official artwork). Every committed image
is reduced to a compact descriptor (32x32 RGB vector + 512-bin RGB
histogram, both cosine-normalized) and scored against all 1025 species.
An image is flagged when its own species is not the top match, or ranks
poorly — those are then reviewed by eye.

This is a heuristic: closely related species (evolution lines) can confuse
it, so flagged images are candidates for review, not proven errors.

Usage (from the repo root):
    uv run --with pillow,numpy python tools/audit_content.py [--sample N]
"""

import collections
import concurrent.futures
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as bd  # noqa: E402

OUT = bd.OUT
CACHE = bd.CACHE
REF_DIR = os.path.join(CACHE, "references")

SPRITE_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{}.png"
ARTWORK_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{}.png"

PIX = 32  # downscaled square
HIST_BINS = 8  # per channel -> 512 bins


def species_folder_map():
    """id -> committed folder name, via official English display names."""
    names = {}
    with open(os.path.join(CACHE, "species_names.csv"), encoding="utf-8") as f:
        for line in f:
            parts = line.split(",")
            if len(parts) >= 3 and parts[1] == "9":
                names[int(parts[0])] = parts[2]
    folders = {}
    for i in range(1, 1026):
        folders[i] = bd.clean_species_name(names[i])
    return folders


def fetch_ref(url, dest):
    if os.path.exists(dest):
        return
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())


def download_references():
    os.makedirs(REF_DIR, exist_ok=True)
    jobs = []
    for i in range(1, 1026):
        jobs.append((SPRITE_URL.format(i), os.path.join(REF_DIR, f"sprite_{i}.png")))
        jobs.append((ARTWORK_URL.format(i), os.path.join(REF_DIR, f"art_{i}.png")))
    missing = [(u, d) for u, d in jobs if not os.path.exists(d)]
    if missing:
        print(f"downloading {len(missing)} reference images ...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(fetch_ref, u, d) for u, d in missing]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print("ref fetch failed:", e, file=sys.stderr)


def descriptor(img):
    """Crop to the non-background content, then (pix, hist, extra) vectors."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    a = np.asarray(img, dtype=np.float32)
    # background estimate: median of border pixels
    border = np.concatenate([a[0, :, :], a[-1, :, :], a[:, 0, :], a[:, -1, :]])
    bgc = np.median(border, axis=0)
    mask = np.abs(a - bgc).sum(axis=2) > 45
    if mask.sum() >= 32:
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        h, w = a.shape[:2]
        pad = int(0.03 * max(y1 - y0, x1 - x0))
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1, x1 = min(h, y1 + pad), min(w, x1 + pad)
        aspect = (x1 - x0) / (y1 - y0)
    else:
        aspect = 1.0
    img = img.crop((0, 0, a.shape[1], a.shape[0]) if mask.sum() < 32 else (x0, y0, x1, y1))
    img = img.resize((PIX, PIX), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    pix = arr.ravel()
    pix = pix - pix.mean()
    n = np.linalg.norm(pix)
    pix = pix / n if n else pix
    q = (np.asarray(img, dtype=np.uint8) // (256 // HIST_BINS)).astype(np.int64)
    hist = np.bincount(
        ((q[:, :, 0] * HIST_BINS + q[:, :, 1]) * HIST_BINS + q[:, :, 2]).ravel(),
        minlength=HIST_BINS ** 3,
    ).astype(np.float32)
    hist /= hist.sum() + 1e-9
    import colorsys
    r, g, b = arr.reshape(-1, 3).mean(axis=0)
    hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
    extra = np.array(
        [np.log(aspect), ss, vv, np.cos(2 * np.pi * hh), np.sin(2 * np.pi * hh)],
        dtype=np.float32,
    )
    extra = extra / (np.linalg.norm(extra) + 1e-9)
    return pix, hist, extra


def main():
    sample_n = None
    if "--sample" in sys.argv:
        sample_n = int(sys.argv[sys.argv.index("--sample") + 1])

    folders = species_folder_map()
    folder_to_idx = {folders[i]: i for i in range(1, 1026)}
    missing_folders = [folders[i] for i in range(1, 1026) if not os.path.isdir(os.path.join(OUT, folders[i]))]
    if missing_folders:
        print("!! no folder for:", missing_folders[:10], file=sys.stderr)
        sys.exit(2)

    download_references()

    # reference: official artwork per species (consistent framing)
    refs_pix, refs_hist, refs_extra = [], [], []
    for i in range(1, 1026):
        p = os.path.join(REF_DIR, f"art_{i}.png")
        pix, hist, extra = descriptor(Image.open(p))
        refs_pix.append(pix)
        refs_hist.append(hist)
        refs_extra.append(extra)
    R_pix = np.stack(refs_pix)  # (1025, 768)
    R_hist = np.stack(refs_hist)  # (1025, 512)
    R_extra = np.stack(refs_extra)  # (1025, 5)

    files = []
    for sp in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, sp)
        if not os.path.isdir(p):
            continue
        for fname in sorted(os.listdir(p)):
            files.append((sp, fname))
    if sample_n:
        rng = np.random.default_rng(7)
        files = [files[i] for i in rng.choice(len(files), size=min(sample_n, len(files)), replace=False)]

    print(f"scoring {len(files)} images against {R_pix.shape[0]} species ...", flush=True)
    flags = []
    rank_hist = collections.Counter()
    for k, (sp, fname) in enumerate(files):
        try:
            pix, hist, extra = descriptor(Image.open(os.path.join(OUT, sp, fname)))
        except Exception as e:
            flags.append((99, sp, fname, f"UNREADABLE {e}"))
            continue
        sim = 0.45 * (R_pix @ pix) + 0.45 * (R_hist @ hist) + 0.10 * (R_extra @ extra)
        own = folder_to_idx[sp] - 1
        rank = int((sim > sim[own]).sum())
        rank_hist[rank] += 1
        if rank >= 1:
            top = int(sim.argmax())
            flags.append((rank, sp, fname, f"best={folders[top + 1]} (sim {sim[top]:.2f} vs own {sim[own]:.2f})"))
        if (k + 1) % 2000 == 0:
            print(f"  {k + 1}/{len(files)} done, {len(flags)} flagged", flush=True)

    print("\nrank distribution (rank 0 = own species is top match):")
    for r in sorted(rank_hist)[:8]:
        print(f"  rank {r}: {rank_hist[r]}")
    if flags:
        print(f"\n{len(flags)} images to review (rank >= 1):")
        for rank, sp, fname, why in sorted(flags):
            print(f"  rank={rank:>2} {sp}/{fname} -> {why}")
    else:
        print("\nALL IMAGES MATCH THEIR SPECIES (content-wise)")


if __name__ == "__main__":
    main()
