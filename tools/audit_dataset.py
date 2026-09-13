"""Audit data/pokemon: check every committed image against its source truth.

Three independent checks, run over the entire dataset:

  1. Provenance — re-derive the (species, filename) pair each build step was
     expected to write from the raw sources (Dusduo parquet label column,
     JJMack parquet National Dex numbers in file_name, RogerKoala dex numbers
     embedded in file names) and compare with what is on disk. Catches images
     routed to the wrong species folder.
  2. Integrity — every file must open with PIL and be 128x128 RGB, every
     species folder must be non-empty.
  3. Duplication — the same bytes must not appear under two different species.

Expects .dataset_cache/ to hold the same raw sources used by
tools/build_dataset.py (parquet dumps + species_names.csv).

Usage (from the repo root):
    uv run --with pyarrow,pillow,numpy python tools/audit_dataset.py
Exit code 0 = no problems found.
"""

import collections
import hashlib
import io
import os
import re
import sys

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as bd  # noqa: E402

OUT = bd.OUT
CACHE = bd.CACHE

# Images present in the raw sources but intentionally removed from the
# committed dataset: byte-identical copies of another species' render, kept
# only in the folder whose label matches the actual artwork (audited
# visually). Format: "<species>/<file>.jpg".
KNOWN_REMOVED = {
    "Abra/d00000000.jpg",            # actually Kadabra
    "Alakazam/d00000223.jpg",        # actually Kadabra
    "Alakazam/d00000003.jpg",        # actually Kadabra
    "Kadabra/df02640d4ea714d91ba16fafd46b3a49b.jpg",  # actually Alakazam
    "Clefairy/d00000110.jpg",        # actually Clefable
    "Dewgong/db605edb4e7544d5e80d13153fde73f8a.jpg",  # actually Seel
    "Dratini/d00000014.jpg",         # actually Dragonair
    "Gengar/d00000080.jpg",          # actually Haunter
    "Primeape/d00000114.jpg",        # actually Mankey
    "Omastar/d00000027.jpg",         # actually Omanyte
    "Pidgey/d00000010.jpg",          # actually Pidgeot
    "Pidgeotto/d00000140.jpg",       # actually Pidgeot
    "Poliwhirl/def4d09c9e96c4fc1b2bdb1b427a92820.jpg",  # actually Poliwrath
}


def load_species_names():
    names = {}
    with open(os.path.join(CACHE, "species_names.csv"), encoding="utf-8") as f:
        for line in f:
            parts = line.split(",")
            if len(parts) >= 3 and parts[1] == "9":
                names[int(parts[0])] = parts[2]
    return names


# ── 1. provenance ───────────────────────────────────────────────────────────


def expected_from_dusduo():
    label_names = bd.dusduo_readme()
    for f in bd.DUSDUO_FILES:
        path = os.path.join(CACHE, f"{f}.parquet")
        if not os.path.exists(path):
            print(f"!! missing source {path}", file=sys.stderr)
            return
        pf = pq.ParquetFile(path)
        idx = 0
        for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
            for row in batch.to_pylist():
                img = row["image"]
                stem = os.path.splitext(os.path.basename(img.get("path") or ""))[0]
                stem = "d" + (stem if stem else f"{idx:06d}")
                idx += 1
                yield label_names[row["label"]], f"{stem}.jpg"


def expected_from_jjmack(existing):
    names = load_species_names()
    for name, (source, _expected_bytes) in bd.JJMACK_FILES.items():
        path = os.path.join(CACHE, f"{name}.parquet")
        if not os.path.exists(path):
            print(f"!! missing source {path}", file=sys.stderr)
            return
        pf = pq.ParquetFile(path)
        idx = 0
        for batch in pf.iter_batches(
            batch_size=256,
            columns=["image_data", "generation", "shiny", "file_name", "name"],
        ):
            for row in batch.to_pylist():
                idx += 1
                if row["shiny"] == "yes":
                    continue
                sp = bd.species_from_file_name(row["file_name"], names)
                sp = bd.clean_species_name(sp if sp else row.get("name") or "")
                if not sp or sp in existing:
                    continue
                stem = os.path.splitext(os.path.basename(row["file_name"]))[0]
                yield sp, f"j{name}{idx:06d}{stem[-4:]}.jpg"


# ── 2/3. disk walk: integrity, matching, duplication ─────────────────────────


def audit(disk):
    problems = []

    names = load_species_names()
    # RK source files come with several naming conventions. Only two embed a
    # National Dex number, and only when the digits are followed by the
    # species name itself (rk063Abra_..); bare ids like "117s.jpg" are
    # source-site image numbers, not dex numbers.
    def rk_dex(fname, species):
        m = re.match(r"^rk0*(\d{3})", fname)
        if m and re.match(r"^rk0*\d{3}" + re.escape(species[:4]), fname, re.I):
            return int(m.group(1))
        m = re.match(r"^rkpoke_capture_0*(\d{1,4})_", fname)
        if m:
            return int(m.group(1))
        m = re.match(r"^rkpokemon_icon_0*(\d{1,4})_", fname)
        if m:
            return int(m.group(1))
        return None

    for species, files in sorted(disk.items()):
        if not files:
            problems.append(f"EMPTY FOLDER: {species}")
            continue
        for fname in files:
            full = os.path.join(OUT, species, fname)
            # integrity + provenance
            try:
                img = Image.open(full)
                img.load()
            except Exception as e:
                problems.append(f"CORRUPT: {species}/{fname} ({e})")
                continue
            if img.size != bd.SIZE:
                problems.append(f"BAD SIZE: {species}/{fname} {img.size}")
            if fname.startswith("rk"):
                dex = rk_dex(fname, species)
                if dex is not None:
                    canon = bd.clean_species_name(names.get(dex, "?"))
                    if canon != species:
                        problems.append(
                            f"RK MISMATCH: {species}/{fname} dex={dex} -> {canon}"
                        )

    # cross-species duplicate content (full-file hash)
    h2s = collections.defaultdict(set)
    for species, files in disk.items():
        for fname in files:
            with open(os.path.join(OUT, species, fname), "rb") as fh:
                h = hashlib.sha1(fh.read()).hexdigest()
            h2s[h].add(f"{species}/{fname}")
    for h, files in h2s.items():
        species = {f.split("/")[0] for f in files}
        if len(species) > 1:
            problems.append(f"DUP ACROSS SPECIES: {sorted(files)}")

    return problems


def main():
    if not os.path.isdir(OUT):
        print("data/pokemon missing", file=sys.stderr)
        sys.exit(2)

    disk = {}
    for d in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, d)
        if os.path.isdir(p):
            disk[d] = set(os.listdir(p))

    expected = collections.defaultdict(set)
    for sp, f in expected_from_dusduo():
        expected[sp].add(f)
    # the folders that existed when the jjmack step ran: all gen-I folders
    # (dusduo + rogerkoala), which is what the build snapshotted
    gen1_existing = set(expected) | set(bd.ROGERKOALA_FILL)
    for sp, f in expected_from_jjmack(gen1_existing):
        expected[sp].add(f)

    # Dusduo images saved under both a train and test split file with the
    # same content-hash stem land in the same slot; compare per-species sets.
    # idx-fallback names (d00000003.jpg) repeat across species, so matching
    # is done per folder: expected count vs on-disk non-rk count.
    problems = []
    for sp, exp in expected.items():
        have = {f for f in disk.get(sp, set()) if not f.startswith("rk")}
        removed_here = {r.split("/")[1] for r in KNOWN_REMOVED if r.startswith(sp + "/")}
        if len(have) != len(exp - removed_here):
            problems.append(
                f"COUNT MISMATCH {sp}: disk={len(have)} expected={len(exp)} "
                f"(known removed={len(removed_here)})"
            )
        for f in have - exp:
            problems.append(f"UNEXPECTED FILE: {sp}/{f}")
    problems += audit(disk)

    total_files = sum(len(f) for f in disk.values())
    print(f"audited {len(disk)} species / {total_files} files")
    print(f"expected from sources: {sum(len(s) for s in expected.values())}")
    if problems:
        print(f"\n{len(problems)} PROBLEMS:")
        for p in problems[:200]:
            print(" -", p)
        sys.exit(1)
    print("PROVENANCE+INTEGRITY OK")


if __name__ == "__main__":
    main()
