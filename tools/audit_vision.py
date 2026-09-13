"""Vision audit for data/pokemon: verify each image with a fine-tuned ViT.

Uses skshmjn/Pokemon-classifier-gen9-1025 (ViT-base fine-tuned on the 1025
National Dex species, Apache-2.0) to classify every committed image and
compare the prediction with the species folder. Images whose folder species
is not the top prediction are flagged for manual review (the model can
confuse closely related species, so flags are candidates, not verdicts).

Usage (from the repo root):
    uv run --with transformers,torch,pillow python tools/audit_vision.py [--sample N]
Writes .dataset_cache/vision_flags.txt and prints per-species accuracy.
"""

import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as bd  # noqa: E402

OUT = bd.OUT
CACHE = bd.CACHE
FLAGS = os.path.join(CACHE, "vision_flags.txt")


def key(name):
    return name.lower().replace(" ", "").replace("-", "").replace(".", "")


def main():
    sample_n = None
    if "--sample" in sys.argv:
        sample_n = int(sys.argv[sys.argv.index("--sample") + 1])

    files = []
    for sp in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, sp)
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                files.append((sp, f))
    if sample_n:
        import random
        random.seed(7)
        files = random.sample(files, min(sample_n, len(files)))

    from transformers import pipeline
    clf = pipeline(
        "image-classification",
        model="skshmjn/Pokemon-classifier-gen9-1025",
        device=-1,
        top_k=5,
    )

    per_species = {}
    flags = []
    for i, (sp, fname) in enumerate(files):
        img = Image.open(os.path.join(OUT, sp, fname)).convert("RGB")
        preds = clf(img)
        top = preds[0]
        top_name = bd.clean_species_name(top["label"])
        top5_keys = {key(bd.clean_species_name(p["label"])) for p in preds}
        st = per_species.setdefault(sp, [0, 0])
        st[1] += 1
        if key(top_name) == key(sp):
            st[0] += 1
        else:
            flags.append((sp, fname, top_name, top["score"], key(sp) in top5_keys))
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(files)} done, {len(flags)} flagged", flush=True)

    with open(FLAGS, "w", encoding="utf-8") as f:
        for sp, fname, pred, score, in_top5 in sorted(flags):
            f.write(f"{'TOP5' if in_top5 else 'OUT5'} {sp}/{fname} -> {pred} ({score:.2f})\n")

    wrong = sum(1 for sp, (ok, tot) in per_species.items() if ok < tot)
    n_flags = len(flags)
    out5 = sum(1 for fl in flags if not fl[4])
    print(f"\nchecked {len(files)} images | {n_flags} mispredicted ({out5} outside top-5)")
    print(f"species with at least one mismatch: {wrong}/{len(per_species)}")
    print(f"flags written to {FLAGS}")
    # worst species by mismatch count
    worst = sorted(
        ((tot - ok, ok, tot, sp) for sp, (ok, tot) in per_species.items()),
        reverse=True,
    )[:15]
    print("\nworst species (mismatches, correct, total):")
    for mm, ok, tot, sp in worst:
        if mm:
            print(f"  {sp}: {mm}/{tot} wrong ({ok} correct)")


if __name__ == "__main__":
    main()
