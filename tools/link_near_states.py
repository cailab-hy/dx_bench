"""BED-03 near scenes (D-39) share the garment layout of the easy scene with the same seed: copy its cloth states.

    python3 tools/link_near_states.py   # eval (benchmarks/dx_bed_v0/scenes) and train (data/scenes_train)
"""
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
n = missing = 0
for pat, easy_fmt in (("benchmarks/dx_bed_v0/scenes/bed03_near_*.json", "bed03_easy_{s}.json"),
                      ("data/scenes_train/bed03_near_train_*.json", "bed03_easy_train_{s}.json")):
    for f in sorted(glob.glob(os.path.join(REPO, pat))):
        near = json.load(open(f))
        seed = os.path.basename(f).split("_")[-1].split(".")[0]
        easy_f = os.path.join(os.path.dirname(f), easy_fmt.format(s=seed))
        easy = {o["name"]: o for o in json.load(open(easy_f))["objects"]}
        for o in near["objects"]:
            e = easy.get(o["name"])
            assert e is not None and e["kind"] == o["kind"] and e["xyz"] == o["xyz"], (f, o["name"])
            o["cloth_state"] = e.get("cloth_state")
            missing += o["kind"] in ("tshirt", "towel", "sock") and not o["cloth_state"]
        json.dump(near, open(f, "w"), indent=1)
        n += 1
print(f"[near] linked {n} scenes, {missing} garments without a state")
