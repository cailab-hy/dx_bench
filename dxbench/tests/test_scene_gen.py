"""Scene diversification (D-42): layouts and colours unchanged, evaluation easy canonical, others varied."""
from dxbench.tasks import scene_gen as G
from dxbench.tasks.predicates import TEXT
from dxbench.tasks.scene_spec import DEFAULT_LIGHTING


def _layout(spec):
    return [(o.name, o.kind, o.xyz, o.yaw, o.variant) for o in spec.objects]


def test_diversify_keeps_layout_and_eval_easy_canonical():
    for task in ("BED-01", "BED-02", "BED-03"):
        base = G.GEN[task]("easy", 3, "eval")
        div = G.diversify(G.GEN[task]("easy", 3, "eval"), "eval")
        assert _layout(div) == _layout(base)
        assert div.robot == base.robot and div.lighting == DEFAULT_LIGHTING
        assert div.language["high"] == TEXT[task].high[0]


def test_diversify_varies_hard_and_train():
    langs, lights, moved = set(), set(), 0
    for seed in range(10):
        base = G.gen_bed03("hard", seed, "eval")
        div = G.diversify(G.gen_bed03("hard", seed, "eval"), "eval")
        assert _layout(div) == _layout(base)
        assert div.language["high"] in TEXT["BED-03"].high
        langs.add(div.language["high"])
        lights.add((div.lighting["blinds"], div.lighting["lighting"]))
        moved += div.robot["xy"] != base.robot["xy"]
    assert len(langs) > 2 and len(lights) > 2 and moved >= 8
    tr = G.diversify(G.gen_bed02("easy", 10001, "train"), "train")
    assert abs(tr.robot["xy"][0] - G.gen_bed02("easy", 10001, "train").robot["xy"][0]) <= 0.05


def test_diversify_medium_keeps_robot_and_room_lights():
    base = G.gen_bed02("medium", 4, "eval")
    div = G.diversify(G.gen_bed02("medium", 4, "eval"), "eval")
    assert div.robot == base.robot and div.lighting["lighting"] == "day" and div.lighting["intensity"] == 1.0
