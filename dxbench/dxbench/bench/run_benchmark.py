"""DX bed benchmark runner: experiment YAML -> one Kit process per scene (run_episode.py) -> aggregate -> report.

    ~/miniforge3/envs/dx_arena/bin/python -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/oracle.yaml
    ... --aggregate_only            # re-aggregate existing results
    ... --limit 2                   # first N scenes per (task, difficulty)

Experiment YAML:
  name: oracle_bed03                     # results -> outputs/bench/<name>/
  policy: dxbench.policies.oracle_bed03.Bed03OraclePolicy
  policy_kw: {}                          # --policy_kw k=v
  env_args: [--embodiment, rby1_ik]      # extra environment args (after --scene)
  tasks: {BED-03: [easy]}                # task -> difficulties
  scenes_dir: benchmarks/dx_bed_v0/scenes
  video: false
  server: null                           # optional: shell command started before the episodes (policy server),
  server_ready: "listening"              #   waited for this line, stopped at the end
  timeout_s: 3600
  parallel: 3                            # scenes run at the same time (one Kit process each; the server is shared)
  max_steps: null                        # optional cap (e.g. the hold baseline)
Results: outputs/bench/<name>/<scene_id>/result.json (+ mp4), summary.json, REPORT.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
PY = os.environ.get("DX_ARENA_PY", sys.executable)
ENV_CLASS = {"BED-01": ("DxBed01Environment", "dx_bed01"), "BED-02": ("DxBed02Environment", "dx_bed02"),
             "BED-03": ("DxBed03Environment", "dx_bed03")}


def scenes_for(exp: dict, limit: int | None) -> list[tuple[str, str, Path]]:
    out = []
    for task, diffs in exp["tasks"].items():
        for d in diffs:
            pat = str(REPO / exp.get("scenes_dir", "benchmarks/dx_bed_v0/scenes") / f"{task.lower().replace('-', '')}_{d}_*.json")
            files = sorted(glob.glob(pat))[:limit]
            out += [(task, d, Path(f)) for f in files]
    return out


def run_scene(exp: dict, task: str, scene: Path, out_dir: Path, log: Path) -> dict | None:
    cls, name = ENV_CLASS[task]
    kw = [f"{k}={json.dumps(v) if not isinstance(v, str) else v}" for k, v in (exp.get("policy_kw") or {}).items()]
    cmd = [PY, str(REPO / "dxbench/dxbench/scripts/run_episode.py"), "--policy", exp["policy"], "--out", str(out_dir)]
    if kw:
        cmd += ["--policy_kw", *kw]
    if exp.get("video"):
        cmd += ["--video"]
    if exp.get("max_steps"):
        cmd += ["--max_steps", str(exp["max_steps"])]
    cmd += ["--external_environment_class_path", f"dxbench.tasks.bed_tasks:{cls}", name, "--scene", str(scene),
            *[str(x) for x in exp.get("env_args", [])]]
    for attempt in range(3):  # known Kit startup segfault (~8 %): retry up to 3 times
        with open(log, "w") as f:
            f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
            f.flush()
            try:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO,
                                    timeout=exp.get("timeout_s", 3600)).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        res_f = out_dir / "result.json"
        if res_f.exists():
            return json.loads(res_f.read_text())
        txt = log.read_text(errors="ignore")
        if "platforminfo" not in txt and "Segmentation" not in txt and rc != -11:
            break
        print(f"  retry {attempt + 1} (rc={rc})", flush=True)
    return {"error": f"no result (rc={rc})", "log": str(log)}


def start_server(exp: dict, out: Path):
    cmd = exp.get("server")
    if not cmd:
        return None
    log = open(out / "server.log", "w")
    p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 600:
        if exp.get("server_ready", "listening") in (out / "server.log").read_text(errors="ignore"):
            return p
        if p.poll() is not None:
            raise RuntimeError(f"policy server exited: see {out / 'server.log'}")
        time.sleep(1)
    raise RuntimeError("policy server not ready in 600 s")


def aggregate(exp: dict, out: Path) -> dict:
    rows = []
    for f in sorted(out.glob("*/result.json")):
        r = json.loads(f.read_text())
        sid = f.parent.name
        if r.get("terminated_by") == "success" and not r.get("success"):
            # results written before run_episode used the pre-reset evaluation: the success termination term is
            # the task predicate itself, and success implies the full score (1.0) in all three tasks
            r.update(success=True, score=1.0, failure=None, fixed_post_reset_eval=True,
                     stages_end={k: True for k in (r.get("stages_end") or {})})
        task = {"bed01": "BED-01", "bed02": "BED-02", "bed03": "BED-03"}.get(sid[:5])
        diff = sid.split("_")[1] if "_" in sid else None
        rows.append({"scene": sid, "task": task, "difficulty": diff, **r})
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["task"], r["difficulty"]), []).append(r)
    table = []
    for (task, diff), rs in sorted(groups.items()):
        ok = [r for r in rs if "error" not in r]
        n = len(rs)
        succ = sum(bool(r.get("success")) for r in ok)
        stages = sorted({k for r in ok for k in (r.get("stages_end") or {})})
        stage_rate = {k: sum(bool((r.get("stages_end") or {}).get(k)) for r in ok) / max(1, len(ok)) for k in stages}
        ttf = {k: [r["first_reach_step"][k] / 15.0 for r in ok if k in (r.get("first_reach_step") or {})] for k in stages}
        table.append({"task": task, "difficulty": diff, "episodes": n, "errors": n - len(ok), "success": succ,
                      "success_rate": succ / max(1, n),
                      "score_mean": sum(float(r.get("score") or 0) for r in ok) / max(1, len(ok)),
                      "stage_rate": stage_rate,
                      "stage_time_s_mean": {k: (sum(v) / len(v) if v else None) for k, v in ttf.items()},
                      "tipped": sum(bool((r.get("safety") or {}).get("robot_tipped")) for r in ok),
                      "sim_s_mean": sum(float(r.get("sim_s") or 0) for r in ok) / max(1, len(ok)),
                      "wall_s_mean": sum(float(r.get("wall_s") or 0) for r in ok) / max(1, len(ok)),
                      "failures": sorted({str(r.get("failure")) for r in ok if r.get("failure")})})
    summary = {"experiment": exp["name"], "policy": exp["policy"], "policy_kw": exp.get("policy_kw"),
               "env_args": exp.get("env_args"), "groups": table, "episodes": rows,
               "generated": time.strftime("%Y-%m-%d %H:%M")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "REPORT.md").write_text(report_md(summary))
    return summary


def report_md(s: dict) -> str:
    L = [f"# Benchmark result: {s['experiment']}", "",
         f"- policy: `{s['policy']}` {json.dumps(s.get('policy_kw') or {})}",
         f"- env args: `{' '.join(map(str, s.get('env_args') or []))}`", f"- generated: {s['generated']}", "",
         "| task | difficulty | episodes | success | score mean | stage completion | tipped | errors | sim s | wall s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for g in s["groups"]:
        st = ", ".join(f"{k} {v:.0%}" for k, v in g["stage_rate"].items())
        L.append(f"| {g['task']} | {g['difficulty']} | {g['episodes']} | {g['success']}/{g['episodes']} "
                 f"({g['success_rate']:.0%}) | {g['score_mean']:.2f} | {st} | {g['tipped']} | {g['errors']} | "
                 f"{g['sim_s_mean']:.0f} | {g['wall_s_mean']:.0f} |")
    L += ["", "## Episodes", "", "| scene | success | score | failure | steps | tipped | first reach (s) |",
          "|---|---|---|---|---|---|---|"]
    for r in s["episodes"]:
        if "error" in r:
            L.append(f"| {r['scene']} | error | – | {r['error']} | – | – | – |")
            continue
        fr = ", ".join(f"{k} {v / 15:.0f}" for k, v in (r.get("first_reach_step") or {}).items())
        L.append(f"| {r['scene']} | {r.get('success')} | {float(r.get('score') or 0):.2f} | {r.get('failure') or ''} | "
                 f"{r.get('steps')} | {(r.get('safety') or {}).get('robot_tipped')} | {fr} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="run scenes that already have a result again")
    ap.add_argument("--parallel", type=int, default=None, help="Kit processes at once (default: YAML `parallel`, 1)")
    ap.add_argument("--only", default=None, help="comma-separated scene ids to run (others untouched)")
    a = ap.parse_args()
    exp = yaml.safe_load(open(a.experiment))
    out = REPO / "outputs" / "bench" / exp["name"]
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "experiment.yaml").write_text(yaml.safe_dump(exp))
    if not a.aggregate_only:
        srv = start_server(exp, out)
        try:
            def need(sc):
                f = out / sc.stem / "result.json"
                return a.rerun or not f.exists() or "error" in json.loads(f.read_text())

            only = set(a.only.split(",")) if a.only else None
            todo = [(t, d, sc) for t, d, sc in scenes_for(exp, a.limit) if need(sc) and (not only or sc.stem in only)]

            def one(item):
                i, (task, diff, scene) = item
                time.sleep(20 * (i % max(1, int(a.parallel or exp.get("parallel", 1)))) if i < 8 else 0)  # stagger starts
                sid = scene.stem
                od = out / sid
                t0 = time.time()
                r = run_scene(exp, task, scene, od, out / "logs" / f"{sid}.log")
                print(f"[bench] {i + 1}/{len(todo)} {sid}: success={r.get('success')} score={r.get('score')} "
                      f"err={r.get('error')} {time.time() - t0:.0f}s", flush=True)
                if "error" in r:
                    od.mkdir(parents=True, exist_ok=True)
                    (od / "result.json").write_text(json.dumps(r))

            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max(1, int(a.parallel or exp.get("parallel", 1)))) as ex:
                list(ex.map(one, enumerate(todo)))
        finally:
            if srv is not None:
                os.killpg(srv.pid, 15)
    s = aggregate(exp, out)
    for g in s["groups"]:
        print(f"[bench] {s['experiment']} {g['task']} {g['difficulty']}: {g['success']}/{g['episodes']} "
              f"score {g['score_mean']:.2f}", flush=True)


if __name__ == "__main__":
    main()
