"""T2: compare rby1-sdk model_v1.2.urdf with the USD joints (from T1 JSON). No Kit needed.

    bash tools/rby1_check/fetch_urdf.sh      # once
    python tools/rby1_check/t2_urdf_compare.py [--usd-key model_a] [--grip-key rbgripper_d405_collision]

Outputs reports/rby1_check/t2/{urdf_diff.md, joint_order.md, t2.json}.
Besides names/limits/effort/velocity it does a zero-pose kinematic check: world joint origin and
axis from USD (body0 pose x localPos0/localRot0 x physics:axis) vs URDF forward kinematics.
"""
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET

import numpy as np

from common import CPP_JOINT_NAMES, REPORTS, URDF, load_json, out_dir, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--usd-key", default="model_a", help="T1 JSON with the 24 body DOF")
ap.add_argument("--grip-key", default="rbgripper_d405_collision", help="T1 JSON with gripper joints")
args = ap.parse_args()


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def quat_to_R(q):  # q = [w, x, y, z]
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def parse_urdf(path):
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else [0, 0, 0])]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else [0, 0, 0])]
        ax = j.find("axis")
        lim = j.find("limit")
        mim = j.find("mimic")
        joints[j.get("name")] = {
            "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "xyz": xyz, "rpy": rpy,
            "axis": [float(v) for v in ax.get("xyz").split()] if ax is not None else [1, 0, 0],
            "lower": float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
            "upper": float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
            "effort": float(lim.get("effort")) if lim is not None and lim.get("effort") else None,
            "velocity": float(lim.get("velocity")) if lim is not None and lim.get("velocity") else None,
            "mimic": {"joint": mim.get("joint"), "multiplier": mim.get("multiplier")} if mim is not None else None,
        }
    # zero-pose FK: world frame of every link and joint
    child_of = {v["child"]: k for k, v in joints.items()}
    links = {l.get("name") for l in root.findall("link")}
    roots = [l for l in links if l not in child_of]
    link_T = {roots[0]: np.eye(4)}

    def T_of(link):
        if link in link_T:
            return link_T[link]
        j = joints[child_of[link]]
        Tj = np.eye(4)
        Tj[:3, :3] = rpy_to_R(*j["rpy"])
        Tj[:3, 3] = j["xyz"]
        link_T[link] = T_of(j["parent"]) @ Tj
        return link_T[link]

    for name, j in joints.items():
        Tw = T_of(j["child"])
        j["world_pos"] = Tw[:3, 3].tolist()
        j["world_axis"] = (Tw[:3, :3] @ np.array(j["axis"])).tolist()
    return joints, roots[0]


AXIS_VEC = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}


def usd_joint_world(j, links_by_path):
    a = j["attrs"]
    b0 = j["body0"][0] if j["body0"] else None
    M = np.array(links_by_path[b0]["world_tf"]) if b0 in links_by_path else np.eye(4)
    R0, t0 = M[:3, :3].T, M[3, :3]  # Gf row-vector convention -> column-vector
    lp = np.array(a.get("physics:localPos0") or [0, 0, 0], float)
    lr = a.get("physics:localRot0") or [1, 0, 0, 0]
    Rj = R0 @ quat_to_R(lr)
    tok = a.get("physics:axis")
    axis_tok = tok if tok in AXIS_VEC else "X"  # UsdPhysics fallback is X
    return (R0 @ lp + t0).tolist(), (Rj @ np.array(AXIS_VEC[axis_tok], float)).tolist(), tok


def main():
    od = out_dir("t2")
    uj, uroot = parse_urdf(URDF)
    body = load_json(REPORTS / "t1" / f"{args.usd_key}.json")
    grip = load_json(REPORTS / "t1" / f"{args.grip_key}.json")
    usd = {j["name"]: (j, body) for j in body["joints"]}
    for j in grip["joints"]:
        if j["name"].startswith("gripper_finger"):
            usd[j["name"]] = (j, grip)
    links_by = {id(body): {l["path"]: l for l in body["links"]}, id(grip): {l["path"]: l for l in grip["links"]}}

    rows, diff, kin = [], [], []
    names = sorted(set(uj) | set(usd), key=lambda n: (CPP_JOINT_NAMES.index(n) if n in CPP_JOINT_NAMES else 99, n))
    for n in names:
        u = uj.get(n)
        s_pair = usd.get(n)
        if u is None or s_pair is None:
            if u and u["type"] == "fixed":
                continue
            diff.append({"joint": n, "item": "presence", "urdf": u["type"] if u else "—",
                         "usd": s_pair[0]["type"] if s_pair else "—"})
            continue
        s, src = s_pair
        a = s["attrs"]
        prism = s["type"] == "PhysicsPrismaticJoint"
        conv = 1.0 if prism else math.pi / 180.0
        lo = a.get("physics:lowerLimit")
        hi = a.get("physics:upperLimit")
        lo = lo * conv if isinstance(lo, (int, float)) and math.isfinite(lo) else None
        hi = hi * conv if isinstance(hi, (int, float)) and math.isfinite(hi) else None
        drv = next(iter(s["drives"].values()), {}) if s["drives"] else {}
        mf = drv.get("maxForce") if drv else a.get(f"drive:{'linear' if prism else 'angular'}:physics:maxForce")
        vel = a.get("physxJoint:maxJointVelocity")
        vel = vel * conv if vel is not None else None
        row = {"joint": n, "urdf_type": u["type"], "usd_type": s["type"].replace("Physics", ""),
               "urdf_lower": u["lower"], "usd_lower": lo, "urdf_upper": u["upper"], "usd_upper": hi,
               "urdf_effort": u["effort"], "usd_maxForce": mf, "urdf_velocity": u["velocity"], "usd_maxVel": vel,
               "urdf_mimic": u["mimic"]}
        rows.append(row)

        def cmp(item, x, y, tol):
            if x is None and y is None:
                return
            if x is None or y is None or abs(x - y) > tol:
                diff.append({"joint": n, "item": item, "urdf": x, "usd": y})

        if u["type"] in ("revolute", "prismatic"):
            cmp("lower", u["lower"], lo, 1e-3)
            cmp("upper", u["upper"], hi, 1e-3)
        cmp("effort↔maxForce", u["effort"], mf, 1e-3)
        cmp("velocity↔maxJointVelocity", u["velocity"], vel, 1e-3)
        want = {"revolute": "PhysicsRevoluteJoint", "continuous": "PhysicsRevoluteJoint",
                "prismatic": "PhysicsPrismaticJoint", "fixed": "PhysicsFixedJoint"}.get(u["type"])
        if want and want != s["type"]:
            diff.append({"joint": n, "item": "type", "urdf": u["type"], "usd": s["type"]})

        # kinematics at zero pose (only meaningful when both trees share the base frame)
        p_usd, ax_usd, tok = usd_joint_world(s, links_by[id(src)])
        dp = float(np.linalg.norm(np.array(p_usd) - np.array(u["world_pos"])))
        dot = float(np.dot(ax_usd, u["world_axis"]))
        kin.append({"joint": n, "pos_err_mm": dp * 1000, "axis_dot": dot, "usd_axis_token": tok,
                    "usd_world_axis": ax_usd, "urdf_world_axis": u["world_axis"],
                    "usd_world_pos": p_usd, "urdf_world_pos": u["world_pos"]})

    save_json(od / "t2.json", {"urdf": str(URDF), "urdf_root": uroot, "rows": rows, "diff": diff, "kinematics": kin})

    f = lambda v: "—" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    L = ["# T2 URDF ↔ USD 대조", "",
         f"- URDF: `{URDF}` (root link `{uroot}`)",
         f"- USD(몸체 24 DOF): T1 `{args.usd_key}` · 그리퍼: T1 `{args.grip_key}`",
         "- 단위: 각도 rad, 속도 rad/s 또는 m/s (USD의 deg 값을 환산), effort N·m 또는 N",
         "", "## 차이 나는 항목만", "", "| 관절 | 항목 | URDF | USD |", "|---|---|---|---|"]
    L += [f"| {d['joint']} | {d['item']} | {f(d['urdf'])} | {f(d['usd'])} |" for d in diff] or ["| (없음) | | | |"]
    L += ["", "## 영 자세 기구학 대조 (관절 원점 오차, 축 내적)", "",
          "축 내적 1 = 같은 방향, −1 = 반대(양의 회전 방향이 반대), 0 근처 = 다른 축.", "",
          "| 관절 | 원점 오차 mm | 축 내적 | USD axis 토큰 |", "|---|---|---|---|"]
    for k in kin:
        flag = " ⚠" if k["pos_err_mm"] > 1.0 or k["axis_dot"] < 0.999 else ""
        L.append(f"| {k['joint']}{flag} | {k['pos_err_mm']:.2f} | {k['axis_dot']:+.4f} | `{k['usd_axis_token']}` |")
    L += ["", "## 전체 값", "", "| 관절 | URDF 종류 | USD 종류 | 하한 U/S | 상한 U/S | effort U / maxForce S | vel U/S | mimic |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['joint']} | {r['urdf_type']} | {r['usd_type']} | {f(r['urdf_lower'])} / {f(r['usd_lower'])} | "
                 f"{f(r['urdf_upper'])} / {f(r['usd_upper'])} | {f(r['urdf_effort'])} / {f(r['usd_maxForce'])} | "
                 f"{f(r['urdf_velocity'])} / {f(r['usd_maxVel'])} | {r['urdf_mimic'] or ''} |")
    (od / "urdf_diff.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
