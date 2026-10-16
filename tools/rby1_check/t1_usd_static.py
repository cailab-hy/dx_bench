"""T1: static inspection of the RB-Y1 USD files with Kit's pxr.

    python tools/rby1_check/t1_usd_static.py            # all files in common.USD_FILES
    python tools/rby1_check/t1_usd_static.py --only rbgripper_d405

Writes reports/rby1_check/t1/<key>.json and t1/summary.md. Also opens every file in the Kit
USD context once so that physics/schema deprecation warnings are captured from the Kit log.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")  # same convention as DX USD scripts

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--only", nargs="*", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from common import USD_FILES, kit_log_path, out_dir, save_json  # noqa: E402

PHYS_PREFIXES = ("physics:", "physx", "drive:", "limit:", "mujoco:", "isaac", "newton:")


def authored_phys_attrs(prim) -> dict:
    out = {}
    for a in prim.GetAuthoredAttributes():
        n = a.GetName()
        if n.startswith(PHYS_PREFIXES) or ":physics:" in n or n.startswith("state:"):
            try:
                v = a.Get()
            except Exception:  # noqa: BLE001
                v = "<err>"
            if isinstance(v, (Gf.Vec3f, Gf.Vec3d, Gf.Quatf, Gf.Quatd)):
                v = [float(x) for x in (list(v) if not hasattr(v, "GetReal") else [v.GetReal(), *v.GetImaginary()])]
            elif hasattr(v, "__len__") and not isinstance(v, str) and len(v) < 16:
                try:
                    v = [float(x) for x in v]
                except Exception:  # noqa: BLE001
                    v = str(v)
            elif not isinstance(v, (int, float, str, bool, type(None))):
                v = str(v)
            out[n] = v
    return out


def rel_targets(prim, name):
    r = prim.GetRelationship(name)
    return [str(t) for t in r.GetTargets()] if r else []


def mat_list(m):
    return [[float(m[i][j]) for j in range(4)] for i in range(4)]


def inspect(path) -> dict:
    stage = Usd.Stage.Open(str(path))
    res = {
        "file": str(path),
        "defaultPrim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage),
        "upAxis": str(UsdGeom.GetStageUpAxis(stage)),
        "kilogramsPerUnit": UsdPhysics.GetStageKilogramsPerUnit(stage),
        "used_layers": [l.identifier for l in stage.GetUsedLayers()],
        "applied_schemas": {},
        "prim_types": {},
        "articulation_roots": [],
        "joints": [],
        "links": [],
        "colliders": [],
        "filtered_pairs": [],
        "cameras": [],
        "frames": {},
        "issues": [],
    }
    xc = UsdGeom.XformCache()
    # visuals/collisions of the RB-Y1 assets are instanceable (internal Flattened_Prototype_* references),
    # so instance proxies must be traversed or every collider/mesh inside an instance is missed
    res["instances"] = sum(1 for p in stage.Traverse() if p.IsInstance())
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        t = prim.GetTypeName() or "<none>"
        res["prim_types"][t] = res["prim_types"].get(t, 0) + 1
        schemas = list(prim.GetAppliedSchemas())
        for s in schemas:
            res["applied_schemas"][s] = res["applied_schemas"].get(s, 0) + 1
        p = str(prim.GetPath())
        name = prim.GetName()

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            res["articulation_roots"].append({"path": p, "type": t, "attrs": authored_phys_attrs(prim)})

        if prim.IsA(UsdPhysics.Joint):
            j = {
                "name": name, "path": p, "type": t,
                "body0": rel_targets(prim, "physics:body0"), "body1": rel_targets(prim, "physics:body1"),
                "drives": {}, "attrs": authored_phys_attrs(prim), "schemas": schemas,
            }
            for s in schemas:
                if s.startswith("PhysicsDriveAPI:"):
                    inst = s.split(":", 1)[1]
                    d = UsdPhysics.DriveAPI(prim, inst)
                    j["drives"][inst] = {
                        "type": d.GetTypeAttr().Get(), "stiffness": d.GetStiffnessAttr().Get(),
                        "damping": d.GetDampingAttr().Get(), "maxForce": d.GetMaxForceAttr().Get(),
                        "targetPosition": d.GetTargetPositionAttr().Get(),
                        "targetVelocity": d.GetTargetVelocityAttr().Get(),
                    }
            res["joints"].append(j)

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(prim)
            link = {"name": name, "path": p, "attrs": authored_phys_attrs(prim)}
            mass = m.GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
            inertia = m.GetDiagonalInertiaAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
            link["mass"] = float(mass) if mass is not None else None
            link["diag_inertia"] = [float(x) for x in inertia] if inertia is not None else None
            link["world_tf"] = mat_list(xc.GetLocalToWorldTransform(prim))
            flags = []
            if mass is None or mass <= 0:
                flags.append("mass unauthored/<=0 (PhysX will derive from colliders/density)")
            if inertia is not None and (min(inertia) <= 0 or any(math.isnan(x) for x in inertia)):
                flags.append("inertia <=0 or NaN")
            if inertia is not None and min(inertia) > 0:
                a, b, c = sorted(inertia)
                if a + b < c * (1 - 1e-3):
                    flags.append("inertia violates triangle inequality")
            if mass and mass > 0 and inertia is not None and max(inertia) / mass > 1.0:
                flags.append("inertia/mass suspiciously large")
            link["flags"] = flags
            res["links"].append(link)

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            approx = None
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            en = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            res["colliders"].append({"path": p, "type": t, "approximation": approx, "enabled": en,
                                     "purpose": UsdGeom.Imageable(prim).GetPurposeAttr().Get() if prim.IsA(UsdGeom.Imageable) else None})

        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            res["filtered_pairs"].append({"path": p, "targets": rel_targets(prim, "physics:filteredPairs")})

        if prim.IsA(UsdGeom.Camera):
            c = UsdGeom.Camera(prim)
            g = c.GetCamera()
            f, ha, va = g.focalLength, g.horizontalAperture, g.verticalAperture
            W, H = 640, 480
            # Isaac uses horizontal aperture and square pixels: fy = fx
            fx = f / ha * W if ha else None
            res["cameras"].append({
                "path": p, "focalLength_mm": f, "horizontalAperture_mm": ha, "verticalAperture_mm": va,
                "clippingRange": list(g.clippingRange.GetMin() for _ in [0]) + [g.clippingRange.GetMax()],
                "projection": str(c.GetProjectionAttr().Get()),
                "intrinsics_640x480": {"fx": fx, "fy": fx, "cx": W / 2, "cy": H / 2,
                                        "hfov_deg": math.degrees(2 * math.atan(ha / (2 * f))) if f else None},
                "world_tf": mat_list(xc.GetLocalToWorldTransform(prim)),
                "attrs": {a.GetName(): str(a.Get()) for a in prim.GetAuthoredAttributes() if not a.GetName().startswith("xformOp")},
            })
        if name in ("ee_left", "ee_right", "link_head_2", "base", "link_torso_5", "tool_left", "tool_right") or "tcp" in name.lower():
            res["frames"].setdefault(name, []).append({"path": p, "type": t, "world_tf": mat_list(xc.GetLocalToWorldTransform(prim))})

    if len(res["articulation_roots"]) != 1:
        res["issues"].append(f"articulation roots = {len(res['articulation_roots'])}")
    for j in res["joints"]:
        j["role"] = ("tool_mount" if "tool" in j["name"] else
                     "fixed" if j["type"] == "PhysicsFixedJoint" else "dof")
    return res


def kit_open_warnings(path) -> list[str]:
    """Open the file in the Kit USD context and return warning/error log lines emitted meanwhile."""
    import omni.usd

    log = kit_log_path()
    start = log.stat().st_size if log else 0
    ctx = omni.usd.get_context()
    ctx.open_stage(str(path))
    for _ in range(20):
        app.update()
    lines = []
    if log and log.exists():
        with open(log, "r", errors="replace") as fh:
            fh.seek(start)
            for line in fh:
                if "[Warning]" in line or "[Error]" in line:
                    lines.append(line.strip())
    ctx.close_stage()
    app.update()
    return lines


def summarize(key, r) -> list[str]:
    L = [f"## {key}", "", f"- 파일: `{r['file']}`",
         f"- defaultPrim `{r['defaultPrim']}` · metersPerUnit {r['metersPerUnit']} · upAxis {r['upAxis']} · kgPerUnit {r['kilogramsPerUnit']}",
         f"- articulation root {len(r['articulation_roots'])}개: " + ", ".join(f"`{a['path']}`" for a in r["articulation_roots"]),
         f"- 관절 {len(r['joints'])}개 (dof {sum(j['role']=='dof' for j in r['joints'])}), 링크 {len(r['links'])}개, 충돌체 {len(r['colliders'])}개, 카메라 {len(r['cameras'])}개",
         f"- 사용 레이어 {len(r['used_layers'])}개"]
    approx = {}
    for c in r["colliders"]:
        k = f"{c['type']}/{c['approximation']}"
        approx[k] = approx.get(k, 0) + 1
    L.append("- 충돌체 근사: " + ", ".join(f"{k}×{v}" for k, v in sorted(approx.items())))
    for a in r["articulation_roots"]:
        sc = {k: v for k, v in a["attrs"].items() if "SelfCollision" in k or "solver" in k.lower() or "fixed" in k.lower()}
        L.append(f"- articulation 속성 `{a['path']}`: {sc}")
    bad = [l for l in r["links"] if l["flags"]]
    if bad:
        L.append(f"- ⚠ 질량/관성 플래그 {len(bad)}개 링크: " + "; ".join(f"{l['name']}: {', '.join(l['flags'])}" for l in bad[:12]))
    if r["issues"]:
        L.append("- ⚠ " + "; ".join(r["issues"]))
    if r.get("kit_warnings"):
        L.append(f"- Kit 로그 경고/오류 {len(r['kit_warnings'])}줄 (JSON `kit_warnings` 참고)")
    L.append("")
    dof = [j for j in r["joints"] if j["role"] != "fixed" or "tool" in j["name"]]
    if dof:
        L += ["| 관절 | 종류 | 축 | 하한 | 상한 | 드라이브 | K | D | maxForce | 기타 physx |", "|---|---|---|---|---|---|---|---|---|---|"]
        for j in dof:
            a = j["attrs"]
            ax = a.get("physics:axis", "")
            lo, hi = a.get("physics:lowerLimit", ""), a.get("physics:upperLimit", "")
            dr = next(iter(j["drives"].items()), (None, {}))
            extra = {k.split(":")[-1]: v for k, v in a.items() if k.startswith("physx") or "armature" in k or "riction" in k or "maxJointVelocity" in k}
            L.append(f"| {j['name']} | {j['type'].replace('Physics','')} | {ax} | {lo} | {hi} | {dr[0]}/{dr[1].get('type')} | {dr[1].get('stiffness')} | {dr[1].get('damping')} | {dr[1].get('maxForce')} | {extra} |")
        L.append("")
    if r["cameras"]:
        L += ["| 카메라 | f(mm) | hAp | vAp | fx@640 | hFOV° | clip |", "|---|---|---|---|---|---|---|"]
        for c in r["cameras"]:
            i = c["intrinsics_640x480"]
            L.append(f"| `{c['path']}` | {c['focalLength_mm']:.3f} | {c['horizontalAperture_mm']:.3f} | {c['verticalAperture_mm']:.3f} | {i['fx']:.1f} | {i['hfov_deg']:.1f} | {c['clippingRange']} |")
        L.append("")
    return L


def main():
    od = out_dir("t1")
    keys = args.only or list(USD_FILES)
    for k in keys:
        p = USD_FILES[k]
        print(f"[T1] {k}: {p}")
        r = inspect(p)
        r["kit_warnings"] = kit_open_warnings(p)
        save_json(od / f"{k}.json", r)
    # summary always covers every inspected file, not only this run's --only subset
    from common import load_json

    md = ["# T1 USD 정적 검사", ""]
    for k in USD_FILES:
        if (od / f"{k}.json").exists():
            md += summarize(k, load_json(od / f"{k}.json"))
    (od / "summary.md").write_text("\n".join(md) + "\n")
    print(f"[T1] wrote {od}")


if __name__ == "__main__":
    main()
    app.close()
