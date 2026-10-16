"""Assemble reports/rby1_check/REPORT.md from the JSON results.

    python tools/rby1_check/make_report.py

Measured tables are generated from t1/ t2/ t3/ t4/ JSON. Hand-written analysis lives in
report_analysis.md (sections are inserted at <!-- ANALYSIS:<name> --> markers).
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

from common import REPORTS, TOOLS

R = REPORTS


def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def f(v, n=4):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        return f"{v:.{n}g}"
    return str(v)


def ok(b):
    return "✅ 통과" if b else "❌ 실패"


def rel(p):
    return str(Path(p).relative_to(R)) if str(p).startswith(str(R)) else str(p)


def analysis_sections() -> dict[str, str]:
    p = TOOLS / "report_analysis.md"
    if not p.exists():
        return {}
    out, cur, buf = {}, None, []
    for line in p.read_text().splitlines():
        m = re.match(r"<!-- SECTION:(\w+) -->", line)
        if m:
            if cur:
                out[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1), []
        else:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).strip()
    return out


def t3(run):
    return J(R / "t3" / f"{run}.json")


def idle_row(name, run):
    d = t3(run)
    if not d:
        return f"| {name} | (없음) | | | | | | |"
    x = d["result"]
    if "error" in x:
        return f"| {name} | 오류 | {x['error'][:60]} | | | | | |"
    jm = max(x["max_err_by_joint_hold"].items(), key=lambda kv: kv[1])
    zc = sum(1 for v in x["err_zero_crossings_after_1s"].values() if v > 5)
    p2p = max(x["err_p2p_last3s_rad"].items(), key=lambda kv: kv[1])
    return (f"| {name} | {ok(x['pass'])} | {f(x['max_err_rad_all'])} ({jm[0]}) | {f(p2p[1])} ({p2p[0]}) | {zc} | "
            f"{f(x['base_disp_m_total'] * 1000, 3)} | {f(d['wall_s'], 3)} | `{rel(R / 't3' / (run + '.json'))}` |")


IDLE_HDR = ("| 조건 | 판정 | 최대 관절오차 rad (관절) | 마지막 3 s p2p rad | 진동 관절 수* | 베이스 이동 mm | wall s | 파일 |\n"
            "|---|---|---|---|---|---|---|---|")


def section_env():
    p = R / "env.md"
    return p.read_text().replace("# T0 환경 기록", "").strip() if p.exists() else "(env.md 없음)"


def section_t1():
    rows = ["| 파일 | defaultPrim | m/unit | up | root 수 | DOF | 고정 관절 | instance | 충돌체 (instance 포함) | 카메라 | Kit 경고 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted((R / "t1").glob("*.json")):
        d = J(p)
        dof = sum(j["role"] == "dof" for j in d["joints"])
        fixed = [j["name"] for j in d["joints"] if j["role"] != "dof"]
        cols = {}
        for c in d["colliders"]:
            k = f"{c['type']}/{c['approximation'] or '-'}"
            cols[k] = cols.get(k, 0) + 1
        rows.append(f"| `{p.stem}` | `{d['defaultPrim']}` | {d['metersPerUnit']} | {d['upAxis']} | "
                    f"{len(d['articulation_roots'])} | {dof} | {len(fixed)} | {d.get('instances', '—')} | "
                    f"{', '.join(f'{k}×{v}' for k, v in sorted(cols.items())) or '0'} | {len(d['cameras'])} | {len(d.get('kit_warnings', []))} |")
    return "\n".join(rows)


def section_t1_colliders():
    d = J(R / "t1" / "override_24dof.json")
    if not d:
        return ""
    by = {}
    for c in d["colliders"]:
        parts = c["path"].split("/")
        link = parts[2] if len(parts) > 2 else c["path"]
        if link in ("left_gripper", "right_gripper"):
            link = parts[3]
        by.setdefault(link, []).append(f"{c['type']}{'/' + c['approximation'] if c['approximation'] else ''}")
    links = [l["name"] for l in d["links"]]
    L = ["| 링크 | 충돌체 |", "|---|---|"]
    for n in links:
        v = by.get(n)
        L.append(f"| {n} | {', '.join(sorted(set(v))) + f' ×{len(v)}' if v else '**없음**'} |")
    return "\n".join(L)


def section_t2():
    d = J(R / "t2" / "t2.json")
    if not d:
        return "(t2 없음)"
    L = ["| 관절 | 항목 | URDF | USD |", "|---|---|---|---|"]
    for x in d["diff"]:
        L.append(f"| {x['joint']} | {x['item']} | {f(x['urdf'])} | {f(x['usd'])} |")
    bad = [k for k in d["kinematics"] if (k["pos_err_mm"] > 1.0 or k["axis_dot"] < 0.999) and k["joint"] != "head_base"]
    L += ["", f"영 자세 기구학 대조: 몸체 24관절의 원점 오차 최대 "
          f"{max(k['pos_err_mm'] for k in d['kinematics'] if not k['joint'].startswith(('gripper', 'head_base'))):.3f} mm, "
          f"축 내적 최소 {min(k['axis_dot'] for k in d['kinematics'] if not k['joint'].startswith(('gripper', 'head_base'))):+.4f}."]
    if bad:
        L.append("기준 밖: " + ", ".join(f"{k['joint']}(원점 {k['pos_err_mm']:.1f} mm, 축 {k['axis_dot']:+.3f})" for k in bad))
    return "\n".join(L)


def section_joint_order():
    d = next((J(p) for p in sorted((R / "t3").glob("idle_implicit_gpu_dt500_n1.json"))), None)
    if not d:
        return "(T3 결과 없음)"
    from common import CPP_JOINT_NAMES

    names = d["joint_names"]
    L = ["| Isaac Lab 인덱스 | 관절 | rby1-sdk C++ 인덱스 (`CPP_JOINT_NAMES`) | 한계 rad | effort 한계 | 속도 한계 |",
         "|---|---|---|---|---|---|"]
    for i, n in enumerate(names):
        c = CPP_JOINT_NAMES.index(n) if n in CPP_JOINT_NAMES else "— (그리퍼)"
        lim = d["joint_limits_rad"][n]
        lim_s = "무한 (continuous)" if abs(lim[0]) > 1e6 else f"[{lim[0]:.4f}, {lim[1]:.4f}]"
        L.append(f"| {i} | {n} | {c} | {lim_s} | {f(d['joint_effort_limits'][n])} | {f(d['joint_vel_limits'][n])} |")
    L += ["", "C++ 순서 → Isaac Lab 인덱스: `" + str([names.index(n) for n in CPP_JOINT_NAMES]) + "`",
          "", f"body 순서(첫 번째가 articulation root): `{d['body_names']}`"]
    return "\n".join(L)


def section_t3_tables():
    out = {}
    out["idle"] = "\n".join([IDLE_HDR] + [idle_row(n, r) for n, r in [
        ("implicit · GPU · 1/500 · 마찰 full (기본)", "idle_implicit_gpu_dt500_n1"),
        ("implicit · GPU · 1/200 · 마찰 full", "idle_implicit_gpu_dt200_n1"),
        ("ideal_pd(원본식) · GPU · 1/500 · full · effort=USD", "idle_ideal_pd_gpu_dt500_n1"),
        ("ideal_pd(원본식) · GPU · 1/200 · full · effort=USD", "idle_ideal_pd_gpu_dt200_n1"),
        ("implicit · GPU · 1/500 · ready 자세", "idle_implicit_gpu_dt500_n1_ready"),
        ("조정안 (점성 + ζ0.7 + 중력보상) · GPU · 1/500", "idle_implicit_gpu_dt500_n1_tuned_gc"),
        ("조정안 (점성 + ζ0.7 + 중력보상) · GPU · 1/200", "idle_implicit_gpu_dt200_n1_tuned_gc"),
    ]])
    study = []
    for dt in (500, 200):
        for fr, tag in (("none", "nofric"), ("viscous", "visc"), ("full", "")):
            base = f"idle_implicit_gpu_dt{dt}_n1" + (f"_{tag}" if tag else "")
            study.append((f"implicit · 1/{dt} · 마찰 {fr} · 원본 감쇠", base))
            if fr != "full":
                study.append((f"ideal_pd · 1/{dt} · 마찰 {fr} · 원본 감쇠", f"idle_ideal_pd_gpu_dt{dt}_n1_{tag}"))
            else:
                study.append((f"ideal_pd · 1/{dt} · 마찰 full · 원본 감쇠", f"idle_ideal_pd_gpu_dt{dt}_n1"))
            for z in ("0.7", "1.0"):
                ztag = f"{tag or 'full'}_z{z}"
                study.append((f"implicit · 1/{dt} · 마찰 {fr} · ζ={z}", f"idle_implicit_gpu_dt{dt}_n1_{ztag}"))
    study += [("implicit · CPU · 1/500 · 마찰 none", "idle_implicit_cpu_dt500_n1_nofric"),
              ("implicit · CPU · 1/500 · 마찰 viscous", "idle_implicit_cpu_dt500_n1_visc"),
              ("implicit · CPU · 1/500 · 마찰 full", "idle_implicit_cpu_dt500_n1")]
    out["study"] = "\n".join([IDLE_HDR] + [idle_row(n, r) for n, r in study])
    out["clone"] = "\n".join([IDLE_HDR] + [idle_row(n, r) for n, r in [
        ("GPU · num_envs 1", "idle_implicit_gpu_dt500_n1"), ("GPU · num_envs 4", "idle_implicit_gpu_dt500_n4"),
        ("CPU · num_envs 1", "idle_implicit_cpu_dt500_n1")]])
    d4 = t3("idle_implicit_gpu_dt500_n4")
    if d4:
        out["clone"] += "\n\nnum_envs 4 환경별 최대 관절오차: " + ", ".join(f(v) for v in d4["result"]["per_env_max_err_hold"])

    # damping ratio table (from the default run)
    d = t3("idle_implicit_gpu_dt500_n1")
    if d:
        L = ["| 관절 | K N·m/rad | D (=kd×0.002) | 점성마찰 | M_ii + armature kg·m² | f_n Hz | ζ 드라이브만 | ζ 점성 포함 | D(ζ=0.7) | D(ζ=1.0) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for n, v in d["result"]["damping_ratio"].items():
            L.append(f"| {n} | {f(v['K'])} | {f(v['D'])} | {f(v['viscous'])} | {f(v['M_ii'])} | {f(v['f_n_hz'], 3)} | "
                     f"{f(v['zeta_drive'], 3)} | {f(v['zeta_total'], 3)} | {f(v['D_for_zeta_0.7'], 3)} | {f(v['D_for_zeta_1.0'], 3)} |")
        out["zeta"] = "\n".join(L)

    # crouch
    L = ["| 조건 | 판정 | 유지 중 최대 오차 rad | 최악 토크 관절 | 투영 토크/effort | 중력보상/effort | 기울기° | ee z (L/R) m | 파일 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, r in [("(60,−90,60) implicit · 1/500 · 원본 게인", "crouch_implicit_gpu_dt500_n1"),
                 ("(60,−90,60) implicit · 1/200 · 원본 게인", "crouch_implicit_gpu_dt200_n1"),
                 ("(60,−90,60) ideal_pd · 1/500 · effort USD", "crouch_ideal_pd_gpu_dt500_n1"),
                 ("(60,−90,60) 점성만 + ζ0.7", "crouch_implicit_gpu_dt500_n1_tuned"),
                 ("(60,−90,60) 조정안 (점성 + ζ0.7 + 중력보상)", "crouch_implicit_gpu_dt500_n1_tuned_gc"),
                 ("(75,−105,90) 원본 게인", "crouch_implicit_gpu_dt500_n1_deep"),
                 ("(75,−105,90) 조정안", "crouch_implicit_gpu_dt500_n1_deep_tuned_gc"),
                 ("(90,−130,90) 원본 게인", "crouch_implicit_gpu_dt500_n1_deeper"),
                 ("(90,−130,90) 조정안", "crouch_implicit_gpu_dt500_n1_deeper_tuned_gc")]:
        dd = t3(r)
        if not dd:
            continue
        x = dd["result"]
        if "error" in x:
            L.append(f"| {n} | 오류 | {x['error'][:80]} | | | | | | |")
            continue
        w = x.get("worst_torque_joint")
        tq = x["torque_hold"][w] if w else {}
        L.append(f"| {n} | {ok(x['pass'])} | {f(x['max_err_rad_hold'])} | {w} | {f(tq.get('ratio_projected'), 3)} | "
                 f"{f(tq.get('ratio_gravity'), 3)} | {f(x['max_tilt_deg'], 3)} | {f(x['ee_z_m']['left'], 3)} / {f(x['ee_z_m']['right'], 3)} | `t3/{r}.json` |")
    dd = t3("crouch_implicit_gpu_dt500_n1")
    if dd and "torque_hold" in dd["result"]:
        L += ["", "기본 조건(implicit, 1/500) 관절별 유지 토크:", "",
              "| 관절 | 최대 오차 rad | applied 최대 | 투영 최대 | 중력보상 | URDF effort | 투영/effort |", "|---|---|---|---|---|---|---|"]
        x = dd["result"]
        for jn, v in x["torque_hold"].items():
            if jn.startswith(("torso", "right_arm_0", "right_arm_1", "right_arm_3", "head_1")):
                L.append(f"| {jn} | {f(x['max_err_by_joint_hold'][jn])} | {f(v['applied_max'], 3)} | {f(v['projected_max'], 3)} | "
                         f"{f(v['gravity_comp'], 3)} | {f(v['urdf_effort'])} | {f(v['ratio_projected'], 3)} |")
    out["crouch"] = "\n".join(L)

    # sweep
    parts = []
    for r in ("sweep_implicit_gpu_dt500_n1", "sweep_implicit_gpu_dt200_n1", "sweep_implicit_gpu_dt500_n1_fixroot",
              "sweep_implicit_gpu_dt500_n1_tuned_gc"):
        dd = t3(r)
        if not dd:
            continue
        x = dd["result"]
        if "error" in x:
            parts.append(f"`{r}`: 오류 {x['error'][:200]}")
            continue
        L = [f"**{r}** — {ok(x['pass'])} (주기 {x['period_s']} s, 한계의 ±80%)", "",
             "| 관절 | 명령 범위 rad | RMS 오차 | 최대 오차 | 오버슈트 상/하 | 도달 상/하 | 최대 토크 | 기울기° | 링크 최저 z m | 플래그 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for jn, v in x["joints"].items():
            L.append(f"| {jn} | [{v['range_cmd_rad'][0]:.3f}, {v['range_cmd_rad'][1]:.3f}] | {f(v['rms_err_rad'], 3)} | {f(v['max_err_rad'], 3)} | "
                     f"{f(v['overshoot_hi_rad'], 2)} / {f(v['overshoot_lo_rad'], 2)} | {f(v['reached_hi_rad'], 3)} / {f(v['reached_lo_rad'], 3)} | "
                     f"{f(v['max_tau_Nm'], 3)} | {f(v['max_tilt_deg'], 3)} | {f(v['min_link_origin_z_m'], 3)} | {', '.join(v['flags'])} |")
        parts.append("\n".join(L))
    out["sweep"] = "\n\n".join(parts)

    # gripper
    L = []
    for n, r in [("l2/r2 그대로(원본, 거의 수동) · 원본 게인", "gripper_implicit_gpu_dt500_n1"),
                 ("l2/r2 소프트웨어 mimic · 원본 게인", "gripper_implicit_gpu_dt500_n1_mimic"),
                 ("l2/r2 mimic · 조정안 (점성 + ζ0.7 + 중력보상)", "gripper_implicit_gpu_dt500_n1_mimic_tuned_gc")]:
        dd = t3(r)
        if not dd:
            continue
        x = dd["result"]
        if "error" in x:
            L.append(f"- {n}: 오류 `{x['error'][:200]}`")
            continue
        li = x.get("lift", {})
        L.append(f"- **{n}** — {ok(x['pass'])}: 좌우 비대칭 최대 {f(x['max_asym_m'] * 1000, 3)} mm, "
                 f"큐브 상승 {f(li.get('cube_rise_m'), 3)} m, 들기 중 미끄러짐 {f((li.get('slip_during_lift_m') or 0) * 1000, 3)} mm, "
                 f"10 s 유지 미끄러짐 {f((li.get('slip_hold_10s_m') or 0) * 1000, 3)} mm, 낙하 {f(li.get('dropped'))}, "
                 f"도달 오차 {f(x['reach_err']['grasp_pos_m'] * 1000, 3)} mm · 기준 {x['criteria']} (`t3/{r}.json`)")
        L.append("  - 열기/닫기: " + "; ".join(
            f"명령 {s['cmd_open_m']}: r1 {s['r1']:+.4f} r2 {s['r2']:+.4f} l1 {s['l1']:+.4f} l2 {s['l2']:+.4f}" for s in x["open_close"]))
    dd = t3("gripper_implicit_gpu_dt500_n1")
    if dd and "tcp_offset_in_ee_right_m" in dd["result"]:
        x = dd["result"]
        L.append(f"- TCP(손가락 충돌체 박스 중심, ee_right 기준): {[round(v, 4) for v in x['tcp_offset_in_ee_right_m']]} m, "
                 f"접근축 {x['approach_axis_in_ee_right']}, 손가락 박스 크기 {[round(v, 4) for v in x['finger_box_extent_m']]} m")
    out["gripper"] = "\n".join(L)

    # drive
    L = ["| 조건 | 캐스터 μ | 판정 | 직진 실제 / 명령 m | 오차(명령) | 오차(바퀴) | 바퀴 추종 | 회전 실제 / 명령 rad | 오차(명령) | 오차(바퀴) | 바퀴 추종 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in [("implicit 속도 드라이브(D=100) · 1/500", "drive_implicit_gpu_dt500_n1"),
                 ("implicit 속도 드라이브(D=100) · 1/200", "drive_implicit_gpu_dt200_n1"),
                 ("ideal_pd 원본식(적분 위치목표+kp/kd) · 1/500", "drive_ideal_pd_gpu_dt500_n1"),
                 ("implicit 속도 드라이브 · 1/500", "drive_implicit_gpu_dt500_n1_caster0"),
                 ("ideal_pd 원본식 · 1/500", "drive_ideal_pd_gpu_dt500_n1_caster0")]:
        dd = t3(r)
        if not dd:
            continue
        x = dd["result"]
        if "error" in x:
            L.append(f"| {n} | | 오류 | {x['error'][:80]} | | | | | | | |")
            continue
        s_, p_ = x["straight"], x["spin"]
        L.append(f"| {n} | {x.get('caster_mu', 0.3)} | {ok(x['pass'])} | {f(s_['forward_m'], 3)} / {f(s_['pred_from_cmd_m'], 3)} | "
                 f"{f(s_['err_vs_cmd'] * 100, 2)} % | {f(s_['err_vs_wheels'] * 100, 2)} % | {f((s_.get('wheel_tracking_err') or 0) * 100, 2)} % | "
                 f"{f(p_['yaw_rad'], 3)} / {f(p_['pred_yaw_from_cmd'], 3)} | {f(p_['err_vs_cmd'] * 100, 2)} % | "
                 f"{f(p_['err_vs_wheels'] * 100, 2)} % | {f((p_.get('wheel_tracking_err') or 0) * 100, 2)} % |")
    out["drive"] = "\n".join(L)

    # camera
    dd = t3("camera_implicit_gpu_dt500_n1")
    if dd:
        x = dd["result"]
        if "error" in x:
            out["camera"] = f"오류: `{x['error'][:300]}`"
        else:
            L = [f"판정 {ok(x['pass'])}", "",
                 "| 카메라 | RGB 평균/표준편차 | depth 유효 비율 | depth 범위 m | 방향 오차° | 위치 오차 mm | fx fy cx cy | 이미지 |",
                 "|---|---|---|---|---|---|---|---|"]
            for k in ("head", "left_wrist", "right_wrist"):
                v = x[k]
                K = v["K"]
                L.append(f"| {k} | {v['rgb_mean']:.1f} / {v['rgb_std']:.1f} | {v['depth_valid_frac']:.2f} | {v['depth_range_m']} | "
                         f"{v['orient_err_deg']:.3f} | {v['pos_err_m'] * 1000:.2f} | {K[0][0]:.1f} {K[1][1]:.1f} {K[0][2]:.1f} {K[1][2]:.1f} | "
                         + " ".join(f"`{rel(p)}`" for p in v["files"]) + " |")
            out["camera"] = "\n".join(L)
    return out


def section_t4():
    out = []
    for load, title in (("open", "방식 A — `*_CareRoom_scene.usda` 스테이지를 연 뒤 로봇 추가 (요청 방식)"),
                        ("reference", "방식 B — Isaac Lab 스테이지 + `CareRoom.usda`를 전역 asset으로 참조 (Arena 방식)")):
        d = J(R / "t4" / f"t4_{load}.json")
        out.append(f"#### {title}\n\n" + (_t4_one(d) if d else "(결과 없음)"))
    return "\n\n".join(out)


def _t4_one(d):
    if "error" in d:
        return f"오류: {d['error']}"
    L = [f"판정 {ok(d['pass'])} · 기준 {d['criteria']}", "",
         f"- 물리 장치 {d['physics_device']}, 씬 PhysicsScene 재사용 {d['physics_scene_reused']}, dt {d['dt']:.4f}, {d['duration_s']} s",
         f"- 시작 위치 {d['spawn']}, yaw {d['yaw_deg']}°",
         f"- 베이스 최대 이동 {d['base_disp_max_m'] * 1000:.2f} mm, 최대 기울기 {d['base_tilt_max_deg']:.3f}°, "
         f"관절 최대 오차 {d['joint_err_max_rad']:.4f} rad (최종 {d['joint_err_final_rad']:.4f})",
         f"- 정적 겹침(링크 AABB vs 가구 AABB): {len(d['static_overlaps'])}건"
         + (": " + ", ".join(f"{c['link']}↔{c['furniture']} ({c['gap_m']:.3f} m, 충돌체 {'있음' if c['link_has_collider'] else '없음'})" for c in d["static_overlaps"][:10]) if d["static_overlaps"] else ""),
         "- 가장 가까운 가구: " + ", ".join(f"{c['link']}↔{c['furniture']} {c['gap_m']:.3f} m" for c in d["static_nearest_furniture"][:5]),
         f"- 충돌체 없는 링크 {len(d['links_without_collider'])}개: {', '.join(d['links_without_collider'])}",
         f"- 머리 카메라: 침대 중심 화면 안 {f(d['head_camera']['bed_centre_visible'])}, 침대 AABB 꼭짓점 {d['head_camera']['bed_corners_in_image']}/8개 화면 안, "
         f"중심 픽셀 {[round(v) for v in d['head_camera']['bed_centre_px']]}",
         "- 변형체(실행 전 → 후): " + "; ".join(
             f"{k}: " + (f"z {v0['z_min']:.3f}~{v0['z_max']:.3f} → {v1['z_min']:.3f}~{v1['z_max']:.3f}, finite {v1['finite']}"
                         if isinstance(v0, dict) and isinstance(v1, dict) and "z_min" in v1 else f"{v1}")
             for k, v0, v1 in ((k, d["deformables_before"].get(k), d["deformables_after"].get(k)) for k in ("pillow", "blanket", "tshirt"))),
         f"- 실행 로그 경고(잡음 제외): {_log_warnings(R / f't4_run_{d.get("load", "open")}.log')}",
         "- 이미지: " + ", ".join(f"`{rel(v)}`" for v in d["images"].values())]
    return "\n".join(L)


NOISE = ("omni.ext.plugin", "hydra.engine.stats", "materialx", "Serving remote assets", "context_menu",
         "powersave", "already registered", "No material configuration", "pxr.Semantics", "USD Diagnostics are",
         "crashreporter")


def _log_warnings(path) -> str:
    """Unique non-noise [Warning]/[Error] messages of one run log (stdout of the Kit process)."""
    p = Path(path)
    if not p.exists():
        return "(로그 없음)"
    seen = {}
    for line in p.read_text(errors="replace").splitlines():
        if ("[Warning]" in line or "[Error]" in line) and not any(n in line for n in NOISE):
            msg = re.sub(r"^\S+ \[[0-9,]+ms\] ", "", line).strip()
            msg = re.sub(r"env_\d+", "env_N", msg)
            seen[msg] = seen.get(msg, 0) + 1
    if not seen:
        return "없음"
    return "; ".join(f"`{m[:160]}` ×{c}" for m, c in sorted(seen.items(), key=lambda kv: -kv[1])[:6])


def summary_rows():
    def p(run):
        d = t3(run)
        return d["result"].get("pass") if d else None

    t4 = J(R / "t4" / "t4_open.json")
    t4b = J(R / "t4" / "t4_reference.json")
    return [
        ("T0 환경 기록", True, "env.md"),
        ("T1 USD 정적 검사", True, "t1/summary.md — 발견 사항은 아래"),
        ("T2 URDF 대조", True, "t2/urdf_diff.md"),
        ("T3a 정지 유지 · GPU 1/500 (기본)", p("idle_implicit_gpu_dt500_n1"), "idle_implicit_gpu_dt500_n1"),
        ("T3a 정지 유지 · GPU 1/200", p("idle_implicit_gpu_dt200_n1"), "idle_implicit_gpu_dt200_n1"),
        ("T3a 정지 유지 · 조정안 · GPU 1/200", p("idle_implicit_gpu_dt200_n1_tuned_gc"), "idle_implicit_gpu_dt200_n1_tuned_gc"),
        ("T3b 숙임 자세 유지 (원본 게인)", p("crouch_implicit_gpu_dt500_n1"), "crouch_implicit_gpu_dt500_n1"),
        ("T3b 숙임 자세 유지 (조정안, 3자세)", all(p(r) for r in ("crouch_implicit_gpu_dt500_n1_tuned_gc",
            "crouch_implicit_gpu_dt500_n1_deep_tuned_gc", "crouch_implicit_gpu_dt500_n1_deeper_tuned_gc")), "crouch_*_tuned_gc"),
        ("T3c 관절 스윕 (베이스 자유)", p("sweep_implicit_gpu_dt500_n1"), "sweep_implicit_gpu_dt500_n1"),
        ("T3c 관절 스윕 (몸통, 베이스 고정)", p("sweep_implicit_gpu_dt500_n1_fixroot"), "sweep_implicit_gpu_dt500_n1_fixroot"),
        ("T3d 그리퍼 (원본 l2/r2)", p("gripper_implicit_gpu_dt500_n1"), "gripper_implicit_gpu_dt500_n1"),
        ("T3d 그리퍼 (mimic)", p("gripper_implicit_gpu_dt500_n1_mimic"), "gripper_implicit_gpu_dt500_n1_mimic"),
        ("T3d 그리퍼 (mimic + 조정안)", p("gripper_implicit_gpu_dt500_n1_mimic_tuned_gc"), "gripper_implicit_gpu_dt500_n1_mimic_tuned_gc"),
        ("T3e 바퀴 주행 (implicit, 캐스터 μ0.3)", p("drive_implicit_gpu_dt500_n1"), "drive_implicit_gpu_dt500_n1"),
        ("T3e 바퀴 주행 (원본식, 캐스터 μ0.3)", p("drive_ideal_pd_gpu_dt500_n1"), "drive_ideal_pd_gpu_dt500_n1"),
        ("T3e 바퀴 주행 (원본식, 캐스터 μ0)", p("drive_ideal_pd_gpu_dt500_n1_caster0"), "drive_ideal_pd_gpu_dt500_n1_caster0"),
        ("T3f 카메라", p("camera_implicit_gpu_dt500_n1"), "camera_implicit_gpu_dt500_n1"),
        ("T3g GPU/CPU · 복제 4", all(p(r) for r in ("idle_implicit_gpu_dt500_n1", "idle_implicit_gpu_dt500_n4",
                                                     "idle_implicit_cpu_dt500_n1")), "idle_*_n4 / idle_implicit_cpu_*"),
        ("T4 침실 씬 배치 (스테이지 열기)", t4.get("pass") if t4 else None, "t4/t4_open.json"),
        ("T4 침실 씬 배치 (참조, Arena 방식)", t4b.get("pass") if t4b else None, "t4/t4_reference.json"),
    ]


def main():
    A = analysis_sections()
    T3 = section_t3_tables()
    L = ["# RB-Y1 USD 검증 보고서 (0단계 마무리)", "",
         "Isaac Sim 6.1.0 · Isaac Lab 3.0 (isaaclab 17.0.2) · PhysX GPU · conda `dx_arena`. "
         "재실행: `tools/rby1_check/README.md`. 이 파일은 `make_report.py`가 JSON 결과와 `report_analysis.md`로 만든다.", "",
         A.get("summary", ""), "",
         "## 1. 테스트별 결과", "",
         "| 테스트 | 판정 | 결과 파일 |", "|---|---|---|"]
    for n, p, fn in summary_rows():
        L.append(f"| {n} | {'—' if p is None else ok(p)} | `{fn}` |")
    L += ["", A.get("verdict_notes", ""), "",
          "## 2. 환경 (T0)", "", section_env(), "",
          "## 3. USD 정적 검사 (T1)", "", section_t1(), "", A.get("t1", ""), "",
          "#### 링크별 충돌체 (`overrides/rby1_24dof.usda`, instance proxy 포함)", "", section_t1_colliders(), "",
          "## 4. URDF 대조 (T2)", "", "차이 나는 항목만 (단위: rad, rad/s, m, N·m, N):", "", section_t2(), "", A.get("t2", ""), "",
          "## 5. Isaac Lab 로드와 동작 (T3)", "",
          "### T3a 정지 유지 (10 s, 영 자세)", "", "기준: NaN 없음 · 관절 오차 ≤ 0.02 rad · 베이스 이동 ≤ 1 cm. "
          "*진동 관절 수 = 1 s 이후 오차 부호가 6번 이상 바뀐 관절 수.", "", T3.get("idle", ""), "", A.get("t3a", ""), "",
          "#### 감쇠·마찰 비교 — (a) 원본식 명시적 PD, (b) 감쇠비 조정", "", T3.get("study", ""), "", A.get("study", ""), "",
          "#### 감쇠비 표 (기본 조건, 영 자세)", "", T3.get("zeta", ""), "",
          "### T3b 숙임 자세 유지", "", T3.get("crouch", ""), "", A.get("t3b", ""), "",
          "### T3c 관절 스윕", "", T3.get("sweep", ""), "", A.get("t3c", ""), "",
          "### T3d 그리퍼", "", T3.get("gripper", ""), "", A.get("t3d", ""), "",
          "### T3e 바퀴 주행", "", T3.get("drive", ""), "", A.get("t3e", ""), "",
          "### T3f 카메라", "", T3.get("camera", ""), "", A.get("t3f", ""), "",
          "### T3g 물리 장치·복제", "", T3.get("clone", ""), "", A.get("t3g", ""), "",
          "## 6. DX 침실 씬 배치 (T4)", "", section_t4(), "", A.get("t4", ""), "",
          "## 7. 실패 원인과 수정 제안", "", A.get("fixes", ""), "",
          "## 8. 2단계용 사실 정리", "", A.get("facts", ""), "",
          "### 관절 순서 (Isaac Lab ↔ rby1-sdk)", "", section_joint_order(), "",
          "## 9. 이미지·로그 경로", "", A.get("files", ""), "",
          "대표 실행의 로그 경고(잡음 제외):", "",
          f"- T1 (USD 진단 음소거 해제): {_log_warnings(R / 't1_run.log')}",
          f"- T3a 기본: {_log_warnings(R / 't3' / 'logs' / 'idle_impl_500.log')}",
          f"- T3f 카메라: {_log_warnings(R / 't3' / 'logs' / 'camera_impl_500.log')}"]
    (R / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"wrote {R / 'REPORT.md'}")


if __name__ == "__main__":
    main()
