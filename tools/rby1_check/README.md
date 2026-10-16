# rby1_check — RB-Y1 USD 검증 (0단계 마무리)

Isaac Sim 6.1.0 / Isaac Lab 3.0 (isaaclab 17.0.2, PhysX 백엔드) / conda `dx_arena`에서 RB-Y1 USD가
GPU 물리로 동작하는지 검증하는 스크립트 모음. 결과는 `~/dx_bench/reports/rby1_check/`에 쌓이고
`make_report.py`가 `REPORT.md`로 묶는다.

모든 Kit 스크립트는 headless가 기본이고 `--viz kit`을 주면 창을 띄운다. EULA는 DX USD 스크립트와 같은
방식(`OMNI_KIT_ACCEPT_EULA=YES` setdefault)으로 처리한다. `submodules/` 원본은 읽기만 한다.

```bash
conda activate dx_arena
cd ~/dx_bench/tools/rby1_check
```

## 순서

| 단계 | 명령 | 산출물 |
|---|---|---|
| T0 환경 | `python t0_env.py` | `env.md` |
| T1 USD 정적 검사 | `python t1_usd_static.py [--only KEY ...]` | `t1/<key>.json`, `t1/summary.md` |
| URDF 받기 | `bash fetch_urdf.sh` (rby1-sdk sparse clone → `_ext/`, gitignore) | `_ext/rby1-sdk/models/rby1a/` |
| T2 URDF 대조 | `python t2_urdf_compare.py` | `t2/urdf_diff.md`, `t2/t2.json` |
| 24-DOF 레이어 | `python make_override.py` | `overrides/rby1_24dof.usda` |
| T3 동작 | `python t3_load.py --test <name> [opts]` | `t3/<run>.json`, `t3/*.png`, `images/t3f_*` |
| T3a/T3g 매트릭스 | `bash run_idle_matrix.sh` | `t3/idle_*.json` |
| T3a 감쇠·마찰 비교 | `bash run_damping_study.sh` | `t3/idle_*_{nofric,visc,full}_z*.json` |
| T3 나머지 | `bash run_t3_rest.sh` | `t3/{crouch,sweep,gripper,drive,camera}_*.json` |
| T4 침실 | `python t4_bedroom.py --load open\|reference [--yaw 0]` | `t4/t4_<load>.json`, `t4/*_<load>.png` |
| 보고서 | `python make_report.py` | `REPORT.md` |

`run_all.sh`는 위 전체를 차례로 돈다(약 1시간 30분, RTX 6000 Ada 기준). Kit이 시작 1–2초 안에
`platforminfo`(XOpenDisplay/getenv)에서 가끔 segfault로 죽는데(24회 중 2회), 그 실행만 다시 돌리면 된다.

## t3_load.py 옵션

| 옵션 | 뜻 |
|---|---|
| `--test idle\|crouch\|sweep\|gripper\|drive\|camera` | T3a / T3b / T3c / T3d / T3e / T3f |
| `--device cuda:0\|cpu` | 물리 장치 (T3g) |
| `--num_envs N` | 복제 환경 수 (T3g) |
| `--dt 500\|200` | physics Hz |
| `--mode implicit\|ideal_pd` | implicit 관절 드라이브(기본) / 원본식 명시적 PD |
| `--effort urdf\|usd` | effort 한계: URDF(기본) / USD maxForce(원본이 클리핑하는 값) |
| `--damping_ratio Z` | 몸체 관절 감쇠를 ζ=Z가 되게 덮어씀 (현재 자세의 질량행렬 대각 + armature 기준) |
| `--friction full\|viscous\|none` | 관절 마찰: 원본 모델 전체(정지+동+점성) / 점성만 / 없음 (`--no_friction` = none) |
| `--gravity_comp` | 중력보상 토크를 effort target으로 더함 (원본 PD의 SDK feedforward 항에 해당) |
| `--caster_mu M` | 베이스 캐스터 실린더 마찰 (원본 0.3, 캐스터가 구르지 않는 고정 실린더라 회전 시 끌림) |
| `--fix_root` | 베이스 고정 (몸통 스윕을 넘어짐 없이 재기 위함) |
| `--spawn x,y,z` | 로봇 시작 위치 |
| `--pose zero\|ready` | T3a 시작 자세 |
| `--crouch a,b,c` | T3b의 torso_1,2,3 각도(deg), 기본 `60,-90,60` |
| `--grip_mimic` | T3d에서 l2/r2를 l1/r1의 거울 목표로 구동 |
| `--sweep_joints j1,j2` | T3c 대상 관절 제한 |
| `--viz kit` | 화면 확인 |

"조정안(tuned)" = `--friction viscous --damping_ratio 0.7 --gravity_comp`. T3d의 조정안은 여기에 `--grip_mimic`을 더한다.

## 로봇 USD

`overrides/rby1_24dof.usda`는 `assets/generated/rby1_v1_2_a_rbgripper_d405_collision.usd`를 sublayer로
두고 다음만 덮어쓴다.

- 바퀴 2·몸통 6·머리 2 관절: `PhysicsFixedJoint` → `PhysicsRevoluteJoint` + `PhysicsDriveAPI:angular`
- `joints/fixed_base` 비활성 (베이스가 월드에 고정돼 있었음)
- ArticulationRootAPI를 `/RBY1` Xform에서 `/RBY1/base` 강체로 이동 (그대로 두면 PhysX가 floating root로
  `link_torso_5`를 고름)

## 구현할 때 걸린 함정 (2단계에서도 그대로 적용됨)

- Isaac Lab 3.0 쿼터니언은 **(x, y, z, w)** (`InitialStateCfg.rot`, `body_link_quat_w`, 카메라 `quat_w_*` 모두).
- 바퀴처럼 한계 없는 revolute 관절의 `joint_pos`는 감겨서 나온다(한 스텝에 4π 점프 관측). 위치 목표로 바퀴를 몰면
  오차를 unwrap해야 한다.
- 바퀴 관절 프레임이 x축 180° 뒤집혀 있어 **+바퀴 속도 = 후진**.
- `CameraCfg(spawn=None)`로 기존 카메라 prim을 쓸 때 `update_latest_camera_pose=True`가 아니면
  `data.pos_w / quat_w_*`가 초기화 시점 자세에 멈춰 있다.
- `{ENV_REGEX_NS}` 밖(`/World/...`)에 둔 카메라에는 env 안의 prim(로봇 포함)이 렌더링되지 않는다. 관찰용
  카메라도 env 아래에 둔다.
- PhysX 질량행렬 대각(`data.mass_matrix`)에는 joint armature가 들어 있지 않다.
- RB-Y1 USD의 visual/collision은 instanceable(`Flattened_Prototype_*`)이라 `Usd.TraverseInstanceProxies` 없이
  순회하면 충돌체·메시가 안 보인다.
- 열린 스테이지를 쓰는 `isaacsim.core.experimental.prims`는 Isaac Lab 앱에서 켜면 Kit이 죽었다. 변형체 노드는
  `PhysxManager.get_physics_sim_view().create_{surface,volume}_deformable_body_view(path)`로 읽는다.
- Isaac Lab 앱은 `~/.nvidia-omniverse/logs` Kit 로그를 쓰지 않는다. 경고는 각 실행의 stdout 로그(`t3/logs/`, `t4_run_*.log`)에 있다.

## 설정 초안

`rby1_cfg_draft.py`의 `make_rby1_cfg(mode=...)`. 게인 해석과 근거는 파일 머리 주석과 REPORT.md
"2단계용 사실 정리"를 본다.
