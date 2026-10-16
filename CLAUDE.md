# CLAUDE.md — dx_bench

DX허브 11층 디지털 트윈에서 Rainbow Robotics RB-Y1으로 VLA·World Action Model 벤치마크(침실 BED-01/02/03)를 만드는 레포.
새 세션은 이 파일 → `PROGRESS.md`(상태) → `DECISIONS.md` → `HUMAN_TODO.md` 순서로 읽고 시작한다.

## 고정 사항 (바꾸지 말 것)

- Isaac Sim 6.1.0 · Isaac Lab 3.0 (`isaaclab` 17.0.2, Arena 서브모듈 안 `91a0906`) · IsaacLab-Arena release/0.3.1 (`c7edf3a9`) · **PhysX 백엔드**
- conda: `dx_arena` = 시뮬·원격조작·원본 기록·정책 클라이언트 (**lerobot import 금지**, 패키지 설치는 `--dry-run`으로 고정 버전 불변 확인 후에만),
  `dx_policy` = LeRobot 변환·학습·정책 서버 (설치 가능, `VERSIONS.md`에 기록)
- `dx_arena` 고정 버전: isaacsim 6.1.0.0, torch 2.11.0+cu128, numpy 2.3.1, warp-lang 1.16.0, newton 1.5.2, websockets 16.1.1
- `submodules/` 원본 수정 금지. USD는 덮어쓰기 레이어/별도 파일, Arena 확장은 등록 API(`@register_asset`, external environment class path).
- `~/P2-DX_project`(DX 씬)과 conda `env_isaaclab`은 읽기만. 침실 씬: `~/P2-DX_project/DX/USD/DX_Hub_Building_add_obj_v1/*_CareRoom{,_scene}.usda`,
  자산 단독 USD: `.../assets/{Bed,TShirt,CareDesk,...}/*.usda`. 수납장은 `WallAnchor` 월드 고정 관절이 있으니 씬을 옮기지 말 것.
- git: 단계마다 로컬 커밋 `stage N: ...`, **push 금지**.
- 용량: 5 GB 넘는 모델/데이터 다운로드 금지, 데이터·결과 합계 50 GB 이하(`data/`, `outputs/`는 gitignore).

## RB-Y1 사실 (0단계 `reports/rby1_check/REPORT.md`)

- 로봇 USD: `tools/rby1_check/overrides/rby1_24dof.usda` (24-DOF, 자유 베이스, root = `/RBY1/base`). 2단계부터 `dxbench/assets/rby1/rby1_dx.usda`.
- 관절 순서: Isaac Lab BFS 순서 ≠ rby1-sdk C++ 순서 (보고서 §8 표).
- effort(URDF): 몸통 270/120, 팔 70/70/70/40/10/10/8 N·m. 게인 기본 `preset="tuned"`(`tools/rby1_check/rby1_cfg_draft.py`).
- 바퀴: +속도 = 후진, `joint_pos` 감김(unwrap), 반지름 0.1 m, 간격 0.53 m. 캐스터는 미끄러지는 실린더(μ0 사용).
- TCP = `ee_{side}` (0, 0, −0.1045) m, 접근축 ee −z, 손끝 −0.1355 m, 최대 개폭 10 cm. 그리퍼 l1/r1(0 닫힘, −0.05 열림), l2/r2는 거울.
- 카메라: `link_head_2/first_person_camera`, `{side}_gripper/ee_{side}/{side}_hand_camera`, 640×480, fx=fy=343.1.
- torso_1 단독 크게 숙이면 넘어짐 → 몸통은 torso_1·2·3 묶음 자세로. 몸통만으로 손끝 최저 0.30–0.375 m.

## Isaac Lab 3.0 / Kit 함정

- 쿼터니언 (x, y, z, w) — `InitialStateCfg.rot`, `body_link_quat_w`, 카메라 `quat_w_*` 모두.
- `CameraCfg(spawn=None)`엔 `update_latest_camera_pose=True` 필요. 관찰 카메라는 `{ENV_REGEX_NS}` 아래에(전역 카메라는 env prim을 못 그림).
- `data.mass_matrix` 대각에 armature 미포함. `data.gravity_compensation_forces`, `root_view.get_dof_projected_joint_forces()` 사용 가능.
- 변형체 정점: `PhysxManager.get_physics_sim_view().create_{surface,volume}_deformable_body_view(path).get_simulation_nodal_positions()`.
  `isaacsim.core.experimental.prims`를 Isaac Lab 앱에서 켜면 Kit이 죽었다.
- 기존 스테이지를 열어 쓸 때: `SimulationManager.set_physics_sim_device`는 스테이지를 연 뒤에. Isaac Lab SimulationContext가 처리.
- USD instanceable prim(`Flattened_Prototype_*`)은 `Usd.TraverseInstanceProxies`로 순회해야 충돌체·메시가 보인다.
- Kit EULA: `os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")`. Isaac Lab 앱은 Kit 로그 파일을 안 남김 → stdout 로그를 저장.
- Kit이 시작 1–2초에 `platforminfo`에서 segfault(약 8%) → 같은 명령 최대 3회 재시도.
- 파일을 python으로 패치할 때 `open(p,'w')` 전에 문자열 치환이 성공했는지 확인(0단계에서 파일이 0바이트가 된 적 있음).
- Arena: 외부 env 인자 이름이 Arena 인자의 접두어가 되면 argparse가 그쪽으로 매칭한다(`--preset` → `--presets`). 고유한 이름을 쓸 것.
- 지면을 크게 스케일한 얇은 Cube(40×40×0.1)로 만들면 GPU 접촉이 떨려 RB-Y1이 넘어진다 → `UsdGeom.Plane` 충돌체 사용.
- 고정 베이스(`fix_root_link`)에선 `body_link_jacobian_w`에 root가 없어서 body 인덱스 −1 (Isaac Lab DiffIK 액션은 처리함).
- surface deformable: kinematic target 없음. 정점 위치 덮어쓰기는 움직이는 큰 천을 폭주시킴 → 속도 쓰기로 고정(1단계 D-4).
  `DeformableObjectCfg`에 `UsdFileCfg(deformable_props=...)`를 주면 이미 API가 있는 자산(TShirt)에 시뮬 메시가 둘 생김 → props 없이 USD 속성 직접 설정.
  변형체 충돌 설정은 `collision_props=[PhysxCollisionCfg(...)]`(fragment 리스트).
- `MeshRectangleCfg.edge_refinement`는 trimesh `subdivide_to_size`로 반씩만 쪼갬 → 1.5 m 천의 노드 간격 4.7 / 2.3 / 1.2 cm.
- 바퀴 부호: 오른 바퀴는 −y. 반시계 yaw면 오른 바퀴가 빨라야 함(v_r = v + ω·W/2). +바퀴 관절 속도 = 후진.
- 몸통 deep 프리셋 + 팔 전방 뻗기 → 무게중심이 앞바퀴 축(x 0.228) 앞으로 나가 넘어짐. 바닥 집기 자세는 COM 여유를 확인할 것.
- 오른 손목 arm_5 상한 1.92 rad: 그리퍼 아래 방향 + 높은 TCP(z ≳ 1.1 m)에서 한계에 걸림.
- 안전 검사: `rm`에 글롭/변수 경로를 쓰면 막힌다. 결과 정리는 태그로 구분하거나 사람에게 맡길 것.
