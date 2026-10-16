# RB-Y1 USD 검증 보고서 (0단계 마무리)

Isaac Sim 6.1.0 · Isaac Lab 3.0 (isaaclab 17.0.2) · PhysX GPU · conda `dx_arena`. 재실행: `tools/rby1_check/README.md`. 이 파일은 `make_report.py`가 JSON 결과와 `report_analysis.md`로 만든다.

## 요약

**결론: RB-Y1은 Isaac Sim 6.1 / Isaac Lab 3.0 PhysX GPU 물리에서 로드되고 안정적으로 돈다. DX 침실 씬 안에서도 변형체(이불·베개·티셔츠)와 함께 GPU로 돈다.** 다만 2단계로 가기 전에 정해야 할 것이 세 가지 있다.

1. **로봇 USD:** 연구실 `generated/` USD 3종은 바퀴·몸통·머리를 고정한 18-DOF 버전이고 베이스도 월드에 고정돼 있다. 승인받은 덮어쓰기 레이어 `tools/rby1_check/overrides/rby1_24dof.usda`로 24-DOF와 자유 베이스를 되살렸다. 같은 레이어에서 articulation root를 `base`로 옮겼다(그대로 두면 PhysX가 `link_torso_5`를 root로 고름).
2. **게인:** "원본 재현 게인"(stiffness = kp, damping = kd × 0.002)은 감쇠비 ζ가 0.01~0.04이다. 원본은 관절 마찰 모델이 진동을 눌러서 버텼다.
   - 1/500 s에서는 정지 유지를 통과하고, 1/200 s에서는 GPU 마찰 크리프 때문에 실패한다.
   - 중력보상이 없어서 숙임 자세가 처진다(오차 0.09 rad).
   - **조정안**(점성 마찰만 + ζ = 0.7 감쇠 + 중력보상 feedforward)은 1/200 s와 1/500 s 모두에서 정지 유지·숙임·스윕·그리퍼를 통과한다. 게인 체계 변경이라 **승인이 필요하다**. 설정 초안에는 `preset="tuned"`로만 넣고, 기본값은 원본 재현 게인으로 두었다.
3. **그리퍼와 베이스 모델:**
   - 그리퍼 두 번째 손가락(l2/r2)은 사실상 수동 관절이다. 소프트웨어 mimic으로 해결된다.
   - 베이스 캐스터는 구르지 않는 고정 실린더다. 원본 마찰 μ 0.3에서는 제자리 회전 오차가 54~66%이고, μ 0에서는 8.8%로 통과한다.

## 1. 테스트별 결과

| 테스트 | 판정 | 결과 파일 |
|---|---|---|
| T0 환경 기록 | ✅ 통과 | `env.md` |
| T1 USD 정적 검사 | ✅ 통과 | `t1/summary.md — 발견 사항은 아래` |
| T2 URDF 대조 | ✅ 통과 | `t2/urdf_diff.md` |
| T3a 정지 유지 · GPU 1/500 (기본) | ✅ 통과 | `idle_implicit_gpu_dt500_n1` |
| T3a 정지 유지 · GPU 1/200 | ❌ 실패 | `idle_implicit_gpu_dt200_n1` |
| T3a 정지 유지 · 조정안 · GPU 1/200 | ✅ 통과 | `idle_implicit_gpu_dt200_n1_tuned_gc` |
| T3b 숙임 자세 유지 (원본 게인) | ❌ 실패 | `crouch_implicit_gpu_dt500_n1` |
| T3b 숙임 자세 유지 (조정안, 3자세) | ✅ 통과 | `crouch_*_tuned_gc` |
| T3c 관절 스윕 (베이스 자유) | ❌ 실패 | `sweep_implicit_gpu_dt500_n1` |
| T3c 관절 스윕 (몸통, 베이스 고정) | ✅ 통과 | `sweep_implicit_gpu_dt500_n1_fixroot` |
| T3d 그리퍼 (원본 l2/r2) | ❌ 실패 | `gripper_implicit_gpu_dt500_n1` |
| T3d 그리퍼 (mimic) | ❌ 실패 | `gripper_implicit_gpu_dt500_n1_mimic` |
| T3d 그리퍼 (mimic + 조정안) | ✅ 통과 | `gripper_implicit_gpu_dt500_n1_mimic_tuned_gc` |
| T3e 바퀴 주행 (implicit, 캐스터 μ0.3) | ❌ 실패 | `drive_implicit_gpu_dt500_n1` |
| T3e 바퀴 주행 (원본식, 캐스터 μ0.3) | ❌ 실패 | `drive_ideal_pd_gpu_dt500_n1` |
| T3e 바퀴 주행 (원본식, 캐스터 μ0) | ✅ 통과 | `drive_ideal_pd_gpu_dt500_n1_caster0` |
| T3f 카메라 | ✅ 통과 | `camera_implicit_gpu_dt500_n1` |
| T3g GPU/CPU · 복제 4 | ✅ 통과 | `idle_*_n4 / idle_implicit_cpu_*` |
| T4 침실 씬 배치 (스테이지 열기) | ✅ 통과 | `t4/t4_open.json` |
| T4 침실 씬 배치 (참조, Arena 방식) | ✅ 통과 | `t4/t4_reference.json` |

판정은 요청서의 제안 기준 그대로다. "원본 게인" 행이 실패하고 "조정안" 행이 통과하는 항목은, 실패 원인이 USD나 물리 엔진이 아니라 게인·제어 방식에 있다는 뜻이다.
- **T3c 베이스 자유:** 실패 원인은 torso_1 단독 +72°에서 로봇이 앞으로 넘어지는 것 하나다. 상체 전체가 앞바퀴 축 밖으로 나가는 자세라 실기에서도 넘어진다.
- **T3c 베이스 고정:** 넘어짐 기준만 보므로 통과로 표시되지만, torso_1은 effort 270 N·m에서 포화되어 72°를 버티지 못한다(최대 오차 1.99 rad).

## 2. 환경 (T0)

| 항목 | 값 |
|---|---|
| Host OS | Linux-6.8.0-138-generic-x86_64-with-glibc2.35 |
| Python | 3.12.15 (/home/cai/miniforge3/envs/dx_arena/bin/python) |
| GPU / driver / VRAM | NVIDIA RTX 6000 Ada Generation, 580.178.04, 49140 MiB |
| CUDA (driver) | CUDA Version: 13.0 |
| CUDA (torch build) | 12.8 (available=True) |
| isaacsim | 6.1.0.0 |
| Isaac Lab | Isaac Lab 3.0 (isaaclab 17.0.2) |
| isaaclab_physx | 6.0.0 |
| isaaclab_newton | 5.4.1 |
| isaaclab_arena | 0.3.0 |
| torch | 2.11.0+cu128 |
| numpy | 2.3.1 |
| warp-lang | 1.16.0 |
| newton | 1.5.2 |
| lerobot (must be absent in dx_arena) | (not installed) |

## 커밋

| 레포 | 커밋 |
|---|---|
| IsaacLab-Arena | `c7edf3a9d5` (remotes/origin/release/0.3.1) |
| Isaac Lab (Arena submodule) | `91a0906f93` (tags/v3.0.0-EA-1-g91a0906f9) |
| rby1-sim-isaac | `ba5249629d` (heads/main) |

물리 백엔드: PhysX (`isaaclab_physx`) 고정. Newton 비교는 1단계에서 별도로 진행.

## 3. USD 정적 검사 (T1)

| 파일 | defaultPrim | m/unit | up | root 수 | DOF | 고정 관절 | instance | 충돌체 (instance 포함) | 카메라 | Kit 경고 |
|---|---|---|---|---|---|---|---|---|---|---|
| `fullbody_collision` | `/RBY1` | 1.0 | Z | 1 | 18 | 14 | 47 | Capsule/-×15, Cylinder/-×6, Mesh/convexDecomposition×4, Mesh/convexHull×2 | 3 | 0 |
| `gripper_left` | `/Gripper_Left` | 1.0 | Z | 1 | 2 | 0 | 3 | 0 | 0 | 0 |
| `gripper_right` | `/Gripper_Right` | 1.0 | Z | 1 | 2 | 0 | 3 | 0 | 0 | 0 |
| `model_a` | `/RBY1_A_v1_2` | 1.0 | Z | 1 | 24 | 1 | 41 | Capsule/-×17, Cylinder/-×6 | 0 | 0 |
| `override_24dof` | `/RBY1` | 1.0 | Z | 1 | 28 | 3 | 47 | Capsule/-×17, Cylinder/-×6, Mesh/convexDecomposition×4 | 3 | 0 |
| `rbgripper_d405` | `/RBY1` | 1.0 | Z | 1 | 18 | 14 | 47 | Capsule/-×17, Cylinder/-×6 | 3 | 0 |
| `rbgripper_d405_collision` | `/RBY1` | 1.0 | Z | 1 | 18 | 14 | 47 | Capsule/-×17, Cylinder/-×6, Mesh/convexDecomposition×4 | 3 | 0 |

**발견 사항**

- **18-DOF 고정 버전:** `rby1_v1_2_a_rbgripper_d405{,_collision}.usd`와 `rby1_fullbody_collision.usd`는 바퀴 2·몸통 6·머리 2 관절이 `PhysicsFixedJoint`이고 `joints/fixed_base`(body0 없음 = 월드)가 베이스를 고정한다. 한계·드라이브 속성은 prim에 그대로 남아 있어, 덮어쓰기 레이어에서 타입만 `PhysicsRevoluteJoint`로 바꾸고 `PhysicsDriveAPI:angular`를 다시 적용했다. 원본 시뮬레이터는 24-DOF `model_v_1_2_a.usd`에 그리퍼를 실행 중에 붙이는 방식이다.
- **articulation root:** `/RBY1`(강체가 아닌 Xform)에 붙어 있다. 떠 있는 베이스일 때 PhysX는 트리 중심에 가까운 `link_torso_5`(높이 1.29 m)를 root로 고른다. 그러면 Isaac Lab의 `root_pose`가 몸통 기준이 되고 "베이스 이동"도 몸통 흔들림을 재게 된다. 레이어에서 API를 `/RBY1/base`로 옮겼다.
- **충돌체:** visual과 collision이 instanceable prim(`Flattened_Prototype_*` 내부 참조) 안에 있다. 첫 검사 때 instance proxy를 순회하지 않아 "충돌체가 거의 없다"고 잘못 보고했다. 아래 표가 instance proxy까지 포함한 정정 결과다.
  - 충돌체가 없는 링크: `link_torso_3`, `link_head_0/1/2`, `link_{left,right}_arm_6`, `ee_{left,right}`(그리퍼 몸체).
  - 기본 USD의 손가락에는 충돌체가 없고, `_collision` 변형에만 convexDecomposition 4개가 있다.
- **arm_5 캡슐 비균일 스케일:** PhysX가 시뮬마다 "Non-uniform scale may result in a non matching collision representation" 경고를 낸다. `rby1_fullbody_collision.usd`는 이 둘을 convexHull로 굽고 arm_3↔arm_5 FilteredPairs를 추가해 둔 변형이다.
- **질량:** 총 132.2 kg(base 60.1 kg). 0, 음수, 비정상 관성(삼각부등식 위반)은 없다.
- **자기충돌:** 모든 파일이 `physxArticulation:enabledSelfCollisions = 0`이다(원본 코드도 끔).
- **tool 관절:** `tool_{left,right}`는 arm_6 → `{side}_gripper/ee_{side}` 고정 관절이고 localPos0 = (0, 0, −0.1261)로 `gripper.json` mount_pos와 같다. 그런데 USD xform상 ee는 arm_6 아래 0.1101 m에 놓여 있다(16 mm 차이). 시뮬레이션에서는 관절 제약이 이겨서 ee가 −0.1261에 놓인다(T3에서 확인). T2의 "손가락 관절 원점 16 mm 차이"도 같은 원인이다.
- **카메라 3개:** USD 안의 상대 자세는 `camera_frames.json`과 정확히 일치한다(0.0°, 0.0 mm). `optical_camera_frames.json`은 손목 카메라가 90° 다르다(광학 좌표 규약 파일).
- **경고:** Isaac Sim 6.1에서 6개 파일을 USD 진단 음소거를 풀고(`--/persistent/app/usd/muteUsdDiagnostics=false`) 열어도 deprecated 스키마 경고나 오류는 없다. 사용 스키마: `PhysicsArticulationRootAPI`, `PhysxArticulationAPI`, `PhysxJointAPI`, `PhysicsDriveAPI`, `PhysxDrivePerformanceEnvelopeAPI`, `PhysicsJointStateAPI`, `PhysxCollisionAPI`, `PhysxConvexDecompositionCollisionAPI` 등.

#### 링크별 충돌체 (`overrides/rby1_24dof.usda`, instance proxy 포함)

| 링크 | 충돌체 |
|---|---|
| base | Cylinder ×4 |
| wheel_l | Cylinder ×1 |
| wheel_r | Cylinder ×1 |
| link_torso_0 | Capsule ×1 |
| link_torso_1 | Capsule ×1 |
| link_torso_2 | Capsule ×1 |
| link_torso_3 | **없음** |
| link_torso_4 | Capsule ×1 |
| link_torso_5 | Capsule ×1 |
| link_head_0 | **없음** |
| link_head_1 | **없음** |
| link_head_2 | **없음** |
| link_left_arm_0 | Capsule ×1 |
| link_left_arm_1 | Capsule ×1 |
| link_left_arm_2 | Capsule ×1 |
| link_left_arm_3 | Capsule ×1 |
| link_left_arm_4 | Capsule ×1 |
| link_left_arm_5 | Capsule ×1 |
| link_left_arm_6 | **없음** |
| link_right_arm_0 | Capsule ×1 |
| link_right_arm_1 | Capsule ×1 |
| link_right_arm_2 | Capsule ×1 |
| link_right_arm_3 | Capsule ×1 |
| link_right_arm_4 | Capsule ×1 |
| link_right_arm_5 | Capsule ×1 |
| link_right_arm_6 | **없음** |
| ee_finger_l2 | Mesh/convexDecomposition ×1 |
| ee_finger_l1 | Mesh/convexDecomposition ×1 |
| ee_left | **없음** |
| ee_finger_r2 | Mesh/convexDecomposition ×1 |
| ee_finger_r1 | Mesh/convexDecomposition ×1 |
| ee_right | **없음** |

## 4. URDF 대조 (T2)

차이 나는 항목만 (단위: rad, rad/s, m, N·m, N):

| 관절 | 항목 | URDF | USD |
|---|---|---|---|
| right_wheel | effort↔maxForce | — | 100 |
| right_wheel | velocity↔maxJointVelocity | 15.71 | 17.45 |
| left_wheel | effort↔maxForce | — | 100 |
| left_wheel | velocity↔maxJointVelocity | 15.71 | 17.45 |
| torso_0 | effort↔maxForce | 270 | 841 |
| torso_1 | effort↔maxForce | 270 | 841 |
| torso_2 | effort↔maxForce | 270 | 841 |
| torso_3 | effort↔maxForce | 120 | 369 |
| torso_4 | effort↔maxForce | 120 | 369 |
| torso_5 | effort↔maxForce | 120 | 369 |
| right_arm_0 | effort↔maxForce | 70 | 143 |
| right_arm_1 | upper | 0.01745 | 0 |
| right_arm_1 | effort↔maxForce | 70 | 143 |
| right_arm_2 | effort↔maxForce | 70 | 143 |
| right_arm_3 | upper | 0.01745 | 0 |
| right_arm_3 | effort↔maxForce | 40 | 70 |
| right_arm_4 | effort↔maxForce | 10 | 70 |
| right_arm_5 | effort↔maxForce | 10 | 70 |
| right_arm_6 | effort↔maxForce | 8 | 140 |
| right_arm_6 | velocity↔maxJointVelocity | 2.094 | 6.283 |
| left_arm_0 | effort↔maxForce | 70 | 143 |
| left_arm_1 | lower | -0.01745 | 0 |
| left_arm_1 | effort↔maxForce | 70 | 143 |
| left_arm_2 | effort↔maxForce | 70 | 143 |
| left_arm_3 | upper | 0.01745 | 0 |
| left_arm_3 | effort↔maxForce | 40 | 70 |
| left_arm_4 | effort↔maxForce | 10 | 70 |
| left_arm_5 | effort↔maxForce | 10 | 70 |
| left_arm_6 | effort↔maxForce | 8 | 140 |
| left_arm_6 | velocity↔maxJointVelocity | 2.094 | 6.283 |
| head_0 | effort↔maxForce | 1000 | 70 |
| head_0 | velocity↔maxJointVelocity | 3.14 | 3.142 |
| head_1 | effort↔maxForce | 1000 | 70 |
| gripper_finger_l1 | effort↔maxForce | 1000 | 75 |
| gripper_finger_l1 | velocity↔maxJointVelocity | 0.05 | 0.04 |
| gripper_finger_l2 | velocity↔maxJointVelocity | 0.05 | 0.04 |
| gripper_finger_r1 | effort↔maxForce | 1000 | 75 |
| gripper_finger_r1 | velocity↔maxJointVelocity | 0.05 | 0.04 |
| gripper_finger_r2 | velocity↔maxJointVelocity | 0.05 | 0.04 |

영 자세 기구학 대조: 몸체 24관절의 원점 오차 최대 0.000 mm, 축 내적 최소 +1.0000.
기준 밖: gripper_finger_l1(원점 16.0 mm, 축 +1.000), gripper_finger_l2(원점 16.0 mm, 축 +1.000), gripper_finger_r1(원점 16.0 mm, 축 +1.000), gripper_finger_r2(원점 16.0 mm, 축 +1.000)

- **기구학:** 몸체 24관절의 원점과 축이 영 자세에서 URDF와 일치한다. `right/left_arm_0`은 USD `physics:axis`가 빈 토큰이지만, 기본값 X로 읽으면 URDF의 기울어진 축 (0, 0.94, −0.34)과 정확히 일치한다. 스윕에서도 정상으로 움직였다.
- **effort:** USD drive `maxForce`가 URDF effort보다 크다. 몸통 841/369 vs 270/120, 팔 143/143/143/70/70/70/140 vs 70/70/70/40/10/10/8, 머리 70 vs 1000(URDF 값은 자리표시로 보임). 원본은 USD maxForce로 클리핑했고, 이번 설정 초안은 URDF effort를 쓴다(머리만 USD 70).
- **한계:** arm_1·arm_3에서 URDF가 0 너머로 1° 더 허용한다. 속도는 arm_6이 URDF 2.09 vs USD 6.28 rad/s, 바퀴가 15.7 vs 17.5 rad/s다.
- **게인 표의 불일치:** `src/config.py`의 `JOINT_KD_BASE`×10은 `rby1_task.py`의 kd와 arm_2(321.5 vs 521.5)·arm_6(61.26 vs 91.26)이 다르다. 실제로 쓰이는 값은 `rby1_task.py`다.

## 5. Isaac Lab 로드와 동작 (T3)

### T3a 정지 유지 (10 s, 영 자세)

기준: NaN 없음 · 관절 오차 ≤ 0.02 rad · 베이스 이동 ≤ 1 cm. *진동 관절 수 = 1 s 이후 오차 부호가 6번 이상 바뀐 관절 수.

| 조건 | 판정 | 최대 관절오차 rad (관절) | 마지막 3 s p2p rad | 진동 관절 수* | 베이스 이동 mm | wall s | 파일 |
|---|---|---|---|---|---|---|---|
| implicit · GPU · 1/500 · 마찰 full (기본) | ✅ 통과 | 0.01683 (left_arm_0) | 0.004512 (right_arm_0) | 0 | 0.000754 | 17.5 | `t3/idle_implicit_gpu_dt500_n1.json` |
| implicit · GPU · 1/200 · 마찰 full | ❌ 실패 | 0.02833 (right_arm_3) | 0.007168 (left_arm_3) | 0 | 0.00241 | 7.94 | `t3/idle_implicit_gpu_dt200_n1.json` |
| ideal_pd(원본식) · GPU · 1/500 · full · effort=USD | ✅ 통과 | 0.01774 (left_arm_0) | 0.003747 (right_arm_3) | 0 | 0.0044 | 111 | `t3/idle_ideal_pd_gpu_dt500_n1.json` |
| ideal_pd(원본식) · GPU · 1/200 · full · effort=USD | ❌ 실패 | 0.02883 (right_arm_3) | 0.007823 (left_arm_3) | 0 | 2.3 | 8.07 | `t3/idle_ideal_pd_gpu_dt200_n1.json` |
| implicit · GPU · 1/500 · ready 자세 | ✅ 통과 | 0.01683 (left_arm_0) | 0.00451 (right_arm_0) | 0 | 0.000505 | 13.8 | `t3/idle_implicit_gpu_dt500_n1_ready.json` |
| 조정안 (점성 + ζ0.7 + 중력보상) · GPU · 1/500 | ✅ 통과 | 0.002014 (torso_1) | 2.107e-06 (torso_3) | 0 | 0.000813 | 14.9 | `t3/idle_implicit_gpu_dt500_n1_tuned_gc.json` |
| 조정안 (점성 + ζ0.7 + 중력보상) · GPU · 1/200 | ✅ 통과 | 0.005155 (torso_1) | 1.175e-06 (left_arm_0) | 0 | 0.00309 | 8.27 | `t3/idle_implicit_gpu_dt200_n1_tuned_gc.json` |

- **기본 조건(GPU, 1/500, 원본 재현 게인, 원본 마찰 모델):** 최대 관절 오차 0.0168 rad(left_arm_0)로 통과하고, 베이스 이동은 1 µm 미만이다.
- **1/200 s:** 오차가 10초 동안 한 방향으로 계속 커지는 **크리프**(0.028 rad, arm_3)로 실패한다. 진동이 아니라서 ζ를 0.7·1.0으로 올려도 그대로다(아래 표).
- **CPU:** 같은 조건에서 0.005 rad로, GPU와 Coulomb 마찰 처리가 다르다.
- **`--pose ready`:** `rby1_ready_reach_table.usd`에 0이 아닌 관절 상태가 저장돼 있지 않아 영 자세와 같은 결과가 나왔다.

#### 감쇠·마찰 비교 — (a) 원본식 명시적 PD, (b) 감쇠비 조정

| 조건 | 판정 | 최대 관절오차 rad (관절) | 마지막 3 s p2p rad | 진동 관절 수* | 베이스 이동 mm | wall s | 파일 |
|---|---|---|---|---|---|---|---|
| implicit · 1/500 · 마찰 none · 원본 감쇠 | ❌ 실패 | 0.02669 (torso_3) | 0.009918 (torso_3) | 10 | 0.000751 | 14.1 | `t3/idle_implicit_gpu_dt500_n1_nofric.json` |
| ideal_pd · 1/500 · 마찰 none · 원본 감쇠 | ❌ 실패 | 0.05903 (torso_2) | 0.1114 (torso_2) | 19 | 1.97 | 14.3 | `t3/idle_ideal_pd_gpu_dt500_n1_nofric.json` |
| implicit · 1/500 · 마찰 none · ζ=0.7 | ✅ 통과 | 0.01632 (torso_3) | 0.0001906 (left_arm_5) | 0 | 0.000835 | 14.1 | `t3/idle_implicit_gpu_dt500_n1_nofric_z0.7.json` |
| implicit · 1/500 · 마찰 none · ζ=1.0 | ✅ 통과 | 0.01544 (torso_3) | 0.0001473 (left_arm_5) | 0 | 0.000682 | 13.7 | `t3/idle_implicit_gpu_dt500_n1_nofric_z1.0.json` |
| implicit · 1/500 · 마찰 viscous · 원본 감쇠 | ❌ 실패 | 0.02374 (torso_3) | 0.001508 (torso_3) | 5 | 0.000763 | 14 | `t3/idle_implicit_gpu_dt500_n1_visc.json` |
| ideal_pd · 1/500 · 마찰 viscous · 원본 감쇠 | ❌ 실패 | 0.02465 (torso_3) | 0.004042 (torso_3) | 6 | 0.00292 | 14.3 | `t3/idle_ideal_pd_gpu_dt500_n1_visc.json` |
| implicit · 1/500 · 마찰 viscous · ζ=0.7 | ✅ 통과 | 0.01604 (torso_3) | 0.0001753 (left_arm_5) | 0 | 0.000813 | 13.9 | `t3/idle_implicit_gpu_dt500_n1_visc_z0.7.json` |
| implicit · 1/500 · 마찰 viscous · ζ=1.0 | ✅ 통과 | 0.01542 (torso_3) | 0.0001336 (left_arm_5) | 0 | 0.000762 | 13.9 | `t3/idle_implicit_gpu_dt500_n1_visc_z1.0.json` |
| implicit · 1/500 · 마찰 full · 원본 감쇠 | ✅ 통과 | 0.01683 (left_arm_0) | 0.004512 (right_arm_0) | 0 | 0.000754 | 17.5 | `t3/idle_implicit_gpu_dt500_n1.json` |
| ideal_pd · 1/500 · 마찰 full · 원본 감쇠 | ✅ 통과 | 0.01774 (left_arm_0) | 0.003747 (right_arm_3) | 0 | 0.0044 | 111 | `t3/idle_ideal_pd_gpu_dt500_n1.json` |
| implicit · 1/500 · 마찰 full · ζ=0.7 | ✅ 통과 | 0.01741 (left_arm_0) | 0.004594 (right_arm_0) | 0 | 0.000677 | 13.9 | `t3/idle_implicit_gpu_dt500_n1_full_z0.7.json` |
| implicit · 1/500 · 마찰 full · ζ=1.0 | ✅ 통과 | 0.01765 (left_arm_0) | 0.004629 (right_arm_0) | 0 | 0.000697 | 13.8 | `t3/idle_implicit_gpu_dt500_n1_full_z1.0.json` |
| implicit · 1/200 · 마찰 none · 원본 감쇠 | ❌ 실패 | 0.02638 (head_1) | 0.008166 (torso_3) | 10 | 0.00372 | 8.13 | `t3/idle_implicit_gpu_dt200_n1_nofric.json` |
| ideal_pd · 1/200 · 마찰 none · 원본 감쇠 | ❌ 실패 | 0.05756 (torso_2) | 0.1097 (torso_2) | 18 | 4.47 | 8.29 | `t3/idle_ideal_pd_gpu_dt200_n1_nofric.json` |
| implicit · 1/200 · 마찰 none · ζ=0.7 | ✅ 통과 | 0.01507 (torso_3) | 0.0001248 (left_arm_5) | 0 | 0.00303 | 7.89 | `t3/idle_implicit_gpu_dt200_n1_nofric_z0.7.json` |
| implicit · 1/200 · 마찰 none · ζ=1.0 | ✅ 통과 | 0.01443 (torso_3) | 7.35e-05 (left_arm_5) | 0 | 0.003 | 7.86 | `t3/idle_implicit_gpu_dt200_n1_nofric_z1.0.json` |
| implicit · 1/200 · 마찰 viscous · 원본 감쇠 | ❌ 실패 | 0.02331 (torso_3) | 0.001228 (torso_3) | 4 | 0.00361 | 7.92 | `t3/idle_implicit_gpu_dt200_n1_visc.json` |
| ideal_pd · 1/200 · 마찰 viscous · 원본 감쇠 | ❌ 실패 | 0.02428 (torso_3) | 0.002413 (torso_3) | 6 | 2.31 | 8.13 | `t3/idle_ideal_pd_gpu_dt200_n1_visc.json` |
| implicit · 1/200 · 마찰 viscous · ζ=0.7 | ✅ 통과 | 0.01494 (torso_3) | 0.0001131 (left_arm_5) | 0 | 0.00305 | 7.89 | `t3/idle_implicit_gpu_dt200_n1_visc_z0.7.json` |
| implicit · 1/200 · 마찰 viscous · ζ=1.0 | ✅ 통과 | 0.01438 (torso_3) | 6.437e-05 (left_arm_5) | 0 | 0.00296 | 7.81 | `t3/idle_implicit_gpu_dt200_n1_visc_z1.0.json` |
| implicit · 1/200 · 마찰 full · 원본 감쇠 | ❌ 실패 | 0.02833 (right_arm_3) | 0.007168 (left_arm_3) | 0 | 0.00241 | 7.94 | `t3/idle_implicit_gpu_dt200_n1.json` |
| ideal_pd · 1/200 · 마찰 full · 원본 감쇠 | ❌ 실패 | 0.02883 (right_arm_3) | 0.007823 (left_arm_3) | 0 | 2.3 | 8.07 | `t3/idle_ideal_pd_gpu_dt200_n1.json` |
| implicit · 1/200 · 마찰 full · ζ=0.7 | ❌ 실패 | 0.02911 (right_arm_3) | 0.007533 (left_arm_3) | 0 | 0.00244 | 7.87 | `t3/idle_implicit_gpu_dt200_n1_full_z0.7.json` |
| implicit · 1/200 · 마찰 full · ζ=1.0 | ❌ 실패 | 0.02943 (right_arm_3) | 0.007686 (right_arm_3) | 0 | 0.00247 | 7.92 | `t3/idle_implicit_gpu_dt200_n1_full_z1.0.json` |
| implicit · CPU · 1/500 · 마찰 none | ❌ 실패 | 0.02676 (torso_3) | 0.01011 (torso_3) | 11 | 0.000612 | 6.13 | `t3/idle_implicit_cpu_dt500_n1_nofric.json` |
| implicit · CPU · 1/500 · 마찰 viscous | ❌ 실패 | 0.02385 (torso_3) | 0.001521 (torso_3) | 5 | 0.000349 | 6.15 | `t3/idle_implicit_cpu_dt500_n1_visc.json` |
| implicit · CPU · 1/500 · 마찰 full | ✅ 통과 | 0.004985 (torso_1) | 0.001261 (torso_2) | 0 | 0.000245 | 14.7 | `t3/idle_implicit_cpu_dt500_n1.json` |

요청대로 바로 고치지 않고 (a)와 (b)를 비교했다.

**(a) 원본 방식(IdealPD 명시적 PD, 1/500, USD maxForce 클리핑)과 비교**
- 원본 마찰 모델이 있으면 implicit와 거의 같다(0.0177 vs 0.0168 rad).
- **마찰을 빼면 명시적 PD가 훨씬 불안정하다.** torso_2 진동 p2p가 0.11 rad이고 베이스가 2~4 mm 흔들린다. 같은 조건의 implicit은 0.01 rad다.
- 원본의 500 Hz 토크 루프는 kd × dt 감쇠가 거의 없어서, 관절 점성 마찰(몸통 20/10, 어깨 4.7 N·m·s/rad)에 기대어 안정했던 것이다.
- 1/200에서 명시적 PD는 같은 경향에 더 나쁘다.

**(b) 감쇠비 기준 조정안**
- 감쇠비 표(아래)를 보면, 원본 감쇠는 드라이브만으로 ζ 0.009~0.04, 점성 마찰을 포함해도 0.03~0.37이다. torso_3·4는 고유진동수 1.5~1.6 Hz에 ζ ≈ 0.1이다.
- D = 2ζ√(K·(M_ii + armature))로 ζ = 0.7을 맞추면, 마찰 none이나 viscous에서 1/200과 1/500 모두 진동이 사라진다(진동 관절 0개, p2p < 0.2 mrad).
- 남는 오차 0.015 rad는 torso_3의 정적 처짐(중력/K)이다.
- 중력보상 feedforward를 더하면 1/500에서 0.0020 rad, 1/200에서 0.0052 rad(`preset="tuned"`)가 된다.
- **Coulomb(정지/동) 마찰은 GPU에서 1/200 크리프를 만들기 때문에, 조정안은 점성 마찰만 남긴다.**
- 감쇠값은 영 자세의 질량행렬 기준이라 자세에 따라 바뀐다. 2단계에서는 영 자세 값을 상수로 쓰고, 숙임 자세에서 다시 확인할 것을 권한다.

#### 감쇠비 표 (기본 조건, 영 자세)

| 관절 | K N·m/rad | D (=kd×0.002) | 점성마찰 | M_ii + armature kg·m² | f_n Hz | ζ 드라이브만 | ζ 점성 포함 | D(ζ=0.7) | D(ζ=1.0) |
|---|---|---|---|---|---|---|---|---|---|
| torso_0 | 3911 | 7.04 | 20 | 43.72 | 1.51 | 0.00851 | 0.0327 | 579 | 827 |
| torso_1 | 3911 | 7.04 | 20 | 42.67 | 1.52 | 0.00862 | 0.0331 | 572 | 817 |
| torso_2 | 3911 | 7.04 | 20 | 17.38 | 2.39 | 0.0135 | 0.0519 | 365 | 521 |
| torso_3 | 573.6 | 2.086 | 10 | 5.44 | 1.63 | 0.0187 | 0.108 | 78.2 | 112 |
| torso_4 | 573.6 | 2.086 | 10 | 6.44 | 1.5 | 0.0172 | 0.0994 | 85.1 | 122 |
| torso_5 | 573.6 | 2.086 | 10 | 3.025 | 2.19 | 0.025 | 0.145 | 58.3 | 83.3 |
| right_arm_0 | 208.6 | 1.043 | 4.7 | 1.508 | 1.87 | 0.0294 | 0.162 | 24.8 | 35.5 |
| right_arm_1 | 208.6 | 1.043 | 4.7 | 1.651 | 1.79 | 0.0281 | 0.155 | 26 | 37.1 |
| right_arm_2 | 208.6 | 0.643 | 4.7 | 0.2998 | 4.2 | 0.0407 | 0.338 | 11.1 | 15.8 |
| right_arm_3 | 91.27 | 0.4172 | 1.9 | 0.7201 | 1.79 | 0.0257 | 0.143 | 11.3 | 16.2 |
| right_arm_4 | 39.11 | 0.1825 | 1.9 | 0.2035 | 2.21 | 0.0323 | 0.369 | 3.95 | 5.64 |
| right_arm_5 | 39.11 | 0.1825 | 1.9 | 0.2264 | 2.09 | 0.0307 | 0.35 | 4.17 | 5.95 |
| right_arm_6 | 39.11 | 0.1225 | 1 | 0.201 | 2.22 | 0.0218 | 0.2 | 3.93 | 5.61 |
| left_arm_0 | 208.6 | 1.043 | 4.7 | 1.501 | 1.88 | 0.0295 | 0.162 | 24.8 | 35.4 |
| left_arm_1 | 208.6 | 1.043 | 4.7 | 1.651 | 1.79 | 0.0281 | 0.155 | 26 | 37.1 |
| left_arm_2 | 208.6 | 0.643 | 4.7 | 0.2998 | 4.2 | 0.0407 | 0.338 | 11.1 | 15.8 |
| left_arm_3 | 91.27 | 0.4172 | 1.9 | 0.7201 | 1.79 | 0.0257 | 0.143 | 11.3 | 16.2 |
| left_arm_4 | 39.11 | 0.1825 | 1.9 | 0.2035 | 2.21 | 0.0323 | 0.369 | 3.95 | 5.64 |
| left_arm_5 | 39.11 | 0.1825 | 1.9 | 0.2264 | 2.09 | 0.0307 | 0.35 | 4.17 | 5.95 |
| left_arm_6 | 39.11 | 0.1225 | 1 | 0.201 | 2.22 | 0.0218 | 0.2 | 3.93 | 5.61 |
| head_0 | 39.11 | 0.1825 | 3 | 0.5049 | 1.4 | 0.0205 | 0.358 | 6.22 | 8.89 |
| head_1 | 39.11 | 0.1825 | 3 | 0.5071 | 1.4 | 0.0205 | 0.357 | 6.23 | 8.91 |

### T3b 숙임 자세 유지

| 조건 | 판정 | 유지 중 최대 오차 rad | 최악 토크 관절 | 투영 토크/effort | 중력보상/effort | 기울기° | ee z (L/R) m | 파일 |
|---|---|---|---|---|---|---|---|---|
| (60,−90,60) implicit · 1/500 · 원본 게인 | ❌ 실패 | 0.08832 | torso_1 | 0.549 | 0.544 | 0 | 0.494 / 0.497 | `t3/crouch_implicit_gpu_dt500_n1.json` |
| (60,−90,60) implicit · 1/200 · 원본 게인 | ❌ 실패 | 0.1044 | torso_1 | 0.56 | 0.553 | 0 | 0.489 / 0.494 | `t3/crouch_implicit_gpu_dt200_n1.json` |
| (60,−90,60) ideal_pd · 1/500 · effort USD | ❌ 실패 | 0.09501 | torso_1 | 0.495 | 0.544 | 0 | 0.496 / 0.499 | `t3/crouch_ideal_pd_gpu_dt500_n1.json` |
| (60,−90,60) 점성만 + ζ0.7 | ❌ 실패 | 0.08322 | torso_1 | 0.552 | 0.55 | 0 | 0.497 / 0.496 | `t3/crouch_implicit_gpu_dt500_n1_tuned.json` |
| (60,−90,60) 조정안 (점성 + ζ0.7 + 중력보상) | ✅ 통과 | 0.003635 | torso_1 | 0.452 | 0.474 | 0 | 0.526 / 0.526 | `t3/crouch_implicit_gpu_dt500_n1_tuned_gc.json` |
| (75,−105,90) 원본 게인 | ❌ 실패 | 0.1231 | torso_1 | 0.728 | 0.727 | 0 | 0.447 / 0.446 | `t3/crouch_implicit_gpu_dt500_n1_deep.json` |
| (75,−105,90) 조정안 | ✅ 통과 | 0.004079 | torso_3 | 0.647 | 0.567 | 0 | 0.539 / 0.539 | `t3/crouch_implicit_gpu_dt500_n1_deep_tuned_gc.json` |
| (90,−130,90) 원본 게인 | ❌ 실패 | 0.1044 | torso_1 | 0.584 | 0.573 | 0 | 0.3 / 0.302 | `t3/crouch_implicit_gpu_dt500_n1_deeper.json` |
| (90,−130,90) 조정안 | ✅ 통과 | 0.0009773 | torso_3 | 0.575 | 0.488 | 0 | 0.375 / 0.375 | `t3/crouch_implicit_gpu_dt500_n1_deeper_tuned_gc.json` |

기본 조건(implicit, 1/500) 관절별 유지 토크:

| 관절 | 최대 오차 rad | applied 최대 | 투영 최대 | 중력보상 | URDF effort | 투영/effort |
|---|---|---|---|---|---|---|
| torso_0 | 0.0009818 | 3.86 | 4.07 | 3.3 | 270 | 0.0151 |
| torso_1 | 0.04054 | 159 | 148 | 147 | 270 | 0.549 |
| torso_2 | 0.01182 | 46.2 | 36 | 34 | 270 | 0.133 |
| torso_3 | 0.08832 | 50.6 | 47.9 | 46.5 | 120 | 0.399 |
| torso_4 | 0.003477 | 1.99 | 0.665 | 0.375 | 120 | 0.00554 |
| torso_5 | 0.003371 | 1.93 | 0.123 | 0.0791 | 120 | 0.00103 |
| right_arm_0 | 0.08046 | 16.8 | 13.9 | 13.7 | 70 | 0.199 |
| right_arm_1 | 1.762e-07 | 0.00054 | 0.16 | 0.043 | 70 | 0.00229 |
| right_arm_3 | 0.05324 | 4.86 | 6.08 | 5.99 | 40 | 0.152 |
| head_1 | 0.00202 | 0.079 | 0.73 | 0.727 | 70 | 0.0104 |

- **원인은 토크 부족이 아니라 중력보상이 없어서 생기는 처짐이다.**
  - (60°, −90°, 60°) 자세에서 투영 토크/URDF effort는 최대 0.55(torso_1 148 N·m)로 80% 기준 안이다.
  - 관절 오차는 0.088 rad(torso_3: 46.5 N·m ÷ K 573.6 ≈ 0.081 rad)로 기준 0.05를 넘는다.
- **조정안은 세 자세 모두 통과한다**(최대 오차 0.001~0.004 rad).
- **토크 여유가 가장 작은 자세:** (75°, −105°, 90°)에서 torso_1이 effort의 73%다. PDF의 "앞 0.5 m에서 정격 71%"와 같은 수준이다.
- **손끝 높이:** (60, −90, 60)에서 0.49 m → (90, −130, 90)에서 0.30 m(원본 게인, 처짐 포함) / 0.375 m(조정안). 바닥 집기 자세는 몸통만으로는 닿지 않으니, 2단계에서 팔·손목 IK와 함께 다시 잡아야 한다.
- 기울기는 모두 0°라서 넘어질 위험이 없다.

### T3c 관절 스윕

**sweep_implicit_gpu_dt500_n1** — ❌ 실패 (주기 10.0 s, 한계의 ±80%)

| 관절 | 명령 범위 rad | RMS 오차 | 최대 오차 | 오버슈트 상/하 | 도달 상/하 | 최대 토크 | 기울기° | 링크 최저 z m | 플래그 |
|---|---|---|---|---|---|---|---|---|---|
| torso_0 | [-0.209, 0.209] | 0.0213 | 0.035 | 0.029 / 0.028 | 0.239 / -0.237 | 137 | 0 | 0.28 |  |
| torso_1 | [-0.419, 1.257] | 0.0984 | 0.445 | 0.065 / 0.023 | 1.32 / -0.442 | 270 | 120 | 0.0396 | tip>10deg, max_err>0.1rad |
| torso_2 | [-2.094, 1.257] | 0.148 | 0.574 | 0.065 / 0.051 | 1.32 / -2.15 | 270 | 0 | 0.103 | max_err>0.1rad |
| torso_3 | [-0.628, 1.257] | 0.108 | 0.183 | 0.13 / 0.098 | 1.39 / -0.727 | 106 | 0 | 0.28 | max_err>0.1rad |
| torso_4 | [-0.419, 0.419] | 0.0544 | 0.0987 | 0.053 / 0.055 | 0.472 / -0.474 | 56.9 | 0 | 0.28 |  |
| torso_5 | [-1.885, 1.885] | 0.0331 | 0.103 | 0 / 0 | 1.87 / -1.87 | 61.8 | 0 | 0.28 | max_err>0.1rad |
| right_arm_0 | [-2.513, 2.513] | 0.105 | 0.173 | 0 / 0 | 2.4 / -2.41 | 37.1 | 0 | 0.28 | max_err>0.1rad |
| right_arm_1 | [-2.513, 0.000] | 0.083 | 0.189 | 5.3e-07 / 0 | 5.33e-07 / -2.4 | 40.4 | 0 | 0.28 | max_err>0.1rad |
| right_arm_2 | [-2.513, 2.513] | 0.0487 | 0.0905 | 0 / 0 | 2.49 / -2.49 | 20.2 | 0 | 0.28 |  |
| right_arm_3 | [-2.094, 0.000] | 0.0709 | 0.158 | 0 / 0 | -0.000613 / -1.97 | 14.8 | 0 | 0.28 | max_err>0.1rad |
| right_arm_4 | [-2.513, 2.513] | 0.107 | 0.185 | 0 / 0 | 2.46 / -2.46 | 7.48 | 0 | 0.28 | max_err>0.1rad |
| right_arm_5 | [-1.257, 1.536] | 0.0914 | 0.136 | 0 / 0 | 1.44 / -1.16 | 5.48 | 0 | 0.28 | max_err>0.1rad |
| right_arm_6 | [-2.164, 2.164] | 0.0548 | 0.133 | 0 / 0 | 2.14 / -2.14 | 5.41 | 0 | 0.28 | max_err>0.1rad |
| left_arm_0 | [-2.513, 2.513] | 0.105 | 0.171 | 0 / 0 | 2.4 / -2.42 | 36.6 | 0 | 0.28 | max_err>0.1rad |
| left_arm_1 | [0.000, 2.513] | 0.0829 | 0.188 | 0 / 0 | 2.4 / 5.56e-06 | 40.2 | 0 | 0.28 | max_err>0.1rad |
| left_arm_2 | [-2.513, 2.513] | 0.0487 | 0.0905 | 0 / 0 | 2.49 / -2.49 | 20.2 | 0 | 0.28 |  |
| left_arm_3 | [-2.094, 0.000] | 0.0704 | 0.154 | 0 / 0 | -0.000576 / -1.97 | 14.4 | 0 | 0.28 | max_err>0.1rad |
| left_arm_4 | [-2.513, 2.513] | 0.107 | 0.185 | 0 / 0 | 2.46 / -2.46 | 7.48 | 0 | 0.28 | max_err>0.1rad |
| left_arm_5 | [-1.257, 1.536] | 0.0913 | 0.136 | 0 / 0 | 1.44 / -1.16 | 5.48 | 0 | 0.28 | max_err>0.1rad |
| left_arm_6 | [-2.164, 2.164] | 0.0548 | 0.133 | 0 / 0 | 2.14 / -2.14 | 5.41 | 0 | 0.28 | max_err>0.1rad |
| head_0 | [-1.257, 1.257] | 0.0904 | 0.153 | 0 / 0 | 1.21 / -1.21 | 6.12 | 0 | 0.28 | max_err>0.1rad |
| head_1 | [-1.257, 1.257] | 0.0918 | 0.141 | 0 / 0 | 1.23 / -1.22 | 5.63 | 0 | 0.28 | max_err>0.1rad |

**sweep_implicit_gpu_dt200_n1** — ❌ 실패 (주기 10.0 s, 한계의 ±80%)

| 관절 | 명령 범위 rad | RMS 오차 | 최대 오차 | 오버슈트 상/하 | 도달 상/하 | 최대 토크 | 기울기° | 링크 최저 z m | 플래그 |
|---|---|---|---|---|---|---|---|---|---|
| torso_0 | [-0.209, 0.209] | 0.0216 | 0.0347 | 0.031 / 0.031 | 0.241 / -0.24 | 137 | 0 | 0.28 |  |
| torso_1 | [-0.419, 1.257] | 0.0989 | 0.444 | 0.073 / 0.024 | 1.33 / -0.443 | 270 | 120 | 0.0405 | tip>10deg, max_err>0.1rad |
| torso_2 | [-2.094, 1.257] | 0.135 | 0.504 | 0.066 / 0.049 | 1.32 / -2.14 | 270 | 0 | 0.101 | max_err>0.1rad |
| torso_3 | [-0.628, 1.257] | 0.108 | 0.183 | 0.13 / 0.099 | 1.39 / -0.728 | 107 | 0 | 0.28 | max_err>0.1rad |
| torso_4 | [-0.419, 0.419] | 0.0549 | 0.099 | 0.055 / 0.057 | 0.474 / -0.476 | 57.5 | 0 | 0.28 |  |
| torso_5 | [-1.885, 1.885] | 0.0331 | 0.103 | 0 / 0 | 1.87 / -1.87 | 65.7 | 0 | 0.28 | max_err>0.1rad |
| right_arm_0 | [-2.513, 2.513] | 0.105 | 0.173 | 0 / 0 | 2.4 / -2.41 | 38.6 | 0 | 0.28 | max_err>0.1rad |
| right_arm_1 | [-2.513, 0.000] | 0.0831 | 0.19 | 3.4e-06 / 0 | 3.42e-06 / -2.4 | 42.1 | 0 | 0.28 | max_err>0.1rad |
| right_arm_2 | [-2.513, 2.513] | 0.0488 | 0.0906 | 0 / 0 | 2.49 / -2.49 | 22.1 | 0 | 0.28 |  |
| right_arm_3 | [-2.094, 0.000] | 0.0701 | 0.15 | 0 / 0 | -0.00155 / -1.97 | 14.5 | 0 | 0.28 | max_err>0.1rad |
| right_arm_4 | [-2.513, 2.513] | 0.107 | 0.185 | 0 / 0 | 2.46 / -2.46 | 7.85 | 0 | 0.28 | max_err>0.1rad |
| right_arm_5 | [-1.257, 1.536] | 0.0921 | 0.136 | 0 / 0 | 1.44 / -1.16 | 5.69 | 0 | 0.28 | max_err>0.1rad |
| right_arm_6 | [-2.164, 2.164] | 0.0549 | 0.133 | 0 / 0 | 2.14 / -2.14 | 5.7 | 0 | 0.28 | max_err>0.1rad |
| left_arm_0 | [-2.513, 2.513] | 0.105 | 0.168 | 0 / 0 | 2.4 / -2.41 | 37.5 | 0 | 0.28 | max_err>0.1rad |
| left_arm_1 | [0.000, 2.513] | 0.0831 | 0.189 | 0 / 0 | 2.4 / 1.37e-05 | 41.7 | 0 | 0.28 | max_err>0.1rad |
| left_arm_2 | [-2.513, 2.513] | 0.0488 | 0.0906 | 0 / 0 | 2.49 / -2.49 | 22.1 | 0 | 0.28 |  |
| left_arm_3 | [-2.094, 0.000] | 0.0701 | 0.15 | 0 / 0 | -0.00145 / -1.97 | 14.5 | 0 | 0.28 | max_err>0.1rad |
| left_arm_4 | [-2.513, 2.513] | 0.107 | 0.185 | 0 / 0 | 2.46 / -2.46 | 7.85 | 0 | 0.28 | max_err>0.1rad |
| left_arm_5 | [-1.257, 1.536] | 0.0916 | 0.136 | 0 / 0 | 1.44 / -1.16 | 5.7 | 0 | 0.28 | max_err>0.1rad |
| left_arm_6 | [-2.164, 2.164] | 0.0549 | 0.133 | 0 / 0 | 2.14 / -2.14 | 5.7 | 0 | 0.28 | max_err>0.1rad |
| head_0 | [-1.257, 1.257] | 0.0905 | 0.153 | 0 / 0 | 1.21 / -1.21 | 6.3 | 0 | 0.28 | max_err>0.1rad |
| head_1 | [-1.257, 1.257] | 0.0919 | 0.141 | 0 / 0 | 1.23 / -1.22 | 5.81 | 0 | 0.28 | max_err>0.1rad |

**sweep_implicit_gpu_dt500_n1_fixroot** — ✅ 통과 (주기 10.0 s, 한계의 ±80%)

| 관절 | 명령 범위 rad | RMS 오차 | 최대 오차 | 오버슈트 상/하 | 도달 상/하 | 최대 토크 | 기울기° | 링크 최저 z m | 플래그 |
|---|---|---|---|---|---|---|---|---|---|
| torso_0 | [-0.209, 0.209] | 0.0212 | 0.0351 | 0.029 / 0.026 | 0.238 / -0.235 | 138 | 0 | 0.28 |  |
| torso_1 | [-0.419, 1.257] | 1.39 | 1.99 | 0.31 / 0 | 1.57 / -0.000165 | 270 | 0 | 0.0485 | max_err>0.1rad |
| torso_2 | [-2.094, 1.257] | 0.148 | 0.574 | 0.064 / 0.049 | 1.32 / -2.14 | 270 | 0 | 0.105 | max_err>0.1rad |
| torso_3 | [-0.628, 1.257] | 0.108 | 0.183 | 0.13 / 0.098 | 1.39 / -0.727 | 106 | 0 | 0.28 | max_err>0.1rad |
| torso_4 | [-0.419, 0.419] | 0.0538 | 0.098 | 0.052 / 0.053 | 0.47 / -0.472 | 56.5 | 0 | 0.28 |  |
| torso_5 | [-1.885, 1.885] | 0.0331 | 0.103 | 0 / 0 | 1.87 / -1.87 | 61.8 | 0 | 0.28 | max_err>0.1rad |

**sweep_implicit_gpu_dt500_n1_tuned_gc** — ❌ 실패 (주기 10.0 s, 한계의 ±80%)

| 관절 | 명령 범위 rad | RMS 오차 | 최대 오차 | 오버슈트 상/하 | 도달 상/하 | 최대 토크 | 기울기° | 링크 최저 z m | 플래그 |
|---|---|---|---|---|---|---|---|---|---|
| torso_0 | [-0.209, 0.209] | 0.00215 | 0.00682 | 0.0016 / 0.003 | 0.211 / -0.212 | 108 | 0 | 0.28 |  |
| torso_1 | [-0.419, 1.257] | 0.606 | 1.63 | 0 / 0.1 | 1.14 / -0.524 | 270 | 128 | 0.0713 | tip>10deg, max_err>0.1rad |
| torso_2 | [-2.094, 1.257] | 0.00662 | 0.0173 | 0.0031 / 0.0033 | 1.26 / -2.1 | 270 | 0 | 0.212 |  |
| torso_3 | [-0.628, 1.257] | 0.00963 | 0.0332 | 0.004 / 0.0022 | 1.26 / -0.631 | 80.6 | 0 | 0.28 |  |
| torso_4 | [-0.419, 0.419] | 0.00411 | 0.0126 | 0.0017 / 0.0019 | 0.421 / -0.421 | 35.4 | 0 | 0.28 |  |
| torso_5 | [-1.885, 1.885] | 0.0157 | 0.0406 | 0.0024 / 0.0024 | 1.89 / -1.89 | 69.1 | 0 | 0.28 |  |
| right_arm_0 | [-2.513, 2.513] | 0.0271 | 0.0647 | 0.0037 / 0.004 | 2.52 / -2.52 | 39 | 0 | 0.28 |  |
| right_arm_1 | [-2.513, 0.000] | 0.0202 | 0.071 | 5.3e-07 / 0.0043 | 5.31e-07 / -2.52 | 39.2 | 0 | 0.28 |  |
| right_arm_2 | [-2.513, 2.513] | 0.0256 | 0.0374 | 0 / 0 | 2.51 / -2.51 | 17.5 | 0 | 0.28 |  |
| right_arm_3 | [-2.094, 0.000] | 0.0178 | 0.0558 | 0 / 0.0037 | -0.000139 / -2.1 | 12.8 | 0 | 0.28 |  |
| right_arm_4 | [-2.513, 2.513] | 0.0544 | 0.0774 | 0 / 0 | 2.51 / -2.51 | 6.24 | 0 | 0.28 |  |
| right_arm_5 | [-1.257, 1.536] | 0.0313 | 0.0481 | 0 / 0 | 1.54 / -1.26 | 3.92 | 0 | 0.28 |  |
| right_arm_6 | [-2.164, 2.164] | 0.0254 | 0.0511 | 0.0018 / 0.0018 | 2.17 / -2.17 | 5.34 | 0 | 0.28 |  |
| left_arm_0 | [-2.513, 2.513] | 0.0271 | 0.0647 | 0.0037 / 0.0039 | 2.52 / -2.52 | 38.9 | 0 | 0.28 |  |
| left_arm_1 | [0.000, 2.513] | 0.0201 | 0.0692 | 0.0043 / 1e-06 | 2.52 / -1e-06 | 41.1 | 0 | 0.28 |  |
| left_arm_2 | [-2.513, 2.513] | 0.0256 | 0.0374 | 0 / 0 | 2.51 / -2.51 | 17.5 | 0 | 0.28 |  |
| left_arm_3 | [-2.094, 0.000] | 0.0175 | 0.0558 | 0 / 0.0037 | -0.000131 / -2.1 | 12.5 | 0 | 0.28 |  |
| left_arm_4 | [-2.513, 2.513] | 0.0544 | 0.0774 | 0 / 0 | 2.51 / -2.51 | 6.24 | 0 | 0.28 |  |
| left_arm_5 | [-1.257, 1.536] | 0.0309 | 0.0481 | 0 / 0 | 1.54 / -1.26 | 3.92 | 0 | 0.28 |  |
| left_arm_6 | [-2.164, 2.164] | 0.0254 | 0.0511 | 0.0018 / 0.0018 | 2.17 / -2.17 | 5.34 | 0 | 0.28 |  |
| head_0 | [-1.257, 1.257] | 0.0426 | 0.0609 | 0 / 0 | 1.26 / -1.26 | 4.91 | 0 | 0.28 |  |
| head_1 | [-1.257, 1.257] | 0.0426 | 0.061 | 0 / 0 | 1.26 / -1.26 | 4.42 | 0 | 0.28 |  |

- 조건: 각 관절을 0 → 0.8·상한 → 0.8·하한 → 0으로 10초 사인파로 움직이고, 속도 feedforward를 준다. 관절마다 상태를 리셋한다.
- **원본 게인**
  - 팔·머리의 최대 오차 0.13~0.19 rad는 대부분 중력 처짐이다. 팔을 수평으로 들면 어깨 중력 토크가 약 20 N·m인데 K가 208.6이다.
  - 몸통은 오버슈트가 0.03~0.13 rad로 ζ가 낮은 결과가 그대로 보인다.
- **조정안:** torso_1을 빼고 모든 관절이 최대 오차 ≤ 0.077 rad, 오버슈트 ≤ 0.004 rad다.
- **torso_1 단독 +72°:** 베이스 자유면 로봇이 앞으로 넘어지고(기울기 120°), 베이스를 고정해도 effort 270 N·m에서 포화돼 자세를 못 버틴다. 실기에서도 금지 자세이니, 몸통은 torso_1·2·3을 묶은 자세(T3b)로만 쓰는 것이 맞다.
- NaN이나 폭주는 없다. 링크 원점 최저 높이는 0.04 m(넘어진 torso_1 경우)로 바닥 관통도 없다.

### T3d 그리퍼

- **l2/r2 그대로(원본, 거의 수동) · 원본 게인** — ❌ 실패: 좌우 비대칭 최대 50 mm, 큐브 상승 0.18 m, 들기 중 미끄러짐 16.3 mm, 10 s 유지 미끄러짐 1.6 mm, 낙하 ❌, 도달 오차 69.1 mm · 기준 {'symmetric(<=2mm)': False, 'lift>=0.18m': False, 'slip<=1cm': True} (`t3/gripper_implicit_gpu_dt500_n1.json`)
  - 열기/닫기: 명령 0.05: r1 -0.0500 r2 +0.0000 l1 -0.0500 l2 +0.0000; 명령 0.0: r1 -0.0000 r2 +0.0002 l1 -0.0000 l2 +0.0001; 명령 0.05: r1 -0.0500 r2 +0.0017 l1 -0.0500 l2 +0.0017; 명령 0.0: r1 -0.0000 r2 +0.0032 l1 -0.0000 l2 +0.0031
- **l2/r2 소프트웨어 mimic · 원본 게인** — ❌ 실패: 좌우 비대칭 최대 0.00928 mm, 큐브 상승 0.144 m, 들기 중 미끄러짐 0.801 mm, 10 s 유지 미끄러짐 0.0797 mm, 낙하 ❌, 도달 오차 69.1 mm · 기준 {'symmetric(<=2mm)': True, 'lift>=0.18m': False, 'slip<=1cm': True} (`t3/gripper_implicit_gpu_dt500_n1_mimic.json`)
  - 열기/닫기: 명령 0.05: r1 -0.0500 r2 +0.0500 l1 -0.0500 l2 +0.0500; 명령 0.0: r1 -0.0000 r2 +0.0000 l1 -0.0000 l2 +0.0000; 명령 0.05: r1 -0.0500 r2 +0.0500 l1 -0.0500 l2 +0.0500; 명령 0.0: r1 -0.0000 r2 +0.0000 l1 -0.0000 l2 +0.0000
- **l2/r2 mimic · 조정안 (점성 + ζ0.7 + 중력보상)** — ✅ 통과: 좌우 비대칭 최대 0.00194 mm, 큐브 상승 0.198 m, 들기 중 미끄러짐 0.0649 mm, 10 s 유지 미끄러짐 0.111 mm, 낙하 ❌, 도달 오차 0.484 mm · 기준 {'symmetric(<=2mm)': True, 'lift>=0.18m': True, 'slip<=1cm': True} (`t3/gripper_implicit_gpu_dt500_n1_mimic_tuned_gc.json`)
  - 열기/닫기: 명령 0.05: r1 -0.0500 r2 +0.0500 l1 -0.0500 l2 +0.0500; 명령 0.0: r1 -0.0000 r2 -0.0000 l1 -0.0000 l2 -0.0000; 명령 0.05: r1 -0.0500 r2 +0.0500 l1 -0.0500 l2 +0.0500; 명령 0.0: r1 -0.0000 r2 -0.0000 l1 -0.0000 l2 -0.0000
- TCP(손가락 충돌체 박스 중심, ee_right 기준): [-0.0, -0.0, -0.1045] m, 접근축 [0.0, 0.0, -1.0], 손가락 박스 크기 [0.032, 0.032, 0.062] m

- **l2/r2는 수동 관절이다.** 원본 rb_gripper 서버는 `gripper_finger_l1`·`r1`만 구동하고(0 = 닫힘, −0.05 = 열림), l2/r2는 USD의 약한 드라이브(K 0.309 N/m, D 1.2e-4)만 가진 사실상 수동 관절이다. 그래서 열기 명령을 내려도 l2/r2가 따라오지 않고(비대칭 50 mm), 큐브를 잡으면 r2가 한계(+50 mm)까지 밀려나 한쪽 손가락과 끝멈춤 사이에 끼인다(들 때 16 mm 미끄러짐).
- **소프트웨어 mimic**(l2/r2를 같은 게인으로 −l1 목표로 구동): 대칭 0.01 mm, 큐브를 중앙에 잡고, 미끄러짐은 들 때 0.8 mm·10초 유지 0.08 mm다.
- **"20 cm 들기" 실패는 팔 처짐 때문이다.** 원본 게인에서는 grasp 자세 TCP가 중력 처짐으로 6.9 cm 아래에 있고, 실제 상승도 14.4 cm에 그친다. 그리퍼 문제가 아니다. 조정안(mimic + 중력보상)에서는 세 기준을 모두 통과한다.
- **측정 방법:** IK는 기구학 반복(관절 순간 이동)으로 미리 풀었고, 큐브는 팔이 grasp 자세에 안착한 뒤 실제 TCP 위치에 놓았다. 그래서 팔 추종 오차와 그리퍼 성능이 분리된다. 큐브는 5 cm, 0.1 kg, μ 1.0이다.

### T3e 바퀴 주행

| 조건 | 캐스터 μ | 판정 | 직진 실제 / 명령 m | 오차(명령) | 오차(바퀴) | 바퀴 추종 | 회전 실제 / 명령 rad | 오차(명령) | 오차(바퀴) | 바퀴 추종 |
|---|---|---|---|---|---|---|---|---|---|---|
| implicit 속도 드라이브(D=100) · 1/500 | 0.3 | ❌ 실패 | 0.707 / 0.825 | 14 % | 5.1 % | 9.9 % | 0.464 / 1.37 | 66 % | 55 % | 28 % |
| implicit 속도 드라이브(D=100) · 1/200 | 0.3 | ❌ 실패 | 0.7 / 0.824 | 15 % | 4.8 % | 11 % | 0.436 / 1.37 | 68 % | 56 % | 32 % |
| ideal_pd 원본식(적분 위치목표+kp/kd) · 1/500 | 0.3 | ❌ 실패 | 0.765 / 0.825 | 7.2 % | 6.2 % | 1.2 % | 0.626 / 1.37 | 54 % | 53 % | 3.4 % |
| implicit 속도 드라이브 · 1/500 | 0.0 | ❌ 실패 | 0.788 / 0.825 | 4.4 % | 0.0079 % | 4.4 % | 1.17 / 1.37 | 15 % | 8 % | 7.8 % |
| ideal_pd 원본식 · 1/500 | 0.0 | ✅ 통과 | 0.821 / 0.825 | 0.45 % | 0.011 % | 0.44 % | 1.25 / 1.37 | 8.8 % | 8 % | 0.84 % |

- **바퀴 제어:** 원본은 "적분된 위치 목표 + kp/kd"(kp 262.8, kd 3754.9 → D 7.51)이고, 기본 설정은 승인대로 damping만 둔 속도 드라이브(D = 100)로 단순화했다. 두 방식의 차이는 다음과 같다.
  - 속도 드라이브는 바퀴 점성 마찰(3 N·m·s/rad)과 캐스터 끌림에 비례하는 정상상태 지연이 남는다. 바퀴 추종 오차는 직진 4~10%, 회전 8~28%다.
  - 원본식은 위치 적분항이 있어 지연이 없다(0.4~3.4%).
  - **원본식을 Isaac Lab에서 쓰려면 랩 처리가 필요하다.** 바퀴처럼 한계 없는 revolute 관절의 `joint_pos`는 감겨서 보고되어(한 스텝에 4π 점프 관측) 위치 오차를 unwrap해야 한다. 그렇지 않으면 오차가 튀어 로봇이 4 m 폭주했다(첫 측정에서 확인).
- **부호:** 바퀴 관절 프레임이 x축 기준 180° 뒤집혀 있어(localRot (0, 1, 0, 0)) **바퀴 속도 +가 후진**이다. 원본 기준 목표 (−1.0, −0.5)와도 맞다.
- **캐스터:** 베이스의 실린더 4개(mu 0.3, 원본 `MODEL_A_EXTRA_GEOMETRY`)가 구르지 않고 미끄러진다. 직진에서는 헤딩이 흐르고(−0.13 rad), 제자리 회전은 54~66% 덜 돈다. μ = 0이면 직진 0.45%, 회전 8.8%(원본식)로 통과한다. 남은 8%는 회전 중 바퀴 미끄러짐이다.
- **기구학 상수:** 바퀴 반지름 0.1 m와 간격 0.53 m는 USD 관절 원점(y ±0.265, z 0.1)과 일치한다. 바퀴 측정 회전에서 예측한 직진 거리와 실제 이동은 0.01% 이내로 같다.

### T3f 카메라

판정 ✅ 통과

| 카메라 | RGB 평균/표준편차 | depth 유효 비율 | depth 범위 m | 방향 오차° | 위치 오차 mm | fx fy cx cy | 이미지 |
|---|---|---|---|---|---|---|---|
| head | 217.7 / 30.9 | 0.73 | [0.48166531324386597, 19.99978256225586] | 0.000 | 0.00 | 343.1 343.1 320.0 240.0 | `images/t3f_head_rgb.png` `images/t3f_head_depth.png` |
| left_wrist | 214.0 / 34.3 | 0.64 | [0.04794403538107872, 19.329681396484375] | 0.000 | 0.00 | 343.1 343.1 320.0 240.0 | `images/t3f_left_wrist_rgb.png` `images/t3f_left_wrist_depth.png` |
| right_wrist | 212.4 / 34.6 | 0.64 | [0.04794413596391678, 16.98945426940918] | 0.025 | 0.00 | 343.1 343.1 320.0 240.0 | `images/t3f_right_wrist_rgb.png` `images/t3f_right_wrist_depth.png` |

- 기존 USD 카메라 prim 3개에 `CameraCfg(spawn=None)`을 붙여 640×480 RGB·depth를 저장했다. 세 장 모두 검은 화면이 아니다.
- 방향은 `camera_frames.json`과 0.03° 이내로 일치한다. 이때 `update_latest_camera_pose=True`가 필요하다. 기본값 False면 `data.pos_w`·`quat_w_*`가 초기화 시점 자세에 멈춰서 첫 측정에서 21°·79° 오차로 나왔다.
- 내부 파라미터는 fx = fy = 343.1, cx = 320, cy = 240이다(f 19.3 mm, 수평 aperture 36 mm, 수평 FOV 86°, clip 0.01~20 m).
- 관찰 카메라 이미지 `images/t3f_observer_rgb.png`에서 로봇 전체가 정상으로 렌더링된다.

### T3g 물리 장치·복제

| 조건 | 판정 | 최대 관절오차 rad (관절) | 마지막 3 s p2p rad | 진동 관절 수* | 베이스 이동 mm | wall s | 파일 |
|---|---|---|---|---|---|---|---|
| GPU · num_envs 1 | ✅ 통과 | 0.01683 (left_arm_0) | 0.004512 (right_arm_0) | 0 | 0.000754 | 17.5 | `t3/idle_implicit_gpu_dt500_n1.json` |
| GPU · num_envs 4 | ✅ 통과 | 0.01736 (left_arm_0) | 0.004491 (right_arm_0) | 0 | 0.00172 | 13.9 | `t3/idle_implicit_gpu_dt500_n4.json` |
| CPU · num_envs 1 | ✅ 통과 | 0.004985 (torso_1) | 0.001261 (torso_2) | 0 | 0.000245 | 14.7 | `t3/idle_implicit_cpu_dt500_n1.json` |

num_envs 4 환경별 최대 관절오차: 0.01736, 0.01672, 0.01673, 0.01673

- GPU와 CPU 모두 T3a를 통과한다. CPU는 0.005 rad, GPU는 0.017 rad로, 차이는 Coulomb 마찰 처리에서 온다. 조정안(점성 마찰만)에서는 GPU와 CPU 결과가 같은 수준이다.
- num_envs 4에서 환경별 최대 오차는 0.0167~0.0174 rad로 서로 같다. 복제 환경도 안정적이다.

## 6. DX 침실 씬 배치 (T4)

#### 방식 A — `*_CareRoom_scene.usda` 스테이지를 연 뒤 로봇 추가 (요청 방식)

판정 ✅ 통과 · 기준 {'gpu_physics': True, 'no_static_overlap(collider links)': True, 'robot_stable(base<1cm, tilt<2deg)': True, 'deformables_finite': True, 'bed_in_head_camera': True}

- 물리 장치 cuda:0, 씬 PhysicsScene 재사용 ['/World/PhysicsScene'], dt 0.0050, 10.0 s
- 시작 위치 [-2.6, 17.7, 0.0], yaw 0.0°
- 베이스 최대 이동 0.01 mm, 최대 기울기 0.000°, 관절 최대 오차 0.2614 rad (최종 0.0326)
- 정적 겹침(링크 AABB vs 가구 AABB): 0건
- 가장 가까운 가구: base↔CareDesk 0.507 m, link_torso_4↔CareDesk 0.636 m, base↔TShirt 0.645 m, link_torso_5↔CareDesk 0.655 m, link_torso_0↔CareDesk 0.692 m
- 충돌체 없는 링크 8개: link_torso_3, link_head_0, link_head_1, link_head_2, link_left_arm_6, link_right_arm_6, ee_left, ee_right
- 머리 카메라: 침대 중심 화면 안 ✅, 침대 AABB 꼭짓점 8/8개 화면 안, 중심 픽셀 [145, 217]
- 변형체(실행 전 → 후): pillow: z 0.545~0.683 → 0.548~0.689, finite True; blanket: z 0.535~0.577 → 0.297~0.559, finite True; tshirt: z 0.735~0.743 → 0.727~0.736, finite True
- 실행 로그 경고(잡음 제외): `[Warning] [omni.physicsschema.plugin] Non-uniform scale may result in a non matching collision representation on prim: /World/envs/env_N/Robot/link_right_arm_5/` ×3; `[Warning] [omni.physicsschema.plugin] Non-uniform scale may result in a non matching collision representation on prim: /World/envs/env_N/Robot/link_left_arm_5/c` ×3; `[Warning] [rtx.hydra] RenderDelegate : Warning /rtx/hydra/readTransformsFromFabricInRenderDelegate and geometry streaming are enabled together but this can caus` ×2; `[Warning] [omni.fabric.plugin] Unsupported type encountered during VtValue extraction` ×1; `[Warning] [omni.fabric.plugin] Attempting to set an invalid data source to array attribute` ×1
- 이미지: `t4/head_rgb_open.png`, `t4/head_depth_open.png`, `t4/overview_open.png`

#### 방식 B — Isaac Lab 스테이지 + `CareRoom.usda`를 전역 asset으로 참조 (Arena 방식)

판정 ✅ 통과 · 기준 {'gpu_physics': True, 'no_static_overlap(collider links)': True, 'robot_stable(base<1cm, tilt<2deg)': True, 'deformables_finite': True, 'bed_in_head_camera': True}

- 물리 장치 cuda:0, 씬 PhysicsScene 재사용 [], dt 0.0050, 10.0 s
- 시작 위치 [-2.6, 17.7, 0.0], yaw 0.0°
- 베이스 최대 이동 0.01 mm, 최대 기울기 0.000°, 관절 최대 오차 0.2614 rad (최종 0.0326)
- 정적 겹침(링크 AABB vs 가구 AABB): 0건
- 가장 가까운 가구: base↔CareDesk 0.507 m, link_torso_4↔CareDesk 0.636 m, base↔TShirt 0.645 m, link_torso_5↔CareDesk 0.655 m, link_torso_0↔CareDesk 0.692 m
- 충돌체 없는 링크 8개: link_torso_3, link_head_0, link_head_1, link_head_2, link_left_arm_6, link_right_arm_6, ee_left, ee_right
- 머리 카메라: 침대 중심 화면 안 ✅, 침대 AABB 꼭짓점 8/8개 화면 안, 중심 픽셀 [145, 217]
- 변형체(실행 전 → 후): pillow: z 0.545~0.683 → 0.548~0.689, finite True; blanket: z 0.535~0.577 → 0.297~0.559, finite True; tshirt: z 0.735~0.743 → 0.727~0.736, finite True
- 실행 로그 경고(잡음 제외): `[Warning] [omni.physicsschema.plugin] Non-uniform scale may result in a non matching collision representation on prim: /World/envs/env_N/Robot/link_left_arm_5/c` ×3; `[Warning] [omni.physicsschema.plugin] Non-uniform scale may result in a non matching collision representation on prim: /World/envs/env_N/Robot/link_right_arm_5/` ×3; `[Warning] [rtx.hydra] RenderDelegate : Warning /rtx/hydra/readTransformsFromFabricInRenderDelegate and geometry streaming are enabled together but this can caus` ×1; `[Warning] [omni.fabric.plugin] Unsupported type encountered during VtValue extraction` ×1; `[Warning] [omni.fabric.plugin] Attempting to set an invalid data source to array attribute` ×1
- 이미지: `t4/head_rgb_reference.png`, `t4/head_depth_reference.png`, `t4/overview_reference.png`

- **두 방식 모두 통과한다.**
  - 로봇은 GPU 물리(`cuda:0`)에서 변형체 3개(베개 volume 351노드, 이불 surface 1804노드, 티셔츠 surface 3444노드)와 함께 10초 동안 안정적이다. 베이스 이동은 0.01 mm다.
  - 시작 위치의 링크 AABB는 가구와 겹치지 않는다. 가장 가까운 것은 base↔CareDesk 0.51 m다.
  - 머리 카메라에 침대 AABB 8꼭짓점이 모두 들어온다(`t4/head_rgb_*.png`, 주황 상자).
- **방식 A** = 요청 방식이다: `*_CareRoom_scene.usda`를 열고 → SimulationContext(cuda:0) → 로봇 추가. 씬의 `/World/PhysicsScene`(GPU dynamics, TGS)을 그대로 재사용한다. Isaac Lab이 SimulationContext 안에서 장치를 설정하므로, "스테이지를 연 뒤에 장치 설정" 순서가 자동으로 지켜진다.
- **방식 B** = Arena 방식이다: Isaac Lab 스테이지에 `CareRoom.usda`를 전역 asset으로 참조한다. 같은 prim 경로가 나오고, 물리 씬은 Isaac Lab이 GPU dynamics로 만든다. 2단계·3단계에서는 B가 자연스럽다.
- **최종 관절 오차 0.033 rad**는 1/200 s에서 원본 게인을 쓴 결과다. T3a의 1/200 크리프와 같은 현상이다.
- 이불은 10초 동안 매트리스 가장자리로 흘러내려 z_min이 0.535 → 0.297 m가 된다. 씬 자체의 거동이고 로봇과는 무관하다.
- **렌더링 함정:** `/World/...` 전역에 둔 카메라에는 `/World/envs/env_*` 아래 prim(로봇 포함)이 렌더링되지 않는다. 처음에는 overview 이미지에 로봇이 없었다. 관찰용 카메라도 `{ENV_REGEX_NS}` 아래에 두어야 한다. 머리 카메라처럼 env 안의 카메라는 env 내부와 전역 prim을 모두 본다.

## 7. 실패 원인과 수정 제안

| 실패 항목 | 원인 | 수정 제안 | 승인 필요 |
|---|---|---|---|
| T3a 1/200, T4 최종 오차 0.033 | 원본 Coulomb 마찰이 GPU에서 크리프 | 정지/동 마찰 제거, 점성만 유지 (`preset="tuned"`) | 예 (게인 체계) |
| 감쇠 부족 (마찰을 빼면 진동) | damping = kd×0.002 → ζ 0.01~0.04 | 몸체 damping을 ζ = 0.7 값으로 (`DAMPING_ZETA07`) | 예 (게인 체계) |
| T3b·T3c·T3d 처짐 | 원본은 SDK feedforward로 중력보상, 시뮬은 ff = 0 | 제어 루프에서 `gravity_compensation_forces`를 effort target으로 더함 | 예 (제어 구조) |
| T3c torso_1 단독 +72° | 물리적 한계 (넘어짐, 토크 포화) | 스윕 범위에서 제외하고 몸통은 자세 단위로만 사용 | 아니오 (기준 조정) |
| T3d 비대칭·미끄러짐 | l2/r2가 사실상 수동 관절 | (단기) 소프트웨어 mimic, (2단계) USD에 `PhysxMimicJointAPI`로 l2 = −l1 | USD 변경 시 예 |
| T3e 회전 오차 | 캐스터가 미끄러지는 고정 실린더 μ 0.3 | 캐스터 μ = 0(+ friction combine "min"), 또는 2단계에서 swivel/roll 관절이 있는 캐스터 | USD 변경 시 예 |
| T3e 원본식 바퀴 폭주 | `joint_pos` 랩 | 위치 오차 unwrap, 또는 속도 드라이브 사용 | 아니오 |
| (T4 관찰용 이미지) | 전역 카메라가 env prim을 렌더링하지 않음 | 카메라를 `{ENV_REGEX_NS}` 아래에 생성 | 아니오 |
| 충돌체 빈 곳 | torso_3·머리·arm_6·ee 무충돌, arm_5 캡슐 비균일 스케일 | 2단계에서 `fullbody_collision` 변형을 base로 + 빠진 링크에 convexHull 추가 | 예 (2단계로 미룸, 결정됨) |
| tool 관절 16 mm | USD xform(−0.1101) vs 관절(−0.1261) | 시뮬은 관절을 따르므로 동작에는 문제없음. USD 재생성 때 xform을 −0.1261로 맞추면 정적 FK와 시뮬이 일치 | USD 재생성 시 |

## 8. 2단계용 사실 정리

**쓸 USD (추천)**
- 지금 바로 쓸 것: `tools/rby1_check/overrides/rby1_24dof.usda` (`rby1_v1_2_a_rbgripper_d405_collision.usd` 위의 덮어쓰기 레이어: 24-DOF + 자유 베이스 + root = base + 손가락 충돌체).
- 2단계 정식 embodiment: `make_override.py --base fullbody_collision`으로 arm_5 캡슐이 구워진 변형을 base로 삼고, 빠진 충돌체·캐스터·mimic을 같은 레이어에 추가하는 것을 권한다.

**게인** (`rby1_cfg_draft.py`)
- 원본 재현(기본): stiffness = `rby1_task.py` kp, damping = kd × `PD_CONTROL_DT`(1/500). 원본 PD 식은 `τ = feedback_gain·(kp·e + kd·ė·dt) + ff`다.
- `feedback_gain`은 기본 1이고, rby1-sdk 명령으로 들어올 때 `clip(gain / 10, 0, 1)`로 스케일된다(`rby1_task.py` 533행). `ff`는 SDK가 보내는 feedforward 토크(중력보상)다.
- 조정안: `preset="tuned"` — 점성 마찰만, damping = `DAMPING_ZETA07`(ζ = 0.7, 영 자세), l2/r2 mimic 구동. 제어 루프에서 중력보상 effort target을 더하고, 캐스터 μ = 0으로 둔다. 값은 위 감쇠비 표의 D(ζ = 0.7) 열이다.
- 바퀴: 기본 implicit 속도 드라이브(stiffness 0, damping 100). 원본식은 IdealPD + 적분 위치목표(kp 262.8, D 7.51) + unwrap이다.
- 그리퍼: l1/r1 stiffness 100 N/m, damping 1, effort 75 N(USD), 속도 0.05 m/s.
- 마찰·armature(원본 `motor_profiles.py`): 몸통 하부 (10, 10, 20, 1.7), 몸통 상부 (10, 10, 10, 1.7), 어깨 (5.2, 5.2, 4.7, 0.29), 팔꿈치 (2.2, 2.2, 1.9, 0.4), 손목 (2.2, 2.2, 1.9, 0.2), 손목2 (1.2, 1.2, 1.0, 0.2), 머리·바퀴 default (2, 2, 3, 0.5) = (정지, 동, 점성 N·m·s/rad, armature).
- 솔버 반복 4/1, 자기충돌 꺼짐.

**한계와 effort:** 아래 관절 표(시뮬에서 읽은 값)를 본다. effort는 URDF(몸통 270/120, 팔 70/70/70/40/10/10/8, 머리 70, 바퀴 100), 속도는 URDF다. USD maxForce(원본 클리핑 값)는 T2 표에 있다.

**엔드이펙터와 TCP**
- `ee_left`/`ee_right`는 강체이고 경로는 `/RBY1/{left,right}_gripper/ee_{side}`다. `tool_{side}` 고정 관절로 `link_{side}_arm_6`에 (0, 0, −0.1261) m 오프셋, 회전 없이 붙는다(시뮬 기준. USD xform은 −0.1101).
- 영 자세에서 ee 축은 베이스 축과 같고, 위치는 (0, ±0.22, 0.712) m다.
- **TCP = ee 프레임 (0, 0, −0.1045) m** (닫힌 손가락 충돌 박스 중심). 손끝은 −0.1355 m(arm_6 기준 −0.2616 m, PDF "손목~손끝 0.262 m"와 일치)이고, 접근축은 ee −z다.
- 손가락 박스는 3.2 × 3.2 × 6.2 cm, 최대 개폭은 10 cm(각 손가락 0~5 cm)다.

**카메라** (기존 prim. Isaac Lab에서는 `CameraCfg(spawn=None, update_latest_camera_pose=True)`)

| 이름 | prim (로봇 root 기준) | 부모 기준 위치 m | 640×480 내부 파라미터 |
|---|---|---|---|
| 머리 | `link_head_2/first_person_camera` | (0.048, −0.0003, 0.057) | fx = fy = 343.1, cx 320, cy 240, 수평 FOV 86°, clip 0.01–20 m |
| 왼손목 | `left_gripper/ee_left/left_hand_camera` | (0, 0.0617, −0.0754), 20.6° 기울임 | 같음 |
| 오른손목 | `right_gripper/ee_right/right_hand_camera` | (0, −0.0617, −0.0754), 20.6° 기울임 | 같음 |

회전은 `assets/generated/camera_frames.json`(USD 카메라 규약 = OpenGL, −Z가 시선)과 일치한다.

**Isaac Lab 3.0에서 확인한 사실**
- 쿼터니언은 (x, y, z, w)이다.
- body 순서 첫 번째가 `base`(root)이고, 관절 순서는 아래 표와 같다. Isaac Lab은 BFS 순서라 C++ 순서와 다르므로 인덱스 매핑이 필요하다.
- `data.gravity_compensation_forces`와 `root_view.get_dof_projected_joint_forces()`를 쓸 수 있다.
- `data.mass_matrix` 대각에는 armature가 포함되지 않는다(arm_6 M_ii 0.001 vs armature 0.2).
- 변형체 노드는 `PhysxManager.get_physics_sim_view().create_{surface,volume}_deformable_body_view()`로 읽는다(`isaacsim.core.experimental.prims` 확장은 Isaac Lab 앱에서 켜면 Kit이 죽었다).
- 전역 카메라는 env prim을 렌더링하지 않는다.
- Isaac Lab 앱은 Kit 로그 파일을 쓰지 않아서, 경고는 stdout에서 모은다.
- Kit이 시작 1~2초 안에 `platforminfo`에서 가끔 segfault로 죽는다(이번에 24회 중 2회, 재실행하면 됨).

### 관절 순서 (Isaac Lab ↔ rby1-sdk)

| Isaac Lab 인덱스 | 관절 | rby1-sdk C++ 인덱스 (`CPP_JOINT_NAMES`) | 한계 rad | effort 한계 | 속도 한계 |
|---|---|---|---|---|---|
| 0 | left_wheel | 1 | 무한 (continuous) | 100 | 15.71 |
| 1 | right_wheel | 0 | 무한 (continuous) | 100 | 15.71 |
| 2 | torso_0 | 2 | [-0.2618, 0.2618] | 270 | 2.094 |
| 3 | torso_1 | 3 | [-0.5236, 1.5708] | 270 | 2.094 |
| 4 | torso_2 | 4 | [-2.6180, 1.5708] | 270 | 2.094 |
| 5 | torso_3 | 5 | [-0.7854, 1.5708] | 120 | 3.142 |
| 6 | torso_4 | 6 | [-0.5236, 0.5236] | 120 | 3.142 |
| 7 | torso_5 | 7 | [-2.3562, 2.3562] | 120 | 3.142 |
| 8 | left_arm_0 | 15 | [-3.1416, 3.1416] | 70 | 3.142 |
| 9 | right_arm_0 | 8 | [-3.1416, 3.1416] | 70 | 3.142 |
| 10 | head_0 | 22 | [-1.5708, 1.5708] | 70 | 3.14 |
| 11 | left_arm_1 | 16 | [0.0000, 3.1416] | 70 | 3.142 |
| 12 | right_arm_1 | 9 | [-3.1416, 0.0000] | 70 | 3.142 |
| 13 | head_1 | 23 | [-1.5708, 1.5708] | 70 | 3.14 |
| 14 | left_arm_2 | 17 | [-3.1416, 3.1416] | 70 | 3.142 |
| 15 | right_arm_2 | 10 | [-3.1416, 3.1416] | 70 | 3.142 |
| 16 | left_arm_3 | 18 | [-2.6180, 0.0000] | 40 | 3.142 |
| 17 | right_arm_3 | 11 | [-2.6180, 0.0000] | 40 | 3.142 |
| 18 | left_arm_4 | 19 | [-3.1416, 3.1416] | 10 | 6.283 |
| 19 | right_arm_4 | 12 | [-3.1416, 3.1416] | 10 | 6.283 |
| 20 | left_arm_5 | 20 | [-1.5708, 1.9199] | 10 | 6.283 |
| 21 | right_arm_5 | 13 | [-1.5708, 1.9199] | 10 | 6.283 |
| 22 | left_arm_6 | 21 | [-2.7053, 2.7053] | 8 | 2.094 |
| 23 | right_arm_6 | 14 | [-2.7053, 2.7053] | 8 | 2.094 |
| 24 | gripper_finger_l1 | — (그리퍼) | [-0.0500, 0.0000] | 75 | 0.05 |
| 25 | gripper_finger_l2 | — (그리퍼) | [0.0000, 0.0500] | 1000 | 0.05 |
| 26 | gripper_finger_r1 | — (그리퍼) | [-0.0500, 0.0000] | 75 | 0.05 |
| 27 | gripper_finger_r2 | — (그리퍼) | [0.0000, 0.0500] | 1000 | 0.05 |

C++ 순서 → Isaac Lab 인덱스: `[1, 0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 17, 19, 21, 23, 8, 11, 14, 16, 18, 20, 22, 10, 13]`

body 순서(첫 번째가 articulation root): `['base', 'wheel_l', 'wheel_r', 'link_torso_0', 'link_torso_1', 'link_torso_2', 'link_torso_3', 'link_torso_4', 'link_torso_5', 'link_head_0', 'link_left_arm_0', 'link_right_arm_0', 'link_head_1', 'link_left_arm_1', 'link_right_arm_1', 'link_head_2', 'link_left_arm_2', 'link_right_arm_2', 'link_left_arm_3', 'link_right_arm_3', 'link_left_arm_4', 'link_right_arm_4', 'link_left_arm_5', 'link_right_arm_5', 'link_left_arm_6', 'link_right_arm_6', 'ee_left', 'ee_right', 'ee_finger_l1', 'ee_finger_l2', 'ee_finger_r1', 'ee_finger_r2']`

## 9. 이미지·로그 경로

- **T3 그래프:** `t3/*_idle_err.png`, `t3/*_crouch_err.png`(관절 추종 오차), `t3/*_slip.png`(큐브 미끄러짐)
- **T3f 이미지:** `images/t3f_{head,left_wrist,right_wrist}_{rgb,depth}.png`, depth 원본 `images/t3f_*_depth.npy`, 관찰 카메라 `images/t3f_observer_rgb.png`
- **T4 이미지:** `t4/head_rgb_{open,reference}.png`(침대 AABB 투영 표시), `t4/head_rgb_raw_*.png`, `t4/head_depth_*.png`, `t4/overview_{open,reference}.png`
- **로그:** 실행별 stdout은 `t3/logs/*.log`, `t4_run_{open,reference}.log`, `t1_run.log`
- **영상은 만들지 않았다.** 필요하면 `--viz kit`로 화면을 보거나 2단계에서 녹화를 붙인다.

대표 실행의 로그 경고(잡음 제외):

- T1 (USD 진단 음소거 해제): 없음
- T3a 기본: `[Warning] [omni.physicsschema.plugin] Non-uniform scale may result in a non matching collision representation on prim: /World/envs/env_N/Robot/link_left_arm_5/c` ×2; `[Warning] [omni.physicsschema.plugin] Non-uniform scale may result in a non matching collision representation on prim: /World/envs/env_N/Robot/link_right_arm_5/` ×2; `[Warning] [usdrt.population.plugin] Last global time was not created. Did you call setEnableUsdNoticeHandling?` ×1
- T3f 카메라: `[Warning] [omni.physicsschema.plugin] Non-uniform scale may result in a non matching collision representation on prim: /World/envs/env_N/Robot/link_left_arm_5/c` ×2; `[Warning] [omni.physicsschema.plugin] Non-uniform scale may result in a non matching collision representation on prim: /World/envs/env_N/Robot/link_right_arm_5/` ×2; `[Warning] [rtx.hydra] RenderDelegate : Warning /rtx/hydra/readTransformsFromFabricInRenderDelegate and geometry streaming are enabled together but this can caus` ×1
