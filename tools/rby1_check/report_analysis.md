<!-- Hand-written analysis inserted into REPORT.md by make_report.py. Sections start with SECTION markers. -->

<!-- SECTION:summary -->
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

<!-- SECTION:verdict_notes -->
판정은 요청서의 제안 기준 그대로다. "원본 게인" 행이 실패하고 "조정안" 행이 통과하는 항목은, 실패 원인이 USD나 물리 엔진이 아니라 게인·제어 방식에 있다는 뜻이다.
- **T3c 베이스 자유:** 실패 원인은 torso_1 단독 +72°에서 로봇이 앞으로 넘어지는 것 하나다. 상체 전체가 앞바퀴 축 밖으로 나가는 자세라 실기에서도 넘어진다.
- **T3c 베이스 고정:** 넘어짐 기준만 보므로 통과로 표시되지만, torso_1은 effort 270 N·m에서 포화되어 72°를 버티지 못한다(최대 오차 1.99 rad).

<!-- SECTION:t1 -->
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

<!-- SECTION:t2 -->
- **기구학:** 몸체 24관절의 원점과 축이 영 자세에서 URDF와 일치한다. `right/left_arm_0`은 USD `physics:axis`가 빈 토큰이지만, 기본값 X로 읽으면 URDF의 기울어진 축 (0, 0.94, −0.34)과 정확히 일치한다. 스윕에서도 정상으로 움직였다.
- **effort:** USD drive `maxForce`가 URDF effort보다 크다. 몸통 841/369 vs 270/120, 팔 143/143/143/70/70/70/140 vs 70/70/70/40/10/10/8, 머리 70 vs 1000(URDF 값은 자리표시로 보임). 원본은 USD maxForce로 클리핑했고, 이번 설정 초안은 URDF effort를 쓴다(머리만 USD 70).
- **한계:** arm_1·arm_3에서 URDF가 0 너머로 1° 더 허용한다. 속도는 arm_6이 URDF 2.09 vs USD 6.28 rad/s, 바퀴가 15.7 vs 17.5 rad/s다.
- **게인 표의 불일치:** `src/config.py`의 `JOINT_KD_BASE`×10은 `rby1_task.py`의 kd와 arm_2(321.5 vs 521.5)·arm_6(61.26 vs 91.26)이 다르다. 실제로 쓰이는 값은 `rby1_task.py`다.

<!-- SECTION:t3a -->
- **기본 조건(GPU, 1/500, 원본 재현 게인, 원본 마찰 모델):** 최대 관절 오차 0.0168 rad(left_arm_0)로 통과하고, 베이스 이동은 1 µm 미만이다.
- **1/200 s:** 오차가 10초 동안 한 방향으로 계속 커지는 **크리프**(0.028 rad, arm_3)로 실패한다. 진동이 아니라서 ζ를 0.7·1.0으로 올려도 그대로다(아래 표).
- **CPU:** 같은 조건에서 0.005 rad로, GPU와 Coulomb 마찰 처리가 다르다.
- **`--pose ready`:** `rby1_ready_reach_table.usd`에 0이 아닌 관절 상태가 저장돼 있지 않아 영 자세와 같은 결과가 나왔다.

<!-- SECTION:study -->
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

<!-- SECTION:t3b -->
- **원인은 토크 부족이 아니라 중력보상이 없어서 생기는 처짐이다.**
  - (60°, −90°, 60°) 자세에서 투영 토크/URDF effort는 최대 0.55(torso_1 148 N·m)로 80% 기준 안이다.
  - 관절 오차는 0.088 rad(torso_3: 46.5 N·m ÷ K 573.6 ≈ 0.081 rad)로 기준 0.05를 넘는다.
- **조정안은 세 자세 모두 통과한다**(최대 오차 0.001~0.004 rad).
- **토크 여유가 가장 작은 자세:** (75°, −105°, 90°)에서 torso_1이 effort의 73%다. PDF의 "앞 0.5 m에서 정격 71%"와 같은 수준이다.
- **손끝 높이:** (60, −90, 60)에서 0.49 m → (90, −130, 90)에서 0.30 m(원본 게인, 처짐 포함) / 0.375 m(조정안). 바닥 집기 자세는 몸통만으로는 닿지 않으니, 2단계에서 팔·손목 IK와 함께 다시 잡아야 한다.
- 기울기는 모두 0°라서 넘어질 위험이 없다.

<!-- SECTION:t3c -->
- 조건: 각 관절을 0 → 0.8·상한 → 0.8·하한 → 0으로 10초 사인파로 움직이고, 속도 feedforward를 준다. 관절마다 상태를 리셋한다.
- **원본 게인**
  - 팔·머리의 최대 오차 0.13~0.19 rad는 대부분 중력 처짐이다. 팔을 수평으로 들면 어깨 중력 토크가 약 20 N·m인데 K가 208.6이다.
  - 몸통은 오버슈트가 0.03~0.13 rad로 ζ가 낮은 결과가 그대로 보인다.
- **조정안:** torso_1을 빼고 모든 관절이 최대 오차 ≤ 0.077 rad, 오버슈트 ≤ 0.004 rad다.
- **torso_1 단독 +72°:** 베이스 자유면 로봇이 앞으로 넘어지고(기울기 120°), 베이스를 고정해도 effort 270 N·m에서 포화돼 자세를 못 버틴다. 실기에서도 금지 자세이니, 몸통은 torso_1·2·3을 묶은 자세(T3b)로만 쓰는 것이 맞다.
- NaN이나 폭주는 없다. 링크 원점 최저 높이는 0.04 m(넘어진 torso_1 경우)로 바닥 관통도 없다.

<!-- SECTION:t3d -->
- **l2/r2는 수동 관절이다.** 원본 rb_gripper 서버는 `gripper_finger_l1`·`r1`만 구동하고(0 = 닫힘, −0.05 = 열림), l2/r2는 USD의 약한 드라이브(K 0.309 N/m, D 1.2e-4)만 가진 사실상 수동 관절이다. 그래서 열기 명령을 내려도 l2/r2가 따라오지 않고(비대칭 50 mm), 큐브를 잡으면 r2가 한계(+50 mm)까지 밀려나 한쪽 손가락과 끝멈춤 사이에 끼인다(들 때 16 mm 미끄러짐).
- **소프트웨어 mimic**(l2/r2를 같은 게인으로 −l1 목표로 구동): 대칭 0.01 mm, 큐브를 중앙에 잡고, 미끄러짐은 들 때 0.8 mm·10초 유지 0.08 mm다.
- **"20 cm 들기" 실패는 팔 처짐 때문이다.** 원본 게인에서는 grasp 자세 TCP가 중력 처짐으로 6.9 cm 아래에 있고, 실제 상승도 14.4 cm에 그친다. 그리퍼 문제가 아니다. 조정안(mimic + 중력보상)에서는 세 기준을 모두 통과한다.
- **측정 방법:** IK는 기구학 반복(관절 순간 이동)으로 미리 풀었고, 큐브는 팔이 grasp 자세에 안착한 뒤 실제 TCP 위치에 놓았다. 그래서 팔 추종 오차와 그리퍼 성능이 분리된다. 큐브는 5 cm, 0.1 kg, μ 1.0이다.

<!-- SECTION:t3e -->
- **바퀴 제어:** 원본은 "적분된 위치 목표 + kp/kd"(kp 262.8, kd 3754.9 → D 7.51)이고, 기본 설정은 승인대로 damping만 둔 속도 드라이브(D = 100)로 단순화했다. 두 방식의 차이는 다음과 같다.
  - 속도 드라이브는 바퀴 점성 마찰(3 N·m·s/rad)과 캐스터 끌림에 비례하는 정상상태 지연이 남는다. 바퀴 추종 오차는 직진 4~10%, 회전 8~28%다.
  - 원본식은 위치 적분항이 있어 지연이 없다(0.4~3.4%).
  - **원본식을 Isaac Lab에서 쓰려면 랩 처리가 필요하다.** 바퀴처럼 한계 없는 revolute 관절의 `joint_pos`는 감겨서 보고되어(한 스텝에 4π 점프 관측) 위치 오차를 unwrap해야 한다. 그렇지 않으면 오차가 튀어 로봇이 4 m 폭주했다(첫 측정에서 확인).
- **부호:** 바퀴 관절 프레임이 x축 기준 180° 뒤집혀 있어(localRot (0, 1, 0, 0)) **바퀴 속도 +가 후진**이다. 원본 기준 목표 (−1.0, −0.5)와도 맞다.
- **캐스터:** 베이스의 실린더 4개(mu 0.3, 원본 `MODEL_A_EXTRA_GEOMETRY`)가 구르지 않고 미끄러진다. 직진에서는 헤딩이 흐르고(−0.13 rad), 제자리 회전은 54~66% 덜 돈다. μ = 0이면 직진 0.45%, 회전 8.8%(원본식)로 통과한다. 남은 8%는 회전 중 바퀴 미끄러짐이다.
- **기구학 상수:** 바퀴 반지름 0.1 m와 간격 0.53 m는 USD 관절 원점(y ±0.265, z 0.1)과 일치한다. 바퀴 측정 회전에서 예측한 직진 거리와 실제 이동은 0.01% 이내로 같다.

<!-- SECTION:t3f -->
- 기존 USD 카메라 prim 3개에 `CameraCfg(spawn=None)`을 붙여 640×480 RGB·depth를 저장했다. 세 장 모두 검은 화면이 아니다.
- 방향은 `camera_frames.json`과 0.03° 이내로 일치한다. 이때 `update_latest_camera_pose=True`가 필요하다. 기본값 False면 `data.pos_w`·`quat_w_*`가 초기화 시점 자세에 멈춰서 첫 측정에서 21°·79° 오차로 나왔다.
- 내부 파라미터는 fx = fy = 343.1, cx = 320, cy = 240이다(f 19.3 mm, 수평 aperture 36 mm, 수평 FOV 86°, clip 0.01~20 m).
- 관찰 카메라 이미지 `images/t3f_observer_rgb.png`에서 로봇 전체가 정상으로 렌더링된다.

<!-- SECTION:t3g -->
- GPU와 CPU 모두 T3a를 통과한다. CPU는 0.005 rad, GPU는 0.017 rad로, 차이는 Coulomb 마찰 처리에서 온다. 조정안(점성 마찰만)에서는 GPU와 CPU 결과가 같은 수준이다.
- num_envs 4에서 환경별 최대 오차는 0.0167~0.0174 rad로 서로 같다. 복제 환경도 안정적이다.

<!-- SECTION:t4 -->
- **두 방식 모두 통과한다.**
  - 로봇은 GPU 물리(`cuda:0`)에서 변형체 3개(베개 volume 351노드, 이불 surface 1804노드, 티셔츠 surface 3444노드)와 함께 10초 동안 안정적이다. 베이스 이동은 0.01 mm다.
  - 시작 위치의 링크 AABB는 가구와 겹치지 않는다. 가장 가까운 것은 base↔CareDesk 0.51 m다.
  - 머리 카메라에 침대 AABB 8꼭짓점이 모두 들어온다(`t4/head_rgb_*.png`, 주황 상자).
- **방식 A** = 요청 방식이다: `*_CareRoom_scene.usda`를 열고 → SimulationContext(cuda:0) → 로봇 추가. 씬의 `/World/PhysicsScene`(GPU dynamics, TGS)을 그대로 재사용한다. Isaac Lab이 SimulationContext 안에서 장치를 설정하므로, "스테이지를 연 뒤에 장치 설정" 순서가 자동으로 지켜진다.
- **방식 B** = Arena 방식이다: Isaac Lab 스테이지에 `CareRoom.usda`를 전역 asset으로 참조한다. 같은 prim 경로가 나오고, 물리 씬은 Isaac Lab이 GPU dynamics로 만든다. 2단계·3단계에서는 B가 자연스럽다.
- **최종 관절 오차 0.033 rad**는 1/200 s에서 원본 게인을 쓴 결과다. T3a의 1/200 크리프와 같은 현상이다.
- 이불은 10초 동안 매트리스 가장자리로 흘러내려 z_min이 0.535 → 0.297 m가 된다. 씬 자체의 거동이고 로봇과는 무관하다.
- **렌더링 함정:** `/World/...` 전역에 둔 카메라에는 `/World/envs/env_*` 아래 prim(로봇 포함)이 렌더링되지 않는다. 처음에는 overview 이미지에 로봇이 없었다. 관찰용 카메라도 `{ENV_REGEX_NS}` 아래에 두어야 한다. 머리 카메라처럼 env 안의 카메라는 env 내부와 전역 prim을 모두 본다.

<!-- SECTION:fixes -->
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

<!-- SECTION:facts -->
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

<!-- SECTION:files -->
- **T3 그래프:** `t3/*_idle_err.png`, `t3/*_crouch_err.png`(관절 추종 오차), `t3/*_slip.png`(큐브 미끄러짐)
- **T3f 이미지:** `images/t3f_{head,left_wrist,right_wrist}_{rgb,depth}.png`, depth 원본 `images/t3f_*_depth.npy`, 관찰 카메라 `images/t3f_observer_rgb.png`
- **T4 이미지:** `t4/head_rgb_{open,reference}.png`(침대 AABB 투영 표시), `t4/head_rgb_raw_*.png`, `t4/head_depth_*.png`, `t4/overview_{open,reference}.png`
- **로그:** 실행별 stdout은 `t3/logs/*.log`, `t4_run_{open,reference}.log`, `t1_run.log`
- **영상은 만들지 않았다.** 필요하면 `--viz kit`로 화면을 보거나 2단계에서 녹화를 붙인다.
