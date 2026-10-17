# RB-Y1 행동·관측 공간 (dxbench embodiments)

제어 주기 15 Hz(정책·기록). physics dt = 1/(15·k), 기본 k = 16(1/240 s)이고 env `--physics_hz`로 바꾼다.
쿼터니언은 Isaac Lab 3.0 규약 (x, y, z, w). 각도 rad, 길이 m, 좌표는 로봇 베이스 프레임(+x 전방, +z 위).

## `rby1_joint` — 정책용, 24차원

| 인덱스 | 이름 | 단위·범위 | 의미 |
|---|---|---|---|
| 0–6 | right_arm_0 … right_arm_6 | rad, 절대 | 오른팔 관절 목표 |
| 7–13 | left_arm_0 … left_arm_6 | rad, 절대 | 왼팔 관절 목표 |
| 14 | gripper_right | 0–1 (1 = 열림) | 오른 그리퍼 개폭. 손가락 r1 = −0.05·g, r2 = +0.05·g (소프트웨어 mimic) |
| 15 | gripper_left | 0–1 (1 = 열림) | 왼 그리퍼 |
| 16–21 | torso_0 … torso_5 | rad, 절대 | 몸통 관절 목표. 큰 숙임은 torso_1·2·3 묶음 프리셋으로 |
| 22 | base_v | m/s, ±0.5 | 베이스 전진 속도 |
| 23 | base_omega | rad/s, ±1.0 | 베이스 yaw 속도 (반시계 +) |

- 0–15는 실기 RB-Y1 데이터셋 `rby1_format`의 16-D와 같은 순서·단위다(오른팔 먼저, 그리퍼 1 = 열림). 실기 포맷은 그리퍼
  행동을 0/1 이진으로 쓰므로, 변환기에서 0.5로 문턱 처리한다(`ParallelGripperActionCfg.binary_threshold`).
- 머리 관절은 행동에 없고 기본 자세(0, 0)를 유지한다.
- 베이스 → 바퀴: v_r = v + ω·W/2, v_l = v − ω·W/2 (오른 바퀴가 −y 쪽) (W = 0.53 m), 바퀴 각속도 = −v_side / 0.1 (바퀴 관절 +속도 = 후진).
- 중력보상: 행동 차원 없는 `gravity` 항이 매 physics step에 몸체 관절(몸통·팔·머리) 중력 토크를 effort target으로 더한다
  (`preset="tuned"`, `original`이면 0).

## `rby1_ik` — 원격조작용, 17차원

| 인덱스 | 이름 | 단위 | 의미 |
|---|---|---|---|
| 0–2 | right_tcp_dx, dy, dz | m / step | 오른 TCP 위치 변화량 (베이스 프레임) |
| 3–5 | right_tcp_rx, ry, rz | rad / step | 오른 TCP 회전 변화량 (axis-angle) |
| 6–11 | left_tcp_* | | 왼 TCP 동일 |
| 12–13 | gripper_right, gripper_left | 0–1 | 위와 같음 |
| 14 | torso_preset_step | −1 / 0 / +1 | +0.5를 넘는 순간 다음 프리셋, −0.5 아래로 내려가는 순간 이전 프리셋 (upright → lean → crouch → deep), 0.6 rad/s로 이동 |
| 15–16 | base_v, base_omega | m/s, rad/s | 위와 같음 |

- TCP = `ee_{side}` 프레임 (0, 0, −0.1045) m(닫힌 손가락 박스 중심), 접근축 ee −z.
- 팔 7관절 목표는 `HomingDiffIKAction`이 만든다. Isaac Lab `DifferentialInverseKinematicsAction`(pose, relative mode, DLS)에 홈 복귀 경로를 더한 것이다.
  - 행동 벡터 밖의 `request_home()`을 부르면 관절공간에서 READY 자세로 돌아간다(원격조작 H 키, oracle). D-24.
  - 기록되는 `action.joint_target`과 dx24 행동에는 홈 복귀가 포함된다.
- 몸통 프리셋 (torso_1, torso_2, torso_3) deg: upright (0, 0, 0), lean (45, −70, 45), crouch (60, −90, 60), deep (90, −130, 90).

## 관측 (`obs["policy"]`, 항목별 dict)

| 키 | 차원 | 내용 |
|---|---|---|
| state16 | 16 | 실기 포맷 상태: 팔 14 (오른팔 먼저) + 그리퍼 개폭 [오른, 왼] (0–1, 손가락 반 간격/0.05 m) |
| torso | 6 | 몸통 관절 위치 |
| joint_vel | 20 | 팔 14 + 몸통 6 관절 속도 |
| gripper | 2 | 그리퍼 개폭 [오른, 왼] |
| base_vel | 2 | 베이스 [전진 속도, yaw 속도] (자기 프레임) |
| base_pose | 3 | env 프레임 [x, y, yaw] |
| tcp_right, tcp_left | 7 | 베이스 프레임 TCP [x, y, z, qx, qy, qz, qw] |
| last_action | 24 / 17 | 직전 행동 |

카메라(`--enable_cameras`): `obs["camera_obs"]` = `head_rgb`, `left_wrist_rgb`, `right_wrist_rgb`, 각 480×640×3 uint8
(H×W). 실기 포맷의 손목 영상은 세로(640×480)라 변환기에서 회전한다(`rby1_format.adapt_image`).

## 실기 포맷(`rby1_format`)과의 차이

- 그리퍼 상태 정규화: 실기는 손가락 반 간격 / 0.045 m, 시뮬은 / 0.05 m(시뮬 손가락 범위). 변환기가 실기 기준으로 다시 계산한다.
- 몸통·베이스·단계 정보는 실기 16-D에 없어서 LeRobot 추가 feature로 넣는다(4단계).
