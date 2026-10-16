# 2단계 — RB-Y1 embodiment 회귀 (t5)

Arena 외부 등록 embodiment(`rby1_joint` / `rby1_ik`)로 0단계 점검을 다시 돌린 결과다. `rby1_dx.usda`, 게인 tuned, 물리 240 Hz, 제어 15 Hz.
실행 방법: `tools/rby1_check/run_t5.sh`. 특정 시험만: `ONLY="floor gripper" tools/rby1_check/run_t5.sh`.

| 시험 | embodiment | 결과 | 수치 |
|---|---|---|---|
| idle (5 s 유지) | rby1_joint | 통과 | 관절 오차 최대 0.005 rad, 베이스 이동 0.01 mm, 기울기 0° |
| crouch (몸통 60/−90/60°) | rby1_joint | 통과 | 관절 오차 0.009 rad, 최대 토크 59 %(torso_3), 손끝 z 0.74 m |
| drive (0.3 m/s 3 s, 0.5 rad/s 3 s) | rby1_joint | 통과 | 직진 오차 1.1 %, 회전 오차 7.7 % (침실 240 Hz 1.1 % / 9.5 %, 침실 rby1_ik 480 Hz 1.4 % / 10.4 %) |
| gripper (5 cm 큐브 집어 들기) | rby1_ik | 통과 | 들어올림 14.4 cm, 미끄럼 0.2 mm. 기준 완화(18 → 12 cm), D-17 |
| floor (바닥 집기 자세) | rby1_ik | 통과 | deep 프리셋 + 오른팔 IK: 베이스 앞 15 / 20 cm에서 손끝 1.3 / 1.4 cm, 기울기 0°, COM 여유 9.7 cm |
| bedroom_camera | rby1_joint | 통과 | 머리 카메라에 침대가 보임(침대 중심 픽셀 (142, 321)), `bedroom_head_rgb.png` |
| torso_ramp (진단) | rby1_joint | – | 관절공간 동기 램프로 deep: 기울기 0°, 최대 토크 82 %(torso_3) |

## 이번 단계에서 고친 문제

1. **바닥 집기 중 전복** (D-15).
   - 원인: `TorsoPresetAction`이 torso_1·2·3를 관절마다 따로 등속(0.6 rad/s)으로 옮겨서, 중간에 (90, −90, 90)° 자세를 지났다.
   - 이 자세는 COM이 앞바퀴 축 위에 있고(여유 0.8 cm) 몸통 토크 118 %가 필요해 로봇이 넘어졌다(52°).
   - 수정: 세 관절을 직선 보간하고 코사인 시간 프로파일을 쓴다. 바닥 집기가 통과한다.
2. **그리퍼 시험** (D-17).
   - 0.95 m에서 18 cm를 들면 오른 손목 arm_5 상한에 걸리고, 0.80 m는 IK가 닿지 않는다.
   - 큐브를 0.90 m에 두고 12 cm 기준으로 바꿨다. 잡기 자체는 처음부터 양호했다.
3. 평지 배경을 `UsdGeom.Plane`으로 바꿨다(D-9). 바퀴 바깥 각도 루프를 넣었다(D-8).

## 탐색 기록

- `floor_posture_search.py`: 기구학 탐색(몸통 격자 × IK), 400개 후보 → `floor_posture.json`.
  - 바닥에 닿고 COM 여유 ≥ 5 cm인 자세는 정적 몸통 토크가 88–100 %라 80 % 기준을 넘었다.
  - 실제 동역학(중력보상 + PD)에서 deep 정지 토크는 82 %였다.
