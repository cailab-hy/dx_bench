# 4단계 — 원격조작·원본 기록·LeRobot 변환·데모

## 구성

| 부분 | 위치 | env |
|---|---|---|
| 키보드 → `rby1_ik` | `dxbench/teleop/devices.py` `KeyboardTeleop` + `run_teleop.py --teleop_device keyboard` (GUI: `--visualizer kit`) | dx_arena |
| VR(Quest WebXR) → `rby1_ik` | `dxbench/teleop/vr/`(lightwheel `vr_server` / `vr_teleop` / `vr_video` 이식) + `devices.VRTeleopDevice` | dx_arena |
| scripted VR operator | `dxbench/teleop/scripted_vr.py`: 실제 WebSocket으로 헤드셋 보고를 보냄 | dx_arena |
| 원본 기록 | `dxbench/teleop/recorder.py`(JPEG q95 3대 + `trajectory.npz` + `initial_state.json` + `episode.json`, 성공 에피소드만 저장) | dx_arena |
| LeRobot v3 변환·검증 | `dxbench_policy/convert/raw_to_lerobot.py`, `validate.py`(rby1 16-D / dx24, D-12) | dx_policy |
| 재생 검증 | `tools/replay_check.sh`: 정책 서버 replay → `rby1_lerobot_remote` → `rby1_joint` → 원본과 비교 | 둘 다 |
| oracle 데모 | `tools/collect_demos.sh`(학습 장면 30개, 성공만 저장) | dx_arena |

### 키맵

- **키보드**(lightwheel `collect_keyboard.py` 키맵을 따름):
  - 팔 이동·회전: W/S, A/D, Q/E로 이동하고 Z/X, T/G, C/V로 회전한다.
  - TAB 팔 전환, K 그리퍼, L 정지.
  - 이번에 더한 키: 화살표 = 베이스, U/J = 몸통 프리셋, H = 팔 홈.
  - 녹화 제어: B 시작, ENTER 저장, BACKSPACE 폐기, R 리셋, P 일시정지, ESC 종료.
- **VR:**
  - 기본 조작: grip = clutch, trigger = 그리퍼.
  - 녹화 제어: A 시작, B(길게) 폐기, X 일시정지, Y(길게) 방향 맞춤.
  - 이번에 더한 조작: 오른 스틱 = 베이스, 왼 스틱 위/아래 = 몸통 프리셋.

## 검증

| 항목 | 결과 |
|---|---|
| 키보드 키맵 | pytest(`dxbench/tests/test_teleop_devices.py`). GUI 실사용은 사람 몫(HUMAN_TODO 4) |
| VR 이식 | lightwheel 테스트 32개 통과(`dxbench/teleop/vr/test_*.py`) |
| VR 폐루프(시뮬 없음) | scripted operator → WebSocket → clutch → 델타 → 가짜 TCP 적분: 오차 < 1 cm (pytest) |
| **VR 시뮬 종단** | scripted operator → 실제 env(`rby1_ik`) → 기록 저장. 손 이동 (10.0, −4.4, 9.8) cm vs 목표 (10, −5, 10) cm, 오차 6 mm(`vr_scripted_operator.json`) |
| 변환기 | pytest: 합성 원본 → rby1·dx24 → 검증(bit-exact, 멱등) |
| 실제 시뮬 기록 변환 | 45프레임 기록 → dx24·rby1, state·action bit-exact, 영상 MSE 4.4 |
| 재생 검증(짧은 기록) | 베이스 xy 3.2 cm, 팔 관절 최대 0.038 rad(평균 0.004), TCP 최대 3.2 cm(평균 0.5 cm), 통과(`replay_check_pipeline.json`) |
| **재생 검증(실제 데모 0번, 1361프레임)** | 과제 **성공**(점수 1.0, 두 벌 모두 바구니). 베이스 최종 2.9 mm, TCP 평균 1.8 mm · p99 9.8 mm, 팔 관절 0.05 rad 안 프레임 98.0 % → 통과(D-26). 엄격 기준(모든 프레임)은 불합격: 옷을 들고 갈 때·바구니에 누를 때 2–5프레임 관절 튐 최대 0.25 rad(`replay_check_ep0.json`) |
| **oracle 데모** | 24개(목표 20) → LeRobot v3 dx24·rby1 변환, 검증 통과(아래) |

## 데모 데이터셋

| 항목 | 값 |
|---|---|
| 정책 | `Bed03OraclePolicy`(특권 정보, `rby1_ik`), 학습 장면 `data/scenes_train/bed03_easy_train_100xx`(평가 장면과 다른 seed) |
| 실행 | 끝까지 돈 26회 중 성공 25회(96 %). 실패 1회(10001, 점수 0.5)는 저장 안 됨. GPU 커널 실패로 중간에 죽은 장면 2개(10015, 10017)는 제외 |
| 제외 | 1개(10002): 저장은 됐지만 로그에 PhysX "fail to launch"(수집 중 GPU 메모리 부족) → 변환에서 뺌(`excluded_episodes.txt`) |
| 최종 | **24 에피소드 · 34,891프레임**(15 Hz, 에피소드당 1339–1851, 평균 1454프레임 ≈ 97 s), 고유 장면 21개. 10010·10011·10026은 두 수집기가 동시에 같은 장면을 돌려 두 번씩 들어감(천 솔버가 비결정적이라 궤적은 서로 다름) |
| 원본 | `data/raw/bed03_oracle` 5.3 GB(JPEG q95, D-25; 이전 PNG 21,597프레임 재압축) |
| dx24 | `data/lerobot/bed03_oracle_dx24` 200 MB: state 24, action 24(= `rby1_joint`), 카메라 3대 240 × 320(½ 축소), `dx.*` 열(stage_id, intervention, base_pose, tcp) |
| rby1 | `data/lerobot/bed03_oracle_rby1` 387 MB: 16-D 실기 형식(팔 14 + 그리퍼 2), 카메라 480 × 640 |
| 검증 | 두 형식 모두 24/24: state·action bit-exact, 타임스탬프 OK, 영상 MSE 최대 16.4(dx24) / 8.3(rby1), 기준 50 |

주의: oracle 데모는 보조 잡기(D-4, 손끝 5 cm 안 정점 고정)를 쓴다. 학습된 정책이 시뮬에서 성공해도 실기에서는
마찰 잡기가 필요하다(HUMAN_TODO 1).

## 재현

```bash
N=21 P=3 tools/collect_demos.sh data/scenes_train/bed03_easy_train_*.json   # 수집(성공만 저장)
N=21 P=3 STEPS=20000 tools/stage45.sh                                          # 변환 → 검증 → 재생 검증 → ACT → play
```
