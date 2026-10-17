# DX bench — 진행 상황

> 주말 무인 실행(2026-10-16 ~). 단계별 상태는 아래 표, 결정은 `DECISIONS.md`, 사람이 할 일은 `HUMAN_TODO.md`.
> 마지막 갱신: 2026-10-17 18:10 (3단계 완료·커밋, 4단계 데모 수집 중)

## 상태 요약

| 단계 | 상태 | 완료 기준 | 수치 | 산출물 |
|---|---|---|---|---|
| 0 RB-Y1 검증 | 완료 | — | `reports/rby1_check/REPORT.md` | `tools/rby1_check/` |
| 1 천 사전 검증 (8 h) | **완료** (8bd0fe8) | (a)(b)(c) 잡은 정점 ≤ 2 cm, 60 s 안정, RTF | 보조 잡기 480 Hz: 이불 1.0 / 소매 1.3 / 바닥 0.7 cm, NaN 없음. 마찰 잡기 실패 | `tools/cloth_spike/`, `config/cloth_params.yaml` |
| 2 RB-Y1 embodiment (8 h) | **완료** (2831401) | 회귀 테스트 + 침대가 보이는 머리 카메라 | — | `dxbench/embodiments/` |
| 3 침실 씬·task 3개 (12 h) | **완료** (BED-03 Easy oracle 8/10) | 빌드, 판정 pytest, BED-03 Easy oracle ≥ 8/10 | — | `dxbench/tasks/` |
| 4 원격조작·기록·변환 (10 h) | 코드 작성 | 키보드·VR(scripted)·변환·검증·재생 끝까지 | — | `dxbench/teleop/`, `dxbench_policy/convert/` |
| 5 VLA 추론 (8 h) | 코드 작성 | ACT로 BED-03 Easy 10회 play, 영상+지표 | — | `dxbench_policy/`, `dxbench/policies/` |
| 6 벤치마크 (6 h) | 코드 작성 | 명령 하나로 평가 → 보고서 | — | `benchmarks/dx_bed_v0/`, `docs/BENCHMARK.md` |

## 전체 계획

공통: 시뮬레이션·기록은 `dx_arena`(lerobot import 금지), 변환·학습·정책 서버는 `dx_policy`. Arena 원본은 고치지 않고
`dxbench` 패키지(`pip install --no-deps -e .`)의 등록 API로 확장. 단계마다 테스트 → 이 파일 갱신 → `stage N: ...` 커밋(push 없음).

### 1단계 천 사전 검증
1. `tools/cloth_spike/`에 Isaac Lab 직접 씬(Arena 없이): 고정 베이스 RB-Y1(tuned) + DX `Bed.usda`(이불·베개) /
   `TShirt.usda`(책상 위, 바닥 복제) 참조.
2. 팔은 기구학 IK로 미리 푼 관절 궤적으로 이동(0단계 T3d 방식). 테스트 (a) 이불 모서리 0.5 m 끌기,
   (b) 소매 접어 넘기기, (c) 바닥 옷 0.3 m 들기 × 잡기 ① 마찰 ② 보조(닫힌 손가락 안 반경 r 정점을 TCP에 고정).
3. 측정: 잡은 정점-TCP 거리, 60 s NaN/폭주, RTF(카메라 3대 on/off), 천 자기충돌 on.
4. 스윕: dt 1/240·1/480, 해상도, 두께·마찰, solver 반복. 결과 → `config/cloth_params.yaml`, `reports/cloth_spike/REPORT.md`.

### 2단계 (완료)
- 회귀 6종 모두 통과: idle, crouch, drive, gripper, floor, 침실 카메라(`reports/rby1_check/t5/REPORT.md`).
- 바닥 집기 중 전복의 원인은 몸통 프리셋의 관절별 등속 램프였다. torso_1·2·3 동기 보간으로 고쳤다(D-15).
  deep 자세에서 손끝 1.3 cm, 기울기 0°.
- 그리퍼 시험 기준을 18 → 12 cm로 바꿨다(오른 손목 작업공간 한계, D-17).

### 3단계 (완료, 시간 상자 초과)
- **BED-03 Easy oracle 8/10**(합격 기준 ≥ 8/10), 평균 점수 0.90, 전복 0 → `reports/stage3/REPORT.md`, `reports/stage3/oracle_bed03_easy.md`.
- 자산, 장면 생성기(평가 90 + 학습 90), 천 초기 상태(평가 90개 모두), predicate pytest 14, oracle, 실행기.
- 실제 시뮬에서 BED-03 oracle을 돌리며 12가지 문제를 고쳤다(보고서 표).
  - 시뮬 쪽:
    - DX 바닥 Cube → Plane(D-22).
    - 바퀴·손 ↔ 천 필터(D-18/19).
    - 바구니 60 × 44 cm(D-20).
    - 옷 간격.
  - 정책 쪽: A*, 축 중심 회전, 자세별 도달, 홈 복귀(D-24).
  - 평가 버그: 성공 종료 후 리셋된 장면을 다시 평가하던 문제를 종료 항으로 판정하도록 고침.
- BED-02 Easy oracle(오른팔 단독)은 접기 실패(점수 0.3). BED-01 oracle 미착수 → HUMAN_TODO 8.

### 4–6단계 (코드 완료, 시뮬 검증 일부)
- 실제 시뮬 데이터로 확인한 것:
  - 원본 기록(카메라 3대 PNG + trajectory) → LeRobot v3 변환(rby1 / dx24) → 검증: state·action bit-exact, 영상 MSE 4.4.
  - **재생 검증:** 정책 서버 replay 백엔드 → `rby1_lerobot_remote` → `rby1_joint`. 베이스 3.2 cm, 팔 관절 최대 0.038 rad, TCP 최대 3.2 cm(평균 0.5 cm), 통과(`reports/stage4/replay_check_pipeline.json`).
  - **VR scripted operator 시뮬 종단 시험:** WebSocket → clutch → IK → 기록. 손 이동 오차 6 mm(`reports/stage4/vr_scripted_operator.json`).
  - ACT 학습 → 서버 → 왕복(27 ms). SmolVLA 로드·미세조정 2 step → 서버 왕복(50스텝 청크 200 ms).
- 남은 일(BED-03 oracle 합격 후): 데모 20개 수집 → 변환 → ACT 학습 → 10회 play → 벤치마크(oracle / hold / replay / ACT).
