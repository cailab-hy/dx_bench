# DX bench — 진행 상황

> 주말 무인 실행(2026-10-16 ~). 단계별 상태는 아래 표, 결정은 `DECISIONS.md`, 사람이 할 일은 `HUMAN_TODO.md`.
> 마지막 갱신: 2026-10-17 19:05 (6단계 완료, 100k ACT 추가 실험 진행 중)

## 상태 요약

| 단계 | 상태 | 완료 기준 | 수치 | 산출물 |
|---|---|---|---|---|
| 0 RB-Y1 검증 | 완료 | — | `reports/rby1_check/REPORT.md` | `tools/rby1_check/` |
| 1 천 사전 검증 (8 h) | **완료** (8bd0fe8) | (a)(b)(c) 잡은 정점 ≤ 2 cm, 60 s 안정, RTF | 보조 잡기 480 Hz: 이불 1.0 / 소매 1.3 / 바닥 0.7 cm, NaN 없음. 마찰 잡기 실패 | `tools/cloth_spike/`, `config/cloth_params.yaml` |
| 2 RB-Y1 embodiment (8 h) | **완료** (2831401) | 회귀 테스트 + 침대가 보이는 머리 카메라 | — | `dxbench/embodiments/` |
| 3 침실 씬·task 3개 (12 h) | **완료** (BED-03 Easy oracle 8/10) | 빌드, 판정 pytest, BED-03 Easy oracle ≥ 8/10 | — | `dxbench/tasks/` |
| 4 원격조작·기록·변환 (10 h) | **완료** | 키보드·VR(scripted)·변환·검증·재생 끝까지 | oracle 데모 24개(34,891프레임) dx24·rby1 변환 bit-exact, 재생 검증 통과(과제 성공 재현) | `dxbench/teleop/`, `dxbench_policy/convert/`, `reports/stage4/REPORT.md` |
| 5 VLA 추론 (8 h) | **완료** | ACT로 BED-03 Easy 10회 play, 영상+지표 | ACT(20k) 10회 오류 0, 성공 0/10. SmolVLA 실데이터 왕복 OK. 100k ACT 추가 실험 대기 | `dxbench_policy/serve/`, `dxbench/policies/rby1_lerobot_remote.py`, `reports/stage5/REPORT.md` |
| 6 벤치마크 (6 h) | **완료** | 명령 하나로 평가 → 보고서 | BED-03 Easy: oracle 8/10, hold·replay·ACT 0/10 | `benchmarks/dx_bed_v0/`(RESULTS.md), `docs/BENCHMARK.md`, `reports/stage6/REPORT.md` |

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

### 6단계 (완료)
- 고정 장면 90 + 천 상태 230, 실험 YAML 5개, 실행기·집계·`RESULTS.md` 자동 생성, `docs/BENCHMARK.md`.
- 표본 결과(BED-03 Easy 10장면): oracle 8/10(0.90), hold 0/10, 개루프 재생 0/10(바구니 넘어짐 1), ACT 20k 0/10(전복 1).
- 100k step ACT(D-27)는 커밋 뒤 실행 중.

### 5단계 (완료)
- 서버(hold / replay / lerobot) ⇄ `rby1_lerobot_remote` 클라이언트. zero → replay(과제 성공 재현) → ACT → 폐루프 순서 모두 통과.
- **ACT 20k step**(17분) → BED-03 Easy 10회: 오류 0, mp4 30개, **성공 0/10**(전복 1). 학습 프레임 개루프 오차 = "자세 유지" 수준 → 학습 부족(D-27).
- SmolVLA: 실제 데모로 2 step 미세조정 → 서버 왕복 (50, 24) 청크 205 ms.
- BED-02 양팔 oracle을 GPU 여유 시간에 5번 고쳐 점수 0.3 → 0.55(접기는 놓으면 펴짐, 반 접기 도달 불가 → HUMAN_TODO 8).

### 4단계 (완료)
- **데모 24개**(학습 장면 21개, 끝까지 돈 실행의 96 % 성공), LeRobot v3 dx24(240 × 320) + rby1(480 × 640), 두 형식 모두 검증 통과.
- 실제 데모 0번 재생: 과제 성공 재현, TCP p99 9.8 mm. 접촉 중 짧은 관절 튐 때문에 재생 합격 기준을 프레임 비율로 바꿈(D-26).
- SmolVLA 실데이터 왕복(2 step 미세조정 → 서버): (50, 24) 청크, 205 ms.

#### 수집 기록
- 시뮬로 검증한 것: VR scripted operator 종단(6 mm), 기록 → 변환 → 검증(bit-exact), 재생 검증(통과). 정리: `reports/stage4/REPORT.md`.
- oracle 데모 수집(`tools/collect_demos.sh`, 학습 장면 30개, 6개 동시, 에피소드당 약 75분).
  - 지금까지 끝난 것은 모두 성공했다(종료 판정 수정 후).
- 수집 중 발견한 문제:
  - PNG 원본이 에피소드당 4 GB → JPEG로 바꿈(D-25).
  - GPU 메모리(49 GB)가 차면 PhysX GPU 커널 실패.
    - Kit 6개 × 7 GB에 다른 GPU 작업을 더하면 생긴다.
    - 영향받은 에피소드는 변환에서 뺀다(`tools/stage45.sh`).
    - 수집 중에는 다른 GPU 작업을 하지 않는다.
- 다음 순서: `tools/stage45.sh` 남은 부분 = 변환(dx24 240 × 320 + rby1) → 검증 → 재생 검증 → ACT 학습 → 10회 play.
  - 11:50 `tools/after_collect.sh`가 수집기 종료를 기다렸다가 `N=21 tools/stage45.sh`를 자동 실행(10002 제외분 보충).
  - ACT·재생 평가 상한을 1500 → 1800 step으로 올림(데모가 1340–1567 step, D-23 수정).

### 5–6단계 (코드 완료)
- 정책 서버·클라이언트·SmolVLA 왕복 확인(`reports/stage5/REPORT.md`).
- 벤치마크 실행기·집계·보고서: oracle 8/10, hold(zero) 0/10. replay·ACT는 데이터셋이 나온 뒤 실행.
