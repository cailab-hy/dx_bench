# DX bench — 진행 상황

> 주말 무인 실행(2026-10-16 ~). 단계별 상태는 아래 표, 결정은 `DECISIONS.md`, 사람이 할 일은 `HUMAN_TODO.md`.
> 마지막 갱신: 2026-10-17 00:30 (1단계 보고서, 2단계 바닥 자세 진단, 3–6단계 코드 병행)

## 상태 요약

| 단계 | 상태 | 완료 기준 | 수치 | 산출물 |
|---|---|---|---|---|
| 0 RB-Y1 검증 | 완료 | — | `reports/rby1_check/REPORT.md` | `tools/rby1_check/` |
| 1 천 사전 검증 (8 h) | 완료(스윕 마무리 중) | (a)(b)(c) 잡은 정점 ≤ 2 cm, 60 s 안정, RTF | 보조 잡기 480 Hz: 이불 1.0 / 소매 1.3 / 바닥 0.7 cm, NaN 없음. 마찰 잡기 실패 | `tools/cloth_spike/`, `config/cloth_params.yaml` |
| 2 RB-Y1 embodiment (8 h) | 진행 중 | 회귀 테스트 + 침대가 보이는 머리 카메라 | — | `dxbench/embodiments/` |
| 3 침실 씬·task 3개 (12 h) | 진행 중 | 빌드, 판정 pytest, BED-03 Easy oracle ≥ 8/10 | — | `dxbench/tasks/` |
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

### 2단계 embodiment
`dxbench/assets/rby1/rby1_dx.usda`(mimic, 캐스터 μ0, 충돌체 보강, tool xform) → `rby1_joint`(정책: 팔14+그리퍼2+몸통6+베이스2)
/ `rby1_ik`(원격조작) embodiment, 카메라 3대, 중력보상 ActionTerm, 15 Hz(decimation k=16, dt 1/240).
회귀: idle/crouch/gripper/drive + 바닥 집기 IK 자세.

### 3단계 task
CareRoom 전역 asset 등록, 바구니·옷 변형·방해 물체 USD 생성, 난이도별 초기 상태 생성기(천 정점 `.npz`),
단계 predicate, 부분 점수, 언어 지시, pytest(정답/오답 상태), oracle 정책(BED-03 Easy ≥ 8/10).

### 4단계 원격조작·데이터
키보드(`collect_keyboard.py` 키맵) → `rby1_ik`, VR(lightwheel `scripts/vr/*` 이식, scripted operator 테스트),
원본 기록(PNG + `trajectory.npz` + `initial_state.json`), `dx_policy`에서 LeRobot v3 변환(`rby1_format` 16-D + 추가 feature, 15 fps),
재생 검증, oracle 데모 task당 20개.

### 5단계 추론
`dx_policy` websocket 정책 서버(Arena remote policy 규약), `dx_arena` 클라이언트 `rby1_lerobot_remote`, play 스크립트,
zero → replay → ACT 짧은 학습 → closed-loop, SmolVLA 로드·왕복.

### 6단계 벤치마크
`benchmarks/dx_bed_v0/`(task×난이도×10 장면), 일괄 실행기, 지표 집계, 자동 보고서, `docs/BENCHMARK.md`.

## 단계별 기록

### 1단계 (2026-10-16 ~ 17)
- 보고서 `reports/cloth_spike/REPORT.md`, 파라미터 `config/cloth_params.yaml`, 표 `python3 tools/cloth_spike/summarize.py --tag m`.
- 마찰 잡기는 평평한 천을 못 집음(drift 42–50 cm) → 보조 잡기(속도 핀, D-4/D-5)로 결정, sim-real 차이로 기록.
- 물리 480 Hz가 평가 기본(이불 drift 1.0 cm). 360 Hz는 셔츠·수건은 통과, 이불 2.07 cm → D-11.
- RTF(카메라 없음) 480 Hz 0.36–0.38, 360 Hz 0.47–0.52. 카메라 3대 포함 측정은 다른 작업과 GPU를 같이 써서 낮게 나옴(재측정 예정).

### 2단계
- 완료: `rby1_dx.usda`, `rby1_joint`/`rby1_ik` embodiment, 회귀 idle·crouch·drive·침실 카메라 통과.
- 남음: 그리퍼 들기(arm_5 한계 → 큐브 높이 조정), 바닥 집기 자세.
  - 기구학 탐색: 바닥에 닿는 자세는 있으나 COM 여유가 있는 자세는 몸통 중력 토크가 effort의 88–100 %.
  - deep 프리셋 동역학 시험은 15 cm 앞에서 손끝 0.4 cm까지 닿은 뒤 52° 넘어짐 → 단계별 기울기 진단 중.

### 3단계
- CareRoom 전역 자산, 자산 USD(바구니·셔츠 4종·인형·책), 장면 생성기, predicate pytest 14개 통과.
- 장면: 평가 90개(3 task × 3 난이도 × 10, `benchmarks/dx_bed_v0/scenes`), 학습용 BED-01/02/03 easy 각 30개(`data/scenes_train`).
- 셔츠 변형 USD 버그(customLayerData `color`에 타입 없음 → 레이어 파싱 실패) 수정.
- 천 초기 상태 생성 진행 중(`tools/make_states.sh`), BED-03 oracle 첫 실행 중.

### 4–6단계 (코드 먼저 작성, 시뮬 데이터가 나오면 실행)
- `dx_policy` 설치: lerobot[training,smolvla] 0.6.1 등(`VERSIONS.md`).
- `dxbench_policy` 패키지:
  - 변환기(rby1 / dx24 형식): 합성 데이터 pytest 통과(bit-exact, 영상 MSE).
  - 정책 서버(hold / replay / lerobot): ACT, SmolVLA 왕복 확인(ACT 27 ms, SmolVLA 50스텝 청크 200 ms).
- `dx_arena` 쪽:
  - 클라이언트 `rby1_lerobot_remote`.
  - VR 이식: lightwheel 테스트 32개 통과.
  - 키보드·VR 장치와 scripted VR operator: 시뮬 없는 폐루프 pytest 통과.
  - 원격조작 실행기 `teleop/run_teleop.py`.
  - 벤치마크 실행기 `dxbench.bench.run_benchmark`.
