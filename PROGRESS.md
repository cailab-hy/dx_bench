# DX bench — 진행 상황

> 주말 무인 실행(2026-10-16 ~). 단계별 상태는 아래 표, 결정은 `DECISIONS.md`, 사람이 할 일은 `HUMAN_TODO.md`.
> 마지막 갱신: 2026-10-20 02:45 (사용자 결정 D-U8–D-U11·D-32 반영, v2 ACT·SmolVLA 평가 완료, push 없음)

## 월요일 아침 요약

**1–6단계 모두 완료 기준을 채웠고 단계마다 로컬 커밋했다(push 안 함).** 핵심 수치는 BED-03 Easy(평가 장면 10개) 기준.

### 되는 것

| 영역 | 상태 |
|---|---|
| 천 시뮬(1단계) | 보조 잡기 480 Hz로 이불·소매·바닥 옷 잡기 오차 ≤ 1.3 cm, 60 s 안정. 마찰 잡기는 실패(D-4/5) |
| RB-Y1(2단계) | `rby1_joint`(24-D)·`rby1_ik`(17-D), 회귀 6종 통과(바닥 집기·그리퍼·주행·카메라) |
| 침실 task(3단계) | BED-01/02/03 × 3 난이도 장면 90 + 학습 장면 90, 판정 pytest, 천 초기 상태. **BED-03 Easy oracle 8/10(0.90)** |
| 데이터(4단계) | 키보드·VR(scripted operator 시뮬 종단 6 mm) 원격조작, 성공만 저장하는 원본 기록, LeRobot v3 dx24·rby1 변환 bit-exact. **oracle 데모 24개(34,891프레임)**, 실제 데모 재생 시 과제 성공 재현 |
| 정책(5단계) | dx_policy 서버(hold/replay/lerobot) ⇄ dx_arena 클라이언트. ACT 학습 → **10회 play 오류 0**(mp4 + 지표). SmolVLA 실데이터 미세조정 → 서버 왕복 |
| 벤치마크(6단계) | YAML 하나로 평가 → 실험별 REPORT + `benchmarks/dx_bed_v0/RESULTS.md`. oracle 8/10, hold·개루프 재생·ACT 20k·ACT 100k 모두 0/10. ACT 100k는 장면 하나에서 옷을 바구니에 넣었다가 다시 끌어냄 |

안 된 것: BED-02 oracle 0.55(접기 0/3, 놓으면 펴짐·반 접기 도달 불가), BED-01 oracle 없음, 학습 정책 성공 0.

### 결정이 필요한 것 (3개)

1. **보조 잡기 유지 여부(D-4/D-5):** 손끝 5 cm 안 천 정점을 손에 고정하는 방식이라 시뮬 성공 ≠ 실기 성공. v0 벤치마크는
   이대로 갈지, 마찰 잡기(그리퍼 패드·천 해상도 조정)에 시간을 쓸지.
2. **BED-01/02를 어떻게 채울지(HUMAN_TODO 8):** (a) 수건 전용 천 재질(굽힘 강성↓) 허용 — BED-03 천과 공용 파라미터라
   천 상태 재생성이 필요, 그 뒤 양팔 oracle 계속, 또는 (b) 사람 원격조작 데모(키보드·VR, 4단계 도구 준비됨).
3. **다음 학습 방향(HUMAN_TODO 3·5):** oracle 데모 확대(GPU 하나로 동시 6개, 에피소드당 약 75분 → 100개면 약 21시간) +
   긴 ACT, 아니면 SmolVLA / π0 미세조정(π0 가중치는 5 GB 초과라 사람이 받아야 함).

### 재현 명령

```bash
cd ~/dx_bench
A=~/miniforge3/envs/dx_arena/bin/python; P=~/miniforge3/envs/dx_policy/bin/python
$A -m pytest -q dxbench/tests dxbench/dxbench/teleop/vr        # 49 통과 (시뮬 없음)
$P -m pytest -q dxbench_policy/tests                            # 2 통과
# oracle 한 에피소드 + 영상 (약 1.5시간)
$A dxbench/dxbench/scripts/run_episode.py --policy dxbench.policies.oracle_bed03.Bed03OraclePolicy --out outputs/demo_000 --video \
   --external_environment_class_path dxbench.tasks.bed_tasks:DxBed03Environment dx_bed03 --scene benchmarks/dx_bed_v0/scenes/bed03_easy_000.json
# 벤치마크 실험 (oracle / hold / replay / act / act_100k) → 결과표
$A -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/act_bed03_100k.yaml
$A -m dxbench.bench.make_results                                # -> benchmarks/dx_bed_v0/RESULTS.md
# 데모 수집 → 변환 → 검증 → 재생 검증 → ACT 학습 → 10회 play
N=21 P=3 STEPS=20000 tools/stage45.sh
```

자세한 것: `README.md`, `docs/PIPELINE.md`, `docs/BENCHMARK.md`, 단계별 `reports/stage*/REPORT.md`. 시뮬 작업 정지는
`tools/kill_sims.sh`, 정책 서버 정지는 `tools/kill_servers.sh`(같은 명령줄에 `pkill` 패턴을 쓰면 자기 셸이 죽음).

## 주말 이후 (2026-10-18 ~, 사용자 결정 반영)

| 결정 | 한 일 | 상태 |
|---|---|---|
| D-U11 보조 잡기 반경 5 → 2.5 cm | Isaac Sim 천 레포 조사(`docs/notes/cloth_grasp_survey.md`), oracle 잡기 확인·재접근·건너뛰기(D-29) | **BED-03 Easy oracle 8/10**(새 조건) |
| D-U8 BED-02 수건 재질 | 수건 영률 1e5(D-28, `reports/towel_fold/REPORT.md`), 수건 장면 110개 천 상태 재생성, 양손 반 접기(D-30), 쌓는 자리 이동·양손 쌓기(D-32, 사용자 선택 ③) | oracle 최고 0.7에서 **중단**(D-33) → 사람 데모 제안 |
| D-U9 BED-01 | 사람 원격조작으로 나중에 | — |
| 실제 물체 교체 + 1단계 재확인 (10-21) | 바구니·수건 10색·반팔티 3색을 실제 제품으로 교체(D-36), 장면 190개·평가 천 상태 재생성, 이불에 지지 정점(D-37), BED-03 oracle 잡는 거리 6 cm 당김 | **BED-03 쉬움 oracle 7/10**(0.85). 실패: 가구 옆 수건 도달 2, 작은 바구니에 수건 걸침 1. 학습 장면 상태 99개 남음 → `reports/pad_grasp/REPORT.md` 시험 4 |
| D-U12 잡기 조건 엄격화 (10-20) | LeIsaac·Lightwheel·Isaac Lab Franka 천 예제 조사·영상, Franka에서 패드 상자 보조 잡기 재현, RB-Y1 BED-02 oracle로 squeeze 0.7 vs 0 비교 → `reports/pad_grasp/REPORT.md` | **기본 잡기 = pad, squeeze 0**(D-34). BED-03 oracle 재확인과 데모 재수집 필요 |
| 2단계 과제 정의 마무리 (10-21, 진행 중) | 판정 규칙 수정(D-41: BED-02 대칭 짝 접기 판정, BED-01 덮기 영역·베개·바닥 실패), 보통·어려움 옷 초기 상태 = 들었다 떨어뜨린 더미, **천 초기 상태가 에피소드에 안 불려 오던 버그 수정**(D-42, 3단계부터), BED-02 어려움 더미 재배치(수건이 책상 밖으로 떨어짐), 장면 다양화(지시문·블라인드·조명·시작 위치, D-42) | 판정 점검 시작 점수 0(BED-02 어려움 재점검 중). BED-03 쉬움 oracle 재확인 실행 중 |
| D-U10 데모 확대 + ACT·SmolVLA 동시 | 학습 장면 100개, 새 조건 oracle 데모 92개(`data/raw/bed03_oracle_v2`) → ACT 100k + SmolVLA 20k 동시 학습 → 평가 | **ACT 0/10, SmolVLA 0/10**(학습 장면에서도 0/3). 팔·몸통은 v1보다 2–4배 정확해졌지만 베이스 이동을 못 배움 → `reports/stage_v2/REPORT.md` |

## 상태 요약

| 단계 | 상태 | 완료 기준 | 수치 | 산출물 |
|---|---|---|---|---|
| 0 RB-Y1 검증 | 완료 | — | `reports/rby1_check/REPORT.md` | `tools/rby1_check/` |
| 1 천 사전 검증 (8 h) | **완료** (8bd0fe8) | (a)(b)(c) 잡은 정점 ≤ 2 cm, 60 s 안정, RTF | 보조 잡기 480 Hz: 이불 1.0 / 소매 1.3 / 바닥 0.7 cm, NaN 없음. 마찰 잡기 실패 | `tools/cloth_spike/`, `config/cloth_params.yaml` |
| 2 RB-Y1 embodiment (8 h) | **완료** (2831401) | 회귀 테스트 + 침대가 보이는 머리 카메라 | — | `dxbench/embodiments/` |
| 3 침실 씬·task 3개 (12 h) | **완료** (BED-03 Easy oracle 8/10) | 빌드, 판정 pytest, BED-03 Easy oracle ≥ 8/10 | — | `dxbench/tasks/` |
| 4 원격조작·기록·변환 (10 h) | **완료** | 키보드·VR(scripted)·변환·검증·재생 끝까지 | oracle 데모 24개(34,891프레임) dx24·rby1 변환 bit-exact, 재생 검증 통과(과제 성공 재현) | `dxbench/teleop/`, `dxbench_policy/convert/`, `reports/stage4/REPORT.md` |
| 5 VLA 추론 (8 h) | **완료** | ACT로 BED-03 Easy 10회 play, 영상+지표 | ACT 20k·100k 각 10회 오류 0, 성공 0/10(100k는 옷 하나 넣었다 빼냄). SmolVLA 실데이터 왕복 OK | `dxbench_policy/serve/`, `dxbench/policies/rby1_lerobot_remote.py`, `reports/stage5/REPORT.md` |
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
- 100k step ACT(D-27, 84분 학습): 0/10. 장면 003에서 옷을 바구니에 넣었다가(40 s) 다시 끌어냄.

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

### 5–6단계 (코드 완료 시점 기록, 2026-10-17 오전 — 결과는 위 5·6단계 절)
- 정책 서버·클라이언트·SmolVLA 왕복 확인(`reports/stage5/REPORT.md`).
- 벤치마크 실행기·집계·보고서: oracle 8/10, hold(zero) 0/10. replay·ACT는 데이터셋 뒤에 실행했다(위 6단계 절).

## git log --oneline

```
b1b2088 final: ACT 100k results, Monday-morning summary in PROGRESS.md
3160445 stage 6: dx_bed_v0 benchmark - fixed scenes, experiment configs, metrics, auto report, sample results
3aba493 stage 5: policy server + rby1_lerobot_remote, ACT trained on 24 oracle demos, 10 closed-loop plays on BED-03 Easy
4b833da stage 4: teleop (keyboard, VR, scripted VR), raw recorder, LeRobot v3 conversion, 24 BED-03 oracle demos
975e946 stage 3: bedroom tasks BED-01/02/03 - scenes, cloth states, predicates, oracle; BED-03 Easy oracle 8/10
2831401 stage 2: dxbench package + RB-Y1 embodiments (rby1_joint 24-D / rby1_ik 17-D), regression passes
8bd0fe8 stage 1: cloth spike - assisted grasp (velocity pins) at 480 Hz passes duvet/sleeve/floor, friction grasp fails
761b82b stage 0: RB-Y1 USD validation on Isaac Sim 6.1 / Isaac Lab 3.0
```
