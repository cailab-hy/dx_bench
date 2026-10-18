# 6단계 — 벤치마크 `dx_bed_v0`

## 구성

| 부분 | 위치 |
|---|---|
| 고정 평가 장면 | `benchmarks/dx_bed_v0/scenes/` 90개(3 task × 3 난이도 × seed 0–9). 학습 장면(`data/scenes_train/`, seed 10000+)과 겹치지 않음 |
| 천 초기 상태 | `benchmarks/dx_bed_v0/states/` npz 230개(장면별·천별 "떨어뜨려 안착" 정점, 6 MB). 리셋마다 불러와 장면이 결정적 |
| 실험 설정 | `benchmarks/dx_bed_v0/experiments/*.yaml`(정책, 정책 인자, env 인자, task × 난이도, 정책 서버 명령, 동시 실행 수, step 상한) |
| 실행기 | `python -m dxbench.bench.run_benchmark <yaml>`: 장면마다 Kit 1개, 서버 자동 시작·종료, Kit 시작 segfault 3회 재시도, 오류난 장면만 다시 실행, 시작 시차 |
| 지표 | 성공률, 평균 점수, 단계별 완료율·첫 도달 시간, 전복(기울기 > 10°), task 실패(바구니 넘어짐 등), 오류, 시뮬·벽시계 시간 |
| 자동 보고서 | 실험마다 `outputs/bench/<name>/{REPORT.md, summary.json}`, 전체 표 `python -m dxbench.bench.make_results` → `benchmarks/dx_bed_v0/RESULTS.md` |
| 문서 | `docs/BENCHMARK.md`(과제·난이도·분할·인터페이스·지표·실행·설정 표·결과 해석) |

## 표본 결과 (BED-03 Easy, 평가 장면 10개)

| 실험 | 정책 | 성공 | 평균 점수 | 단계 완료 | 전복 | task 실패 | 오류 | 시뮬 s | 벽시계(분) |
|---|---|---|---|---|---|---|---|---|---|
| `oracle_bed03` | 특권 oracle(`rby1_ik`) | **8/10** | **0.90** | 옷1 100 %, 옷2 80 % | 0 | 0 | 0 | 114 | 85 |
| `hold_bed03` | 자세 유지(서버 hold) | 0/10 | 0.00 | 0 % | 0 | 0 | 0 | 20 | 10 |
| `replay_bed03` | 학습 데모 0번 개루프 재생 | 0/10 | 0.00 | 0 % | 0 | 1(바구니 넘어짐) | 0 | 115 | 28 |
| `act_bed03` | ACT 20k step(24 데모) | 0/10 | 0.00 | 0 % | 1 | 0 | 0 | 120 | 43 |
| `act_bed03_100k` | ACT 100k step(같은 데모) | 0/10 | 0.00 | 0 %(003에서 옷1이 40 s에 바구니 안 → 다시 끌어냄) | 0 | 0 | 0 | 120 | 36 |

- **oracle 8/10:** 장면 생성·판정·물리가 과제를 풀 수 있는 상태라는 상한 기준. 실패 2개는 두 번째 옷.
- **hold·replay 0/10:** 장면을 보지 않는 정책으로는 풀리지 않는다. 장면마다 옷·바구니 위치가 달라 데모 하나를 그대로 재생해도 소용없다.
- **ACT 0/10:** 첫 학습 정책. 배관은 정상(학습 프레임 개루프 확인)이지만 학습량이 부족하다(`reports/stage5/REPORT.md`).
  100k step(`act_bed03_100k`, D-27)도 0/10이지만 장면 003에서 옷 하나를 바구니에 넣었다가 다시 끌어냈다 → 학습량에 따라 행동이 나아진다.

## 결정·제한

- 학습 정책·재생 평가는 1800 step(120 s)에서 자름(D-23). 과제 정의의 180 s보다 짧다.
- 재생 검증 합격 기준은 프레임 비율(D-26).
- 보조 잡기(이 보고서의 결과는 손끝 5 cm 안 천 정점 고정)는 실기와 다르다 → 시뮬 성공이 실기 성공을 뜻하지 않는다(HUMAN_TODO 1).
  2026-10-18부터 반경 2.5 cm(D-U11)·수건 영률 1e5(D-28): 같은 조건 oracle 8/10(`oracle_bed03`), 이전 5 cm 결과는 `oracle_bed03_r50`.
- BED-01/02는 장면·판정·초기 상태까지 준비됐고 oracle이 없다(BED-02 양팔 oracle 0.55, BED-01 미착수, HUMAN_TODO 8).
  그래서 표본 결과는 BED-03 Easy뿐이다. 설정 YAML의 `tasks:`에 넣으면 같은 실행기로 돈다.

## 재현

```bash
A=~/miniforge3/envs/dx_arena/bin/python
for e in oracle_bed03 hold_bed03 replay_bed03 act_bed03 act_bed03_100k; do
  $A -m dxbench.bench.run_benchmark benchmarks/dx_bed_v0/experiments/$e.yaml
done
$A -m dxbench.bench.make_results     # -> benchmarks/dx_bed_v0/RESULTS.md
```
