# 5단계 — 정책 서버·클라이언트, ACT / SmolVLA

## 구성

```
dx_arena (Isaac Sim, lerobot 없음)                      dx_policy (lerobot 0.6.1)
run_episode.py / run_benchmark                           python -m dxbench_policy.serve.server
  └ Rby1LerobotRemotePolicy ("rby1_lerobot_remote")  ⇄   backend: hold | replay | lerobot(ACT, SmolVLA, ...)
    (Arena RemoteChunkReplayPolicy + openpi msgpack)      연결마다 세션(재생 커서 등), 모델은 공유
```

- **프로토콜:** Arena `remote_policy_base` / openpi_client와 같다.
  - 연결하면 메타데이터를 보낸다.
  - 요청: `observation.state`, `observation.images.{front,right,left}`, `task`. `{"reset": true}`는 에피소드 시작.
  - 응답: `{"actions": (H, D)}`.
- **dx24 형식:** 행동 24 = `rby1_joint`(팔 14 + 그리퍼 2 + 몸통 6 + 베이스 2), 상태 24(D-12). 16-D 실기 형식 서버는 클라이언트가 몸통·베이스를 채워 24로 만든다.
- **영상 크기:** 학습 데이터가 줄인 영상(240 × 320)이면 서버가 들어오는 영상을 같은 크기로 맞춘다.

## 검증

| 순서 | 결과 |
|---|---|
| zero(hold) | 서버 왕복 OK, (8, 24) 청크, 12 ms |
| replay | 데이터셋 0번 에피소드를 서버로 재생 → 시뮬 `rby1_joint`. 원본과 비교: 베이스 3.2 cm, 팔 관절 0.038 rad, TCP 최대 3.2 cm(평균 0.5 cm), 통과 |
| ACT 짧은 학습 | 합성 데이터 30 step 학습 → 체크포인트 → 서버 왕복 27 ms |
| SmolVLA(합성) | `lerobot/smolvla_base` 로드(카메라 이름 front/right/left → camera1–3 rename_map). 2 step 미세조정 → 서버 왕복, 50스텝 청크 200 ms |
| **SmolVLA(실데이터)** | BED-03 oracle 데모(dx24)로 2 step 미세조정(`lerobot-train --policy.path=lerobot/smolvla_base`, 메모리 1.9 GB) → 서버 → 실제 데이터셋 프레임 5개 왕복: (50, 24) 청크, 첫 요청 780 ms, 이후 205 ms(`smolvla_roundtrip_real.json`) |
| ACT closed-loop | (BED-03 oracle 데모 학습 후 10회 play — 결과는 아래) |

## ACT (BED-03 Easy)

### 학습

- 데이터: `data/lerobot/bed03_oracle_dx24`(24 에피소드, 34,891프레임, 카메라 3대 240 × 320).
- 설정: `dxbench_policy/train_act.sh`, chunk 30, n_action_steps 8, batch 16, 20k step(9.2 epoch), 17분(RTX 6000 Ada, 2.6 GB).
- loss 6.60(200 step) → 0.274(4k) → 0.122(8k) → **0.062(20k)**, l1 0.059(`train_act_tail.txt`).

### 개루프 확인(배관 검사)

학습 프레임 60개(데이터 전체에 고르게)를 서버로 보내 첫 행동을 기록 행동과 비교(`act_openloop_train.json`):

| 행동 묶음 | ACT 오차 | 참고: "현재 상태 그대로" 오차 | 행동 표준편차 |
|---|---|---|---|
| 팔 14 (rad) | 0.0101 | 0.0102 | 0.169 |
| 그리퍼 2 | 0.0 | 0.0099 | 0.248 |
| 몸통 6 (rad) | 0.0093 | 0.0119 | 0.329 |
| 베이스 2 (m/s, rad/s) | 0.0434 | 0.0195 | 0.214 |

상태·영상·행동의 순서와 정규화는 맞다(오차가 표준편차보다 한 자릿수 작음). 다만 20k step ACT는 "지금 자세 유지"보다
나은 것을 거의 배우지 못했고, 베이스 속도는 그보다 나쁘다.

### 폐루프 10회

`benchmarks/dx_bed_v0/experiments/act_bed03.yaml`(서버 `--backend lerobot --chunk 8`, 클라이언트 `open_loop_horizon 8`,
`rby1_joint`, 카메라 3대 + mp4, 상한 1800 step = 120 s, D-23), 평가 장면 `bed03_easy_000–009`:

| 항목 | 값 |
|---|---|
| **완료 기준: 오류 없이 10회** | **10/10 실행, 오류 0**, 결과 JSON 10개 + mp4 30개(머리·양손목) → `outputs/bench/act_bed03/` |
| 성공 | 0/10, 평균 점수 0.00. 어느 장면에서도 옷이 바구니에 들어가지 않음 |
| 전복 | 1회(bed03_easy_002, 마지막 구간에서 기울기 > 10°) |
| 시간 | 에피소드당 시뮬 120 s, 벽시계 평균 43분(3개 동시 + 다른 Kit 1개) |
| 행동 | 진행 로그의 베이스 궤적: 출발 후 약 4–5 m를 달려 옷과 바구니를 지나친 뒤 멈춤(000: x −2.6 → 2.3 m). 장면을 보고 목표로 가는 모습은 없음 |

해석: 데이터(24개, 1450프레임짜리 긴 과제)와 학습량(9 epoch)에 비해 과제가 길다. 개루프 확인으로 배관은 정상이니
다음은 학습량을 늘리는 것 → 100k step 추가 실험(D-27, `act_bed03_100k`). 큰 학습·데이터 확대는 HUMAN_TODO 5.

## 재현

```bash
N=21 P=3 STEPS=20000 tools/stage45.sh                       # 데모 → 변환 → 검증 → 재생 → ACT 학습 → 10회 play
P=~/miniforge3/envs/dx_policy/bin/python
$P -m dxbench_policy.serve.server --backend lerobot --checkpoint outputs/train/act_bed03/checkpoints/last/pretrained_model --chunk 8 --port 8791 &
$P -m dxbench_policy.serve.roundtrip --port 8791 --dataset data/lerobot/bed03_oracle_dx24 --n 60 --stride 577   # 개루프 확인
```

