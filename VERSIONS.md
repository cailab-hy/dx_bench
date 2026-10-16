# 패키지 버전

## `dx_arena` (고정 — 바꾸지 말 것)

isaacsim 6.1.0.0, torch 2.11.0+cu128, numpy 2.3.1, warp-lang 1.16.0, newton 1.5.2, websockets 16.1.1.
추가 설치: `dxbench` (`pip install --no-deps -e dxbench`, DECISIONS D-6). 그 외 없음.

## `dx_policy`

| 날짜 | 설치 | 버전 | 이유 |
|---|---|---|---|
| (무인 실행 전) | lerobot | 0.6.1 | 기존 |
| (무인 실행 전) | torch / torchvision | 2.11.0+cu130 / 0.26.0 | 기존 |
| 2026-10-17 | `pip install "lerobot[training,smolvla]==0.6.1" websockets msgpack` | datasets 4.8.5, pandas 2.3.3, pyarrow 26.0.0, av 15.1.0, torchcodec 0.11.1, transformers 5.5.4, accelerate 1.15.0, wandb 0.27.2, num2words 0.5.14, websockets 17.2, msgpack 1.2.3 (+ 의존성) | LeRobot v3 데이터셋 변환·ACT 학습·SmolVLA 로드·정책 서버. dry-run에서 torch 불변 확인 |
| 2026-10-17 | dxbench_policy | 0.1.0 (`pip install --no-deps -e dxbench_policy`) | 이 레포의 정책 측 패키지 |
| 2026-10-17 | pytest | 9.1.1 | 변환기 테스트(`dxbench_policy/tests`) |

정책 서버(`dx_policy`, websockets 17.2)와 클라이언트(`dx_arena`, websockets 16.1.1)의 버전이 달라도 WebSocket 프로토콜 + msgpack 메시지라 호환된다.
