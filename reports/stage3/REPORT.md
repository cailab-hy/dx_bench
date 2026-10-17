# 3단계 — 침실 task (BED-01 / 02 / 03)

## 산출물

| 항목 | 위치 |
|---|---|
| CareRoom 전역 자산 | `dxbench/assets/scenes.py` `DxCareRoom` (`/World/DX_Hub_Building_add_obj`에 고정; 수납장 WallAnchor 유지) |
| 새 자산 USD | `dxbench/assets/objects/`: 빨래 바구니 60 × 44 × 32 cm(강체 0.8 kg, 손잡이 홈), 셔츠 4종(s_red 0.85 / m_blue / l_gray 1.15 / m_white), 인형, 책. 수건·양말은 절차 생성 천(`assets/objects.py`) |
| 장면 명세·생성기 | `tasks/scene_spec.py`, `tasks/scene_gen.py` — 평가 seed 0–9(90개), 학습 seed 10000+(task당 30개), 셔츠·수건 변형은 학습/평가 분리(OOD) |
| 천 초기 상태 | `tasks/make_cloth_states.py` + `tools/make_states.sh`. "떨어뜨려 안정" 후 정점 `.npz`, 매 리셋에 로드(평가 결정성) |
| 판정 | `tasks/predicates.py`(numpy) + `tasks/bed_tasks.py`(Arena `TaskBase`). 단계 플래그, 부분 점수, 실패 조건, 언어 지시(영·한) |
| 테스트 | `dxbench/tests/test_predicates.py` 14개(정답/오답 상태), `tools/task_check.py`(시뮬 안 강제 정답 상태) |
| oracle | `policies/oracle_bed03.py`(BED-03), `policies/oracle_bed02.py`(BED-02 Easy, 미완) |
| 실행 | `scripts/run_episode.py`(결과 JSON, mp4, 원본 기록), `tools/run_oracle.sh`, `dxbench.bench.run_benchmark` |

## BED-03 oracle 개발 중 고친 시뮬·정책 문제

| 증상 | 원인 | 조치 |
|---|---|---|
| 바구니로 돌진 후 전복 | 접근점이 바구니와 겹침 | 16방향 접근점 + A* 경로(가구 AABB, 벽, 바구니 팽창) |
| 바닥 천을 밟고 뒤집힘 | 바퀴 ↔ surface deformable 접촉력 폭주 | 바퀴·베이스 ↔ 천 충돌 필터(D-18) |
| 목표 주위를 맴돎 | 0.35 m/s · 0.8 rad/s 호가 마지막 구간보다 넓음, 베이스는 바퀴 축 중심으로 회전 | 축 중점 제어, 큰 방향 오차에서는 제자리 회전 |
| 천을 집지 못함 | 앞 0.30 m에서는 deep 자세여도 손끝이 바닥 4 cm 위 → 보조 잡기 반경(5 cm) 밖 | 집는 거리 0.18 m(손끝 1.3 cm) |
| 바구니 밖에 떨어뜨림 | 서 있는 자세로는 바구니 위에 손이 안 닿음(오차 28 cm) | crouch로 놓기, 0.75 m에서 접근 후 0.50 m로 내림 |
| 바구니 테두리에 걸침 | 50 × 35 cm 바구니가 셔츠보다 작음 | 바구니 60 × 44 cm, 놓은 뒤 한 번 눌러 넣기, 바구니 한 면을 마주 보고 접근(D-20) |
| 떠나면서 바구니를 넘어뜨림 | 제자리 회전 시 베이스 뒤쪽이 휩쓺 | 놓은 뒤 0.3 m 후진 |
| 들고 가다 튕겨 나감 | 손가락만 천과 필터 → 그리퍼 몸체가 고정 정점과 겹침 | 손 전체 필터(D-19) |
| 아무 데서나 튕겨 나감 | DX 바닥 충돌체가 스케일한 얇은 Cube | 바닥을 Plane으로 대체(D-22) |
| 옷이 바구니 밑에 깔린 장면, 들면 바구니가 넘어짐 | 장면 생성 간격(0.55 m)이 옷 크기를 무시 | 바구니–옷 1.02 m, 옷–옷 0.85 m로 장면·초기 상태 재생성 |
| 두 번째 옷에 손이 안 닿음(IK 오차 0.6–0.8 m) | 놓은 뒤 손을 높은 위치(그리퍼 아래 방향)로 접어 손목 한계 근처에서 다음 집기를 시작 | 놓은 뒤 시작 TCP 자세(READY)로 복귀 |
| 병렬 실행 시 즉시 종료 | Arena 기록 HDF5 파일명이 초 단위 타임스탬프라 동시에 시작하면 파일 잠금 충돌 | 프로세스별 파일명(`ArenaRuntime`), 시작 시차 |

## 결과 — BED-03 Easy oracle, 평가 장면 10개 (합격 기준 ≥ 8/10)

| 성공 | 평균 점수 | 첫 번째 옷 바구니 | 두 번째 옷 바구니 | 로봇 전복 | 오류 | 평균 sim 시간 |
|---|---|---|---|---|---|---|
| **8/10** | 0.90 | 100 % | 80 % | 0 | 0 | 114 s (과제 한도 180 s) |

- 장면별 표: `reports/stage3/oracle_bed03_easy.md`. 원본: `outputs/bench/oracle_bed03/`.
- 실패 2건(006, 009)은 두 번째 옷에서 났다.
  - 006: 바구니 위로 팔을 뻗다 IK가 멈춤.
  - 009: 집기가 비어 옷이 안 들림.
  - 그 뒤 oracle에 두 가지 회복을 넣었다(집기 실패 시 홈 복귀 후 재시도, 바구니 위 IK 정지 시 홈 복귀). 데모 수집에는 이 판이 쓰였다.
- 코드 판 주의:
  - 결과 10개 중 일부는 평가 버그 수정 전 실행본이다. 성공 종료 직후 리셋된 장면을 다시 평가해 실패로 기록됐다.
  - 집계기가 `terminated_by == "success"`로 바로잡는다(성공 = 점수 1.0).
  - 004는 홈 복귀 도입 전 코드의 성공이며, 최종 코드로 재실행 중이다.
- 벽시계 시간: 에피소드당 약 75–90분(5개 동시, 480 Hz, 천 2–3벌).

## BED-01 / BED-02 (best-effort)

- BED-02 Easy oracle(오른팔 단독 접기): 점수 0.3(flatten만), 접기 실패.
  - 원인: 오른팔은 책상 높이에서 왼쪽 절반(y > 0)에 닿지 않는다(`reports/rby1_check/t5/reach_desk_*.json`).
  - 양팔 접기가 필요하다 → HUMAN_TODO 8.
- BED-01 oracle은 만들지 않았다. 장면·판정·초기 상태는 준비돼 있다.
