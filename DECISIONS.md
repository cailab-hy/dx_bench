# 결정 기록

형식: 날짜 · 결정 · 근거 · 되돌리는 방법. 사용자가 미리 정한 기본값(D-U*)과 이번 실행에서 스스로 내린 결정(D-*)을 구분한다.

## 사용자가 정한 기본값 (2026-10-16, 주말 실행 지시)

| ID | 결정 | 근거 | 되돌리기 |
|---|---|---|---|
| D-U1 | 게인 기본 `preset="tuned"`(점성 마찰만 + ζ0.7 + 중력보상 effort target), `original` 유지 | 0단계 보고서: 원본 게인 ζ0.01–0.04, GPU 크리프 | embodiment 생성 시 `preset="original"` |
| D-U2 | 그리퍼 mimic: 레이어에 `PhysxMimicJointAPI`(l2=−l1, r2=−r1), GPU 불안정 시 소프트웨어 mimic | T3d l2/r2 수동 | 레이어에서 mimic API 제거 |
| D-U3 | 캐스터 μ=0, combine "min" | T3e 회전 오차 54–66% | 재질 바인딩 변경 |
| D-U4 | 충돌체: `rby1_fullbody_collision.usd` 기반 + 빈 링크 convexHull + tool xform −0.1261 | T1 | `make_override.py --base` |
| D-U5 | 바퀴: damping 속도 드라이브, 정책 행동은 베이스 (v, ω) | 단순성 | embodiment action term |
| D-U6 | 정책·기록 15 Hz, physics dt 1/(15k), 시작 k=16 (1/240) | 실기 rby1_format 15 fps | env cfg decimation |
| D-U7 | PhysX 고정, Newton은 천이 PhysX로 안 될 때만 비교 | 스택 고정 | — |

## 이번 실행에서 내린 결정

| 날짜 | ID | 결정 | 근거 | 되돌리기 |
|---|---|---|---|---|
| 2026-10-16 | D-1 | 0단계 산출물을 첫 커밋(`stage 0`)으로 남기고 `.gitignore`에 `_ext/`, `data/`, `outputs/`, 체크포인트 추가 | 레포에 커밋이 없었음, 대용량 산출물 제외 | `.gitignore` 편집 |
| 2026-10-16 | D-2 | lightwheel_rby1_transfer(LICENSE 없음, 같은 연구실 레포)는 `_ext/`에 clone해 참고·이 비공개 레포 안에서만 이식. 외부 공개 전 소유자 확인 필요 | 사용자 지시(참고 레포), 라이선스 부재 | 이식 파일 삭제 |
| 2026-10-16 | D-3 | 그리퍼 mimic은 소프트웨어 mimic(ParallelGripperAction이 l2 = −l1, r2 = −r1 목표 구동, 4손가락 모두 K 100/D 1) | `PhysxMimicJointAPI`는 rotX/Y/Z 축만 받고 deprecated → 직선 관절 손가락에 쓸 수 없음 (D-U2의 대안 경로) | `dxbench/embodiments/mdp/actions.py` |
| 2026-10-16 | D-4 | 1단계 보조 잡기 = **속도 기반 정점 고정**(잡힌 정점 속도 = 게인·(목표−현재)/dt + 목표 이동 속도, 반경 5 cm, 게인 1.0) + 보조 모드에서 오른손가락↔천 충돌 필터 | 위치 덮어쓰기는 움직이는 큰 천에서 폭주(NaN), 실행 중 `OmniPhysicsVtxXformAttachment`는 천을 정지시킴, 속도만 쓰면 안정 | `--pin_mode position`, `--assist_filter 0` |
| 2026-10-16 | D-5 | 마찰 잡기로는 평평하게 놓인 천을 집지 못함 → 벤치마크 시뮬은 보조 잡기 사용, **sim-real 차이**로 기록 | 강체 매트리스·책상 위 2 mm 천을 평행 그리퍼로 꼬집을 수 없음(실기는 매트리스 탄성·손가락 패드) | 마찰 잡기 연구 시 재검토 |
| 2026-10-16 | D-6 | `dxbench`를 `dx_arena`에 `pip install --no-deps -e dxbench` (dry-run: dxbench만 설치, 고정 버전 불변 확인) | 사용자 지시(2단계) | `pip uninstall dxbench` |
| 2026-10-16 | D-7 | 로봇 USD `dxbench/assets/rby1/rby1_dx.usda` = `rby1_fullbody_collision.usd` + 24-DOF/루트 이동 + 재질(바퀴 μ1.2, 캐스터 μ0 min) + 빈 8링크 convexHull + 그리퍼 Xform −16 mm | D-U3, D-U4 | `make_rby1_usd.py` 재생성 |
| 2026-10-16 | D-8 | 바퀴 액션 = 속도 목표 + 외부 바퀴각 루프(명령각 적분, 감김 푼 실측각, k_pos 20/s) | 캐스터 μ0에서 damping 전용 속도 드라이브는 베이스가 계속 밀림(4 s에 5 cm) | `DiffDriveBaseActionCfg.k_pos=0` |
| 2026-10-16 | D-9 | 평지 테스트 배경은 `UsdGeom.Plane` 충돌체 (크게 스케일한 얇은 Cube 상자는 GPU 접촉이 떨려 로봇이 넘어짐) | t5 idle에서 확인 | `assets/scenes/flat_ground.usda` |
| 2026-10-16 | D-10 | env 인자 이름 `--gains`(tuned/original) — `--preset`은 Arena `--presets`와 argparse 접두어 충돌 | 실행 오류 | — |
| 2026-10-17 | D-11 | 물리 **480 Hz**를 평가·oracle·벤치마크 기본으로. 원격조작은 BED-02/03에 360 Hz 허용(셔츠·수건 통과, RTF ≈0.5), BED-01 이불은 480 Hz 유지 | 1단계: 이불 drift 360 Hz 2.07 cm(기준 2 cm 초과), 480 Hz 0.97 cm. 360 Hz 기록 데이터는 `episode.json`의 `physics_dt`로 구분 | `--physics_hz` 인자, `config/cloth_params.yaml` |
| 2026-10-17 | D-12 | LeRobot 데이터셋 두 형식: `rby1`(실기 16-D 호환) + `dx24`(=`rby1_joint` 24-D: 팔 14 + 그리퍼 2 + 몸통 6 + 베이스 2). 학습·closed-loop는 dx24 | BED 과제는 이동·몸통 숙이기가 필수라 실기 16-D로는 못 풂. 추가 열은 `dx.*`(정책이 무시) | `--fmt rby1` |
| 2026-10-17 | D-13 | `dx_policy`에 `lerobot[training,smolvla]==0.6.1`, websockets, msgpack, pytest 설치, `dxbench_policy` editable | 변환·학습·서버. dry-run에서 torch 불변 확인(`VERSIONS.md`) | `pip uninstall` |
| 2026-10-17 | D-14 | 정책 서버 "zero" 기준선 = **현재 자세 유지(hold)**: 관절 목표 = 현재 관절, 그리퍼 열림, 베이스 정지 | 절대 관절 행동에서 0을 보내면 팔이 0 rad로 휘둘려 위험·무의미 | `--backend hold` 대신 상수 0 백엔드 추가 |
