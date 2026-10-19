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
| D-U8 | (2026-10-18) **BED-02: 수건 전용 천 재질(굽힘 강성↓)을 허용**하고 양팔 oracle을 계속. 반 접기는 순서를 바꿔 도달 가능하게 | 놓으면 접힌 쪽이 다시 펴짐(`reports/stage3/REPORT.md` BED-02 표). 수건이 들어간 장면(BED-03 포함)의 천 초기 상태 재생성과 BED-03 oracle 재확인이 따라옴 | 수건 재질을 공용 `rect_cloth` 값으로 되돌리고 상태 재생성 |
| D-U9 | (2026-10-18) **BED-01은 나중에 사람 원격조작 데모로 채움**(oracle 만들지 않음) | 이불 두 모서리를 양팔로 잡아 펴는 동작은 스크립트가 어려움. 키보드·VR 도구는 준비됨 | — |
| D-U10 | (2026-10-18) **학습: oracle 데모 확대 + 긴 ACT 학습과 SmolVLA 미세조정을 동시에** 진행해 같은 데이터로 비교. π0는 결과를 본 뒤 결정 | ACT 20k·100k 모두 0/10이지만 100k에서 옷을 바구니에 넣는 행동이 처음 나옴 → 데모 수가 병목 | — |
| D-U11 | (2026-10-18) **실기 전이가 목표 → 보조 잡기 반경 5 cm → 2.5 cm**(`cloth_assist.py`, `--assist_radius`). 패드 사이 상자 방식(c+)은 택하지 않음. 조사: `docs/notes/cloth_grasp_survey.md`(GarmentLab·GarmentPile attachment 약 2 cm 겹침과 같은 수준, 닫기 명령 조건은 그들보다 엄격) | 시뮬 성공이 실기 성공을 뜻하도록. 반경을 바꾸면 oracle 성공률 재확인과 데모 재수집이 필요 | 반경 5 cm로 되돌림(`cloth_assist` 설정) |

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
| 2026-10-17 | D-15 | `TorsoPresetAction`: torso_1·2·3를 **함께** 직선 보간(코사인 시간 프로파일, 최대 관절 최고속 0.5 rad/s) | 관절별 등속 램프(0.6 rad/s)는 deep으로 가는 중간에 (90, −90, 90)° 자세를 지나는데, 이 자세는 COM이 앞바퀴 축 위(여유 0.8 cm)이고 토크 118 %라 넘어짐(t5 floor 52°). 동기 보간 후 deep + 바닥 집기 통과(손끝 1.3 cm, 기울기 0°, COM 여유 9.7 cm) | `mdp/actions.py` 이전 apply_actions |
| 2026-10-17 | D-16 | 바닥 집기 자세 = 기존 `deep` 프리셋(90, −130, 90)° + 오른팔 IK, 베이스 앞 15–20 cm | 기구학 탐색(`floor_posture.json`): 바닥에 닿으면서 COM 여유 ≥ 5 cm인 자세는 몸통 토크 88–100 %. 동역학 시험에서 deep 정지 토크 최대 82 %(torso_3) | `TORSO_PRESETS["deep"]` |
| 2026-10-17 | D-17 | 그리퍼 회귀 시험: 큐브 중심 0.90 m, 합격 = 12 cm 이상 들기 + 미끄럼 ≤ 1 cm (0단계 18 cm에서 완화) | 그리퍼 아래 방향을 유지하면 오른 손목 arm_5 상한(1.92 rad) 때문에 TCP z ≈ 1.05 m 위로 자세 유지 불가(회전 오차 0.15–0.25 rad), 0.80 m는 IK 도달 오차 6.7 cm. 잡기 자체는 미끄럼 0.2 mm로 양호 | `--cube_xyz`, t5 기준 |
| 2026-10-17 | D-18 | 바퀴·베이스(캐스터 포함) ↔ 천 충돌 필터(FilteredPairs) — 로봇은 바닥 천 위를 평평한 것처럼 지나감 | 바퀴가 PhysX surface deformable 위로 올라가면 접촉력이 폭주해 로봇이 뒤집힘(BED-03 oracle 153°). 실기도 얇은 천은 밟고 지나감. 천이 바퀴에 말리는 현상은 재현 못 함(sim-real 차이) | `bed_tasks.py`의 `add_base_cloth_filter` 제거 |
| 2026-10-17 | D-19 | 보조 잡기 충돌 필터를 손 전체(손가락 + 그리퍼 몸체 + ee + 손목 arm_5·6)로 확장 | 이동 로봇에서 TCP 5 cm 안에 고정한 정점이 그리퍼 몸체와 겹쳐, 속도 핀이 접촉과 싸우며 에너지를 넣어 베이스가 튕겨 나감(BED-03 oracle 002: 2 s에 2.4 m, 1 m 높이). 1단계는 고정 베이스라 드러나지 않음. 대가: 손목·그리퍼 몸체로 천을 밀 수 없음 | `cloth_assist.HAND_LINKS`를 손가락 4개로 |
| 2026-10-17 | D-20 | 빨래 바구니 60 × 44 × 32 cm로 키움(이전 50 × 35 × 30) + oracle은 바구니 한 면을 마주 보고 접근, 0.50 m에서 놓고 0.40 m까지 한 번 눌러 넣음 | 50 × 35 cm 바구니에 셔츠를 떨어뜨리면 테두리에 걸침(천 평균 높이 0.33 m). 실제 빨래 바구니 크기(60 × 40 × 30 cm 전후) | `make_objects.py` L, W, H, `BasketSpec` |
| 2026-10-17 | D-21 | BED-02 장면: 옷을 책상 앞 모서리에서 25–30 cm 안쪽에 둠(책상 중앙 X), 쌓는 위치 = 앞 모서리 22 cm 안쪽·중앙선 왼쪽 30 cm | 책상 깊이 75 cm, 베이스는 모서리에서 37 cm 떨어져야 하고 팔은 책상 높이에서 0.8 m까지 닿음 → 책상 중앙의 먼 모서리(0.97 m)는 닿지 않음 | `scene_gen.gen_bed02`, `Bed02FoldTask.stack_xy` |
| 2026-10-17 | D-22 | 침실 task env에서 DX 바닥 충돌체(29 × 47 × 0.1 m로 스케일한 Cube)를 끄고 보이지 않는 충돌 Plane(z = 0)으로 대체 (prestartup에서 스테이지만 수정, DX 파일 불변) | 무작위 위치에서 로봇이 2–3 m 튕겨 나가며 뒤집힘(BED-03 oracle 002·003). 평지 시험에서 얇은 스케일 Cube가 GPU 접촉을 떨게 한 것과 같은 원인(D-9) | `bed_tasks._Events.floor_plane` 제거 |
| 2026-10-17 | D-23 | 학습 정책·재생 기준선 평가는 에피소드를 **1800 step(120 s)**에서 자름(과제 정의는 180 s). oracle과 hold는 원래 길이(hold는 300 step). 처음엔 1500이었으나 학습 장면 oracle 데모가 1340–1567 step이라 데모 속도로 따라 해도 잘리므로 1800으로 올림(2026-10-17 11:50) | 침실 task는 480 Hz + 천 + 카메라에서 step당 1–3 s(동시 실행 시). 180 s 실패 에피소드 하나가 2시간 가까이 걸림 | 실험 YAML의 `max_steps` 삭제 |
| 2026-10-17 | D-24 | `rby1_ik` 팔 액션 = `HomingDiffIKAction`: 상대 DiffIK + 관절공간 "홈"(READY) 복귀 요청 경로(행동 벡터 밖, 원격조작 H 키·oracle 사용). 기록되는 관절 목표에는 홈 복귀가 포함 | 바구니 위에 놓은 뒤 상대 IK만으로 손을 되돌리면 관절 한계 근처에서 멈춰, 두 번째 옷 집기 오차가 0.6–0.8 m(BED-03 oracle 000·003). Isaac Lab null-space 관절 한계 회피(gain 0.5)는 그리퍼 자세를 기울여(그리퍼 들기 14.4 → 10.3 cm) 쓰지 않음 | `rby1._ik`를 `DifferentialInverseKinematicsActionCfg`로 |
| 2026-10-17 | D-25 | 원본 기록 영상 PNG → **JPEG q95**(`recorder.py`), 이전 PNG 기록은 `tools/raw_png_to_jpg.py`로 재압축 | PNG(압축 1)는 카메라 3대 × 1500프레임에 에피소드당 약 4 GB → 데모 20개면 데이터 한도(50 GB) 초과. JPEG는 약 0.5 GB, 변환 검증 영상 MSE 5–17(기준 50) | `recorder.py` 저장 형식 |
| 2026-10-17 | D-26 | 재생 검증 합격 기준을 "모든 프레임"에서 **"팔 관절 0.05 rad 안 프레임 ≥ 97 %, TCP 오차 p99 ≤ 2 cm, 최종 베이스 xy ≤ 5 cm"**로 바꿈. 엄격 기준은 `pass_strict`로 계속 기록 | 실제 데모 0번(1361프레임) 재생: 과제 성공(점수 1.0), 베이스 최종 2.9 mm, TCP 평균 1.8 mm·p99 9.8 mm. 그러나 옷을 들고 가거나 바구니에 누를 때 2–5프레임짜리 관절 튐(최대 0.25 rad, right_arm_3)이 27프레임(2 %) 있었음. 시간 어긋남이 아니라(±1프레임 이동해도 그대로) 천 솔버가 비결정적이라 접촉력이 실행마다 달라서 생김 | `compare_replay.py`의 `res["pass"]`를 `pass_strict`로 되돌림 |
| 2026-10-17 | D-27 | 5단계 완료 기준(ACT 10회 play)은 20k step ACT로 채우고, 같은 데모로 **100k step ACT를 추가 학습·평가**(`act_bed03_100k.yaml`, `tools/stage5_long_act.sh`). 재생 기준선 뒤에 GPU를 혼자 쓸 때 돌림 | 20k step ACT(약 9 epoch)는 첫 5회 모두 1800 step 동안 점수 0. 학습 프레임에 대한 개루프 오차가 "현재 상태 복사"와 비슷(팔 0.010 rad vs 0.0102)하고 베이스는 더 나쁨(0.043 vs 0.0195) → 배관 문제는 아니고 학습 부족. 100k step은 약 85분이라 주말 안에 가능 | 실험 YAML 삭제, `outputs/train/act_bed03_100k` 정리 |
| 2026-10-18 | D-28 | **수건 전용 영률 1e6 → 1e5**(`assets/objects.py` `TOWEL_YOUNGS`, 다른 천은 그대로). 수건이 들어간 장면(BED-02 easy·hard, BED-03 전체, 학습 BED-02/03 easy)의 천 초기 상태를 다시 만든다 | 로봇 없는 접기 시험(`tools/cloth_spike/towel_fold.py`, 2.5 cm 잡기로 짧은 변 가운데를 호로 넘긴 뒤 놓기, 설정당 3회): 1e6은 놓은 뒤 접힘 유지 1/6(그대로 펴짐), 3e5 + 두께 0.8 mm 0/3, **1e5 3/3**, 3e4 3/3. 평균 변형률 1e5 0.7–0.9 %(1e6은 0.4 %). 두께만 줄이면 효과 없음. 결과가 실행마다 달라(GPU 솔버 비결정) 반복이 필요했음. 한 점 잡기로는 접힌 쪽이 평평하지 않고 높이 6–12 cm 원뿔로 구겨짐 | `TOWEL_YOUNGS = 1.0e6`, 상태 재생성 |
| 2026-10-18 | D-29 | BED-03 oracle: **닫은 직후 잡힌 정점 수 확인**(`cloth_assist.grasped`, 0이면 손을 열고 다시 내려가기 최대 2회), 같은 자리 재시도 2회 실패 시 **접근 경로 재계획 1회**, 그래도 실패하면 **빈손으로 바구니에 가지 않고 다음 옷으로** | 반경 2.5 cm 첫 평가 7/10(5 cm 8/10). 실패 3개 중 009는 내려가기 끝 TCP 오차 2.2 cm로 빈손, 003은 같은 자리에서 팔이 안 닿음(오차 38 cm), 둘 다 빈손으로 바구니까지 가서 시간 초과. 006(바구니 위 IK 실패)은 5 cm에서도 실패 | `oracle_bed03.py`의 `reopen` / `replans` / `skip` 분기 제거 |
| 2026-10-18 | D-30 | BED-02 oracle 접는 순서를 **양팔 반 접기(가까운 긴 변 양 모서리 → 먼 긴 변) → 오른쪽 → 왼쪽 → 쌓기**로, 호 높이 **10 cm 고정**, 책상까지 거리 5 → 3 cm. 잡는 높이는 천 전체 최고점이 아니라 잡을 위치 주변 천 높이 | 로봇 없는 시험(`towel_fold.py`, 2.5 cm 잡기, 수건 1e5): 변 가운데 한 점 잡기는 원뿔로 구겨짐, 두 모서리 동시 잡기는 평평한 접힘. 한 변의 두 모서리는 같은 쪽이라 양팔이 못 잡으므로 반 접기를 먼저(가까운 긴 변 양 끝은 좌우 ±21 cm라 두 팔 다 닿음). 이 순서로 호 15 cm: 세 접기 0/3, 6 cm: 0/2, **10 cm: 양옆 4/4·세 접기 2/4**, 양옆 먼저: 0/3. 최고점 기준 잡기 높이는 구겨진 곳이 있으면 잡을 곳 위 5 cm 넘게 떠서 아무것도 못 잡음 | `oracle_bed02.py` 이전 버전(커밋 기록) |
| 2026-10-18 | D-31 | D-U10 실행: **학습 장면 70개 추가(seed 10030–10099, 총 100)**, 새 설정(2.5 cm·수건 1e5)으로 **oracle 데모 최대 100개**(`data/raw/bed03_oracle_v2`, `outputs/demos_v2`, 5개 동시) → dx24 변환 → **ACT 100k와 SmolVLA 20k(batch 32)를 동시에 학습** → 각 10회 평가(`act_bed03_v2.yaml` 3개 동시, `smolvla_bed03.yaml` 2개 동시) → `RESULTS.md`. 한 번에 `tools/stage_v2.sh` | 이전 데모 24개는 5 cm·수건 1e6 조건이라 섞지 않음(보존). 수집 중 다른 GPU 학습을 같이 돌리면 PhysX 커널 실패가 났으므로 학습은 수집 뒤에. 용량: 원본 약 21 GB 추가, 합계 약 34 GB(한도 50 GB) | `tools/stage_v2.sh` 중지, `data/raw/bed03_oracle_v2` 정리 |
| 2026-10-18 | D-32 | (사용자 선택 ③) **BED-02 쌓는 자리 이동**: 책상 앞 모서리에서 22 cm 안·책상 중심선 왼쪽 30 cm → **40 cm 안·옷들의 평균 y(로봇 기준 옷 바로 뒤)**(`bed_tasks.Bed02FoldTask`). 접은 수건은 **양손으로 양 끝을 잡아 함께 옮김**(`oracle_bed02` stack). 옆 접기 잡는 점은 가까운 변에서 12 cm 안으로 제한 | 왼쪽 30 cm는 오른팔이 닿지 않아 왼손 한 점으로만 옮겼고, 그러면 반 접기가 풀림(v12). 도달 계산(`reports/rby1_check/t5/reach_desk_±0.12.json`, lean): 각 팔 자기 쪽 ±12 cm는 앞 0.55–0.75 m·높이 0.76–0.80 m에서 오차 ≤ 2.6 cm, 반대쪽 +12 cm는 0.62 m까지만 → 양손이 각자 자기 쪽 끝을 잡으면 정면 0.75 m까지 옮길 수 있음. 옷은 25 cm 안에서 시작하므로 펼친 수건 중심이 쌓는 자리에서 15 cm(반경 12 cm 밖) | `stack_xy = (hi[0]-0.22, cy-0.30)`로 되돌림 |
| 2026-10-19 | D-33 | **BED-02 oracle 작업 중단**(최고 0.7, v12), BED-02 데모는 사람 원격조작 제안(HUMAN_TODO 8). 정전(10-19 오전) 뒤 데모 수집 재개: 저장된 78개 무결성 검사 통과(프레임 = 행동 수 + 1은 원래 형식, v1과 동일), 끊긴 녹화 6개(`pending-*`) 삭제, 결과 없는 장면만 다시 수집. SmolVLA 체크포인트 저장 간격 5k → 10k(용량) | v13–v15에서 개선 없음(0.0–0.55). 남은 문제(반 접기가 덜 닫히면 왼팔 도달 실패, 놓은 뒤 접힘 유지 편차)는 스크립트로 풀기보다 사람 데모가 빠름 | `oracle_bed02.py` 작업 재개 |
