# 사람이 할 일 (우선순위 순)

> 주말 무인 실행 중에 사람이 해야 해서 미룬 일. 위에서부터 처리하면 된다. 각 항목에 명령어와 준비된 파일을 적는다.

1. **결정 검토**: `DECISIONS.md`에서 D-4/D-5(보조 잡기 = sim-real 차이)와 D-11(물리 480 Hz, 원격조작 360 Hz 허용)을 확인.
   되돌리는 방법은 각 행에 있다.
2. **스위블 캐스터 모델링** (D-U3): 현재 캐스터는 μ0 실린더(마찰 없음).
   - 필요한 것: 실기 캐스터 치수(휠 지름, 오프셋, 스위블 축 위치).
   - 할 일: `dxbench/dxbench/assets/rby1/make_rby1_usd.py`에 revolute 스위블 + 휠 관절 2개씩 추가.
3. **π0 / π0.5 가중치** (각 5 GB 초과 → 받지 않음).
   - `dx_policy`에서 `huggingface-cli download lerobot/pi0` 등으로 받는다.
   - 서버: `python -m dxbench_policy.serve.server --backend lerobot --checkpoint <dir>`(`docs/PIPELINE.md` 6절), 확인은 `python -m dxbench_policy.serve.roundtrip`.
4. **사람 원격조작 데모** (키보드 / VR 헤드셋).
   - 실행 명령과 키맵은 4단계 기록에 있다.
   - VR 헤드셋은 실제 착용 테스트가 필요하다(무인 실행에선 scripted operator로만 검증).
5. **큰 학습 실행**: SmolVLA·π0 미세조정, 데모 확대(현재 BED-03 Easy oracle 24개).
   - 무인 실행에서는 ACT 20k step(0/10)과 100k step(`act_bed03_100k`, 결과는 `benchmarks/dx_bed_v0/RESULTS.md`)만 했다.
   - SmolVLA 미세조정 명령은 `docs/PIPELINE.md` 5절(rename_map 포함). 2 step 왕복까지만 확인했다.
6. **결과물 정리**: 안전 검사 때문에 실험 중간 산출물을 지우지 못했다.
   - 대상: `reports/cloth_spike/runs/*_{dbg,probe*,smoke,pin_*}.json`, `outputs/` 아래 디버그 폴더.
   - 필요 없으면 사람이 지운다.
7. **RTF 단독 측정**:
   - 1단계 카메라 포함 RTF는 다른 Kit 작업과 GPU를 같이 쓴 값이다.
   - 다른 작업이 없을 때 `STAGE=rtf tools/cloth_spike/run_matrix.sh`를 다시 돌려 `config/cloth_params.yaml`의 원격조작 Hz를 확정한다.
8. **BED-01 / BED-02 oracle** (best-effort, 미완):
   - BED-02 Easy 오른팔 단독 접기는 실패했다(점수 0.3, flatten만).
     - 원인: 오른팔은 책상 높이에서 왼쪽 절반(y > 0)에 닿지 않는다(`reports/rby1_check/t5/reach_desk_*.json`).
     - 양팔 oracle(`oracle_bed02.py`)은 구현했고 점수 0.55(flatten + 쌓기)지만 접기 0/3(`reports/stage3/REPORT.md` 표).
       - 2026-10-18–19: 수건 영률 1e5(D-28), 양손 반 접기 먼저(D-30), 쌓는 자리를 옷 바로 뒤로 옮기고 양손 쌓기(D-32).
         oracle 최고 0.7(v12)에서 멈춤 → `reports/stage3/REPORT.md` BED-02 v1–v15 표.
       - **제안: BED-02 데모도 사람 원격조작으로**(BED-01과 함께). 양손 반 접기·양손 쌓기는 VR이 맞다.
   - BED-01 oracle은 만들지 않았다(이불 양 모서리를 두 팔로 잡아 끄는 동작).
   - 사람 데모(키보드/VR)를 모으는 편이 빠를 수 있다.
9. **실기 잡기 조건 맞추기** (D-34): 실기 RB-Y1로 펼친 수건을 손끝 높이(0 / 2 / 4 / 6 / 10 mm)·위치 오차별로 수십 번 집어 성공률을 재고, `pad_grasp`의 `tip_margin`(지금 4 mm)과 잡기 실패 확률을 그 결과에 맞춘다.
10. **이불 실측** (D-44): 실험실 이불(2.0 × 1.55 m) **무게**를 재 주세요. 지금은 1.2 kg 추정(`assets/objects.py` `DUVET_MASS`).
    - 실제 그리퍼로 이불 모서리를 당길 때 미끄러지는지도 보면 이불 잡기(D-37 지지 정점) 설정을 정할 수 있다.
