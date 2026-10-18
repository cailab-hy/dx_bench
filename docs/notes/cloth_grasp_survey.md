# 천 잡기 방식 조사 (2026-10-18, D-U11 검토용)

공개 천·옷 조작 프로젝트가 그리퍼에 천을 어떻게 붙잡게 하는지, 각 레포의 소스를 받아 확인했다(`file:line @commit`).
비교 대상인 우리 방식은 `dxbench/embodiments/mdp/cloth_assist.py`다:
- 닫기 명령이 들어오면 TCP 진행 축 반경 5 cm 원통(TCP 아래 5 cm ~ 위 3 cm) 안의 정점을 잡는다.
- 잡은 정점은 매 step 속도를 써서 손을 따라오게 하고, 열기 명령이 들어오면 즉시 놓는다.
- 손 전체와 천의 충돌은 꺼져 있다(D-19).

## 프로젝트별

| 프로젝트 | 시뮬레이터·천 모델 | 잡는 조건 | 잡는 범위 | 붙이는 방법 | 손가락 접촉 필요 | 놓기 | 근거 |
|---|---|---|---|---|---|---|---|
| GarmentLab (NeurIPS'24) | Isaac Sim 2023, PBD 입자 천 | 스크립트 `grasp()` | 3 cm 블록 + 겹침 2 cm 안의 입자 | 보이지 않는 블록(질량 1000, 중력 끔)에 PhysX attachment, 블록은 속도로 손을 따라감 | 아니오(로봇↔옷 충돌 끔) | 블록 삭제 | `Env/env/Control.py:23-82, 114, 206-276` @6ee0620 |
| GarmentPile (CVPR'25) | Isaac Sim 2023.1, PBD | 손이 닿기 전(잡을 점이 정해지면) | 1 cm 블록 + 2 cm | attachment, 블록을 매 step 손 + 5 cm로 순간이동 | 아니오 | attachment 삭제 | `Env_Config/Utils_Project/AttachmentBlock.py:57-72` @046c924 |
| DexGarmentLab (2025) | Isaac Sim 4.5, PBD, Shadow hand | 손 닫기 | 접촉 | 마찰 25 + 점착(adhesion) 0.1. 논문도 점착은 "물리적 힘이 아님"이라고 밝힘 | 예 | 손 펴기 | `Env_Config/Garment/Particle_Garment.py:34-46` @e4e298e |
| LeIsaac fold_cloth | Isaac Lab, PBD ClothPrim | 그리퍼 닫기 | 접촉 | 마찰만(보조 코드 없음, 마찰값은 USD 안, 미확인) | 예 | 열기 | `tasks/fold_cloth/*` @24d3bcd |
| SoftGym (PyFlex) | FleX | 행동의 pick 플래그 | 가장 가까운 입자 1개, ≤ 약 5.6 cm | 역질량 0 + 위치 이동 | 아니오 | 질량 복원 | `softgym/action_space/action_space.py:117-160` @bb383f6 |
| Genesis | PBD | 사용자 지정 | 사용자가 고른 입자 | 링크 좌표 오프셋 유지하며 고정 | 아니오 | `release_particle` | `legacy_coupler.py:737` @50a2aa1 |

Isaac Lab 본체에는 공식 천 접기 task가 없다(issue #748, 입자 천 불안정으로 보류). MuJoCo는 확인하지 않았다.

## 우리 방식과의 차이

1. **반경:**
   - 우리 5 cm가 가장 느슨하다.
   - attachment 계열은 1–3 cm 블록에 2 cm 겹침을 쓰고, SoftGym은 입자 1개만 잡는다.
   - (c) 2–3 cm면 GarmentLab·GarmentPile과 같은 수준이다.
2. **잡는 조건:**
   - 보조 잡기를 쓰는 프로젝트 중 손가락이 실제로 닫혔는지 보는 곳은 없다.
   - 우리는 닫기 명령을 조건으로 쓰므로 이미 attachment 벤치마크보다 엄격하다.
   - 손가락 패드 사이에 천이 있는지까지 보면 조사한 어느 프로젝트보다 엄격해진다.
3. **붙이는 방법:**
   - 다른 곳은 솔버 안의 attachment 제약을 쓴다.
   - 우리는 속도 쓰기를 쓴다. surface deformable(FEM)에는 kinematic target이 없어서 고른 우회책이다(D-4).
   - 잡은 정점의 상대 위치를 유지한다는 점은 Genesis와 같다.
4. **놓기:**
   - 모두 즉시 놓는다. 서서히 놓는 곳은 없다.
   - GarmentLab `keep` 모드만 블록에 중력을 켜서 떨어뜨린다.
5. **마찰 잡기:**
   - Isaac에서도 되지만 PBD 입자 천에 큰 마찰(25)과 비물리적 점착을 줘야 했다(DexGarmentLab).
   - FEM surface deformable로 마찰 잡기를 한 프로젝트는 없다. 우리 1단계에서도 실패했다(D-5).
6. **실기 전이 근거:**
   - 얇다. DexGarmentLab(마찰)은 시뮬 데이터만으로 8–9/15, 실기 데모 15개를 더해 13/15였다.
   - GarmentLab은 실기 결과가 있지만 정책 출력이 "잡을 점"이라 잡기 물리 자체는 실기에서 시험되지 않았다.
