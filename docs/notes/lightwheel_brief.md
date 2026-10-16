# lightwheel_rby1_transfer 요약 (2026-10-16, commit 1178454, `_ext/lightwheel_rby1_transfer`)

Isaac Sim 5.1 / Isaac Lab 2.3 / Python 3.11 기준. **LICENSE 파일 없음**(같은 연구실 cailab-hy 레포) → 이 비공개 레포 안에서만 이식, 외부 공개 전 확인 필요. `scripts/vendor/`(LW-BenchHub predicates)는 Apache-2.0.

## rby1_format.py (Isaac 무관, numpy+PIL)
- `FPS=15`, `ROBOT_TYPE="rby1"`, 원본 데이터셋 `rainbowrobotics/icra_0526_compound_rel`(LeRobot v3).
- `NAMES`(16) = right_arm_0..6, left_arm_0..6, right_gripper_0, left_gripper_0 (오른팔 먼저).
- 팔: 절대 관절 rad, action = 다음 스텝 절대 목표. 그리퍼: 1=열림, 0=닫힘. state = clip(((q_b−q_a)/2)/0.045, 0, 1)
  (`FINGER_OPEN_M=0.045`; l1/r1 음수, l2/r2 양수), action = 0/1 이진 명령(`gripper_commands()`).
- 함수: `state16(q18, names)`, `action16(a18, names)`, `to_sim18(v16, names, template)`, `resample_indices(n_raw, raw_fps, fps=15)`, `adapt_image(img, camera, rendered_for_rby1)`.
- 카메라 키: first_person→front (480,640), right_hand→right (640,480 세로), left_hand→left (640,480, 180° 회전). AV1.
- feature: `action`, `observation.state` float32 (16,), `observation.images.{front,right,left}` video.

## 키보드 수집 (collect_keyboard.py, Isaac 의존, 1150줄)
- 15 fps, SUBSTEPS=ceil(100/fps)=7. Se3Keyboard: W/S x, A/D y, Q/E z, Z/X roll, T/G pitch, C/V yaw;
  B 시작, ENTER 저장(성공했을 때만), BACKSPACE 폐기+리셋, R 리셋, P 일시정지, TAB 팔 전환(왼팔부터), K 그리퍼 토글, L 정지, ESC 종료.
- clutch 없음(키 누르는 동안만 이동). 상대 DiffIK(dls, λ 0.03), 관절 slew 0.8 rad/s, 손가락 0.06 m/s.
- 프레임 기록: observation.state(18), velocity(18), action(18 절대, 손가락 m), ee_pose(14, wxyz), teleop.command, success_original, sim_time, camera_time.

## 원본 기록 (keyboard_recorder.py, Isaac 무관)
- `EpisodeRecorder(root, metadata)`: start(initial_state) / append(numeric, images, success_state) / save(success_original, smoke) / discard() / close().
- `<out>_raw/pending-<uuid>/` → 저장 시 `episode-<uuid>/`: `initial_state.json`, `<camera>/%06d.png`, `trajectory.npz`, `success_state.jsonl`, `episode.json`.

## 변환·검증 (lerobot 필요)
- `export_keyboard_episode.export(raw, root, fmt="rby1")`: `LeRobotDataset.create(repo_id, root, fps=15, features, robot_type="rby1", video_backend="pyav", vcodec="libsvtav1", image_writer_threads=3)` → add_frame(frame with "task") → save_episode(parallel_encoding=False) → finalize(). 매니페스트로 멱등.
- `validate_keyboard_dataset.validate(root)`: 개수, state/action bit-exact 재계산, 타임스탬프, PNG 수, 영상 MSE<50, PyAV 프레임 수 → validation.json.

## VR (scripts/vr/*, 표준 라이브러리+numpy+PIL)
- `vr_server.py`: ThreadingHTTPServer 0.0.0.0:8012, WebXR 페이지(plain HTTP, `adb reverse tcp:8012 tcp:8012`로 localhost=secure context), 수제 RFC6455 websocket `/ws`.
- 헤드셋→PC JSON `{"kind":"frame","v":4,"hands":{left|right:{tracked,position,orientation[wxyz],buttons,axes}},"head":...}` ~90 Hz; PC→헤드셋 binary(4B seq + JPEG).
- `vr_teleop.py`: XR→로봇 축 `[[0,0,-1],[-1,0,0],[0,1,0]]`, 머리 yaw recenter, grip clutch(히스테리시스 0.6/0.4), trigger=그리퍼, A 시작, B 1 s 폐기, X 일시정지, Y 1 s recenter. 출력 `Step.targets[side]=(p,R)` base frame.
- `scripted_operator.py`: `T1Operator(fps).step(...)`가 clutch 수식을 역으로 풀어 헤드셋 리포트를 실제 `WSClient`로 보냄(헤드셋 없이 끝까지 테스트).

## kinematics.py
- USD 물리 관절로 Pinocchio 모델 구성(`USDModel`), vendored usd-core cp311(우리 py3.12엔 안 맞음 → Kit pxr로 대체). TCP = ee_{side} + (0,0,−0.1045). READY(deg) left [15,65,15,−115,−75,−65,−5], right [15,−65,−15,−115,75,−65,−5].

## Isaac Lab 3.0 이식 위험
- 전부 wxyz 가정(DiffIK 입력 등) → 3.0은 xyzw. `isaacsim.core.api` World/SingleArticulation, Se3Keyboard private 메서드 override.
