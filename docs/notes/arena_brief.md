# IsaacLab-Arena 0.3.1 요약 (2026-10-16, 탐색 에이전트 3건 종합)

## 등록·실행
- `@register_asset`(cls.name 키) / `@register_policy`(name, `PolicyBase[Cfg]` 직접 상속) / `@register_retargeter` (`isaaclab_arena/assets/register.py`). 외부 패키지는 자동 import 안 됨 → `--external_environment_class_path pkg.mod:Class env_name`로 모듈 import 시 데코레이터 실행.
- 환경 팩토리: `ExampleEnvironmentBase`(`get_env(args_cli)`, `add_cli_args(parser)`, `self.asset_registry`) 또는 `ArenaEnvironmentFactory[Cfg]`. 반환 `IsaacLabArenaEnvironment(name, embodiment, scene=Scene(assets=[...]), task, env_cfg_callback=fn, episode_recorder_terms={...})`.
- 정책 실행: `python isaaclab_arena/evaluation/policy_runner.py --policy_type <name|dotted.Class> --num_episodes N [--record_camera_video] --external_environment_class_path ... env_name [env args]`. 외부 정책은 dotted path. 결과 `outputs/<ts>/episode_results_rank0.jsonl` + HTML.
  - 버그: `policy.is_remote=True`면 `shutdown_remote` 없음으로 크래시 → is_remote False 유지.
- 실험 실행기: `isaaclab_arena/evaluation/experiment_runner.py --experiment_config x.yaml` (`shared:` + `runs: {<task>_<diff>_<policy>: {environment, environment_builder, policy, rollout_limit, num_rebuilds, variations}}`), 결과 `arena_experiment_result.json`(에피소드 raw), `plot_success_rates.py`. 교차곱 자동 없음 → 스크립트로 YAML 생성.

## Embodiment (`isaaclab_arena/embodiments/embodiment_base.py`)
- `EmbodimentBase(PlaceableAsset)`: name, default_arm_mode(ArmMode.DUAL_ARM), `__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode, collision_mode, spawn_cfg_addon)` → subclass가 scene_config(robot ArticulationCfg prim `{ENV_REGEX_NS}/Robot` + 센서), camera_config(`ArenaCameraCfg`, 기본 TiledCamera 변환, `set_use_tiled_camera(False)`), action_config, observation_config, event_config 채움.
- 관측: 카메라 켜면 `camera_obs` 그룹 `<camfield>_<dtype>`. 정책 그룹은 `PolicyCfg(ObsGroup)`.
- 행동 예: `JointPositionActionCfg(preserve_order=True, use_default_offset=False)`, `DifferentialInverseKinematicsActionCfg`, `BinaryJointPositionActionCfg`, 커스텀 `ActionTermCfg(class_type=...)` (G1 decoupled WBC 참고). 바퀴 베이스는 기존 예 없음 → 커스텀 (v, ω) 항.
- `get_ee_frame_transformer_names()` 양팔이면 override. `_configure_physics_backend(backend)`.
- 원격조작 장치 → `RetargeterRegistry` 키 `"{device}__{embodiment}"`: 새 embodiment 이름엔 retargeter 등록 필요.

## 씬
- `Background(name, usd_path, object_min_z, prim_path=None, initial_pose, reset_nested_physics=True)`; 기본 prim `{ENV_REGEX_NS}/<name>` → 절대 경로(`/World/...`)를 주면 전역 1회 스폰(월드 좌표). `reset_nested_physics=False` 고려. Light: `"light"`(DomeLight `/World/Light`).
- `DeformableObject(name, spawner_cfg, prim_path, initial_pose: Pose|PosePerEnv)`; backend는 deformable_props 타입으로 추론. 리셋은 `set_deformable_object_pose[_per_env]`.

## 물리·주기
- 기본 `sim.dt=1/120, render_interval=2, decimation=8`(15 Hz). 빌더가 `sim.physics`를 덮어씀 → `env_cfg_callback`에서만 dt/decimation/physics 변경(백엔드 타입 유지).
- `ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs, env_spacing, presets, record_trajectories, ...))` → `compose_manager_cfg()` → `make_registered()`. 시나리오 InteractiveSceneCfg(replicate_physics=False).

## Task
- `TaskBase(episode_length_s, task_description)`: `get_scene_cfg`, `get_termination_cfg -> TaskTerminationCfg(timeout_s, success=[CompletionCriteria(name, predicate_sequence=[f(env)->Bool(num_envs)], score)], failures={name: TerminationTermCfg}, subtasks_are_sequential)`, `get_events_cfg`, `get_mimic_env_cfg(arm_mode)`(NotImplementedError 가능), `get_metrics -> [MetricBase]`, `get_observation_cfg`, `get_viewer_cfg`.
- predicate_sequence: 순차·래칭(포인터가 한 스텝에 1칸, 통과 후 재검사 안 함), 가중치 `(pred, w)` 튜플 → `ProgressState.overall_score`가 부분 점수. `env.progress_tracker.get_state()`, `env.extras["progress_tracking"]`, JSONL `"progress"`. 하위 언어 라벨 개념 없음 → 직접 관리.
- `CompositeTaskBase(subtasks, subtasks_are_sequential)`.
- 지표: `MetricBase`(recorder term + compute func), `SuccessRateMetric`. HDF5 경유. 부분 점수는 progress overall_score 사용.
- 변형체 정점: `env.arena_world.get_nodal_positions_w(key)` 또는 `env.scene[name].data.nodal_pos_w.torch`. predicates `tasks/predicates/{spatial,object_settling}.py`.
- 테스트: `run_function_with_persistent_simulation_app(fn, headless=True)` (`isaaclab_arena/tests/utils/persistent_simulation_app.py`), 참고 `tests/test_deformable_object.py`.
- 기록: `record_demos.py --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback --task <env> --arena_teleop_device keyboard`.

## 원격 정책
- `isaaclab_arena/policy/remote_policy_base.py`: `EmbodimentAdapter(action_dim, extract(obs, env_id), pack_request(extracted, lang))`, `RemoteChunkReplayPolicy(config, adapter, open_loop_horizon)` 믹스인 — cfg 필드 `policy_device, remote_host, remote_port, ping_interval, ping_timeout`; 응답 키 `server_response_actions_key`("actions"), chunk (H≥horizon, action_dim). 모듈 수준에서 `openpi_client` import(이미 dx_arena에 설치).
- 전송: websocket(sync client) + msgpack_numpy(`{b"__ndarray__":True, b"data", b"dtype", b"shape"}`), 서버는 연결 시 metadata 먼저 전송, 이후 obs→{"actions": (H,D)}. 오류는 text frame. 서버 측 코드 없음 → dx_policy에 직접 작성(websockets≥12, msgpack). 템플릿 `isaaclab_arena_cosmos/policy/cosmos_remote_policy.py`, 세션 리셋 예 `isaaclab_arena_dreamzero`.
- 리플레이: `ReplayActionPolicyCfg(replay_file_path, device, episode_name)` HDF5 `data/<ep>/actions`.
- 영상: `--record_camera_video` → `<prefix>-env<N>-<cam>-episode-<E>.mp4`.
