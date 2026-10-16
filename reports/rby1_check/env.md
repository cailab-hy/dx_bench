# T0 환경 기록

| 항목 | 값 |
|---|---|
| Host OS | Linux-6.8.0-138-generic-x86_64-with-glibc2.35 |
| Python | 3.12.15 (/home/cai/miniforge3/envs/dx_arena/bin/python) |
| GPU / driver / VRAM | NVIDIA RTX 6000 Ada Generation, 580.178.04, 49140 MiB |
| CUDA (driver) | CUDA Version: 13.0 |
| CUDA (torch build) | 12.8 (available=True) |
| isaacsim | 6.1.0.0 |
| Isaac Lab | Isaac Lab 3.0 (isaaclab 17.0.2) |
| isaaclab_physx | 6.0.0 |
| isaaclab_newton | 5.4.1 |
| isaaclab_arena | 0.3.0 |
| torch | 2.11.0+cu128 |
| numpy | 2.3.1 |
| warp-lang | 1.16.0 |
| newton | 1.5.2 |
| lerobot (must be absent in dx_arena) | (not installed) |

## 커밋

| 레포 | 커밋 |
|---|---|
| IsaacLab-Arena | `c7edf3a9d5` (remotes/origin/release/0.3.1) |
| Isaac Lab (Arena submodule) | `91a0906f93` (tags/v3.0.0-EA-1-g91a0906f9) |
| rby1-sim-isaac | `ba5249629d` (heads/main) |

물리 백엔드: PhysX (`isaaclab_physx`) 고정. Newton 비교는 1단계에서 별도로 진행.
