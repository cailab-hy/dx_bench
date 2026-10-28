"""Skip Isaac Lab's second observation pass in a control step when nothing changed in between (2026-10-28).

With any recorder term active (Arena always adds its metrics / progress recorders) `ManagerBasedRLEnv.step` computes
the observations twice: for the recorders right after the physics steps, and again after the resets. Without a reset,
interval event or command term in between, the state is the same, so the second pass returns the first result
(~3 ms of the ~99 ms teleop control step at 120 Hz; the cameras are read once either way).
"""
from __future__ import annotations


def dedupe_observations(env) -> bool:
    """Install on `env` (gym env or the unwrapped Isaac Lab env). Returns False (not installed) when the observation
    groups keep a history or the env has interval events / command terms."""
    U = getattr(env, "unwrapped", env)
    om = U.observation_manager
    if any(len(v) for v in getattr(om, "_group_obs_term_history_buffer", {}).values()):
        return False
    if "interval" in U.event_manager.available_modes or getattr(U.command_manager, "active_terms", None):
        return False
    compute, reset_idx = om.compute, U._reset_idx
    state = {"resets": 0, "key": None, "obs": None}

    def _reset_idx(*a, **k):
        state["resets"] += 1
        return reset_idx(*a, **k)

    def _compute(update_history: bool = False):
        key = (U._sim_step_counter, state["resets"])
        if key != state["key"]:
            state["obs"], state["key"] = compute(update_history=update_history), key
        return state["obs"]

    U._reset_idx = _reset_idx
    om.compute = _compute
    return True
