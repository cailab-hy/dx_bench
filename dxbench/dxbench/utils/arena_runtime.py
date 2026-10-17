"""Launch Kit + build an Arena env the same way policy_runner.py does, for dxbench scripts.

    from dxbench.utils.arena_runtime import ArenaRuntime
    rt = ArenaRuntime(add_args=lambda p: p.add_argument("--test", default="idle"))
    env = rt.make_env()          # parses sys.argv: [script flags] --external_environment_class_path ... <env> [env flags]
    ...
    rt.close()
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Callable

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")


class ArenaRuntime:
    def __init__(self, add_args: Callable[[argparse.ArgumentParser], None] | None = None, argv: list[str] | None = None):
        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

        self._argv = argv
        self.parser = get_isaaclab_arena_cli_parser()
        if add_args:
            add_args(self.parser)
        args, _ = self.parser.parse_known_args(argv)
        self._ctx = SimulationAppContext(args)
        self._ctx.__enter__()
        self.args = args
        self.env = None

    def make_env(self, render_mode: str | None = None):
        from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

        parser = get_isaaclab_arena_environments_cli_parser(self.parser)
        self.args, overrides = parser.parse_known_args(self._argv)
        builder = get_arena_builder_from_cli(self.args, hydra_overrides=overrides)
        # Arena names the recorder HDF5 "<base>_<YYYYmmdd_HHMMSS>_rank0": processes started in the same second
        # (parallel benchmark / demo collection) collide on its file lock -> one file per process
        if getattr(builder, "cfg", None) is not None and getattr(builder.cfg, "recorder_dataset_filename", "x") is None:
            builder.cfg.recorder_dataset_filename = f"arena_dataset_pid{os.getpid()}"
        self.env = builder.make_registered(render_mode=render_mode) if render_mode else builder.make_registered()
        return self.env

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self._ctx.__exit__(None, None, None)
