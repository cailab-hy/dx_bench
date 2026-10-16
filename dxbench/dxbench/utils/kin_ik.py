"""Kinematic IK for the RB-Y1 arms on an Isaac Lab Articulation (teleports joints, no physics step).

Damped least squares on TCP position + orientation with a backtracking line search; joint limits clamped.
Used for planning (oracle policies, posture search, cloth spike). Quaternions from Isaac Lab are (x, y, z, w).
"""
from __future__ import annotations

import math

import torch

TCP_OFFSET = (0.0, 0.0, -0.1045)


def T(x):
    import warp as wp

    if hasattr(x, "torch"):
        return x.torch
    return wp.to_torch(x) if isinstance(x, wp.array) else x


def quat_xyzw_to_mat(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.tolist()
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], device=q.device)


def rot_z(a: float, device="cpu") -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=device)


class ArmIK:
    def __init__(self, robot, sim, scene, side: str = "right", env_id: int = 0):
        self.r, self.sim, self.scene, self.e = robot, sim, scene, env_id
        self.names = list(robot.joint_names)
        self.side = side
        ids, _ = robot.find_joints([f"{side}_arm_{i}" for i in range(7)], preserve_order=True)
        self.arm = list(ids.tolist() if hasattr(ids, "tolist") else ids)
        self.bi = robot.body_names.index(f"ee_{side}")
        self.dev = T(robot.data.joint_pos).device

    def fk_set(self, q: torch.Tensor):
        self.r.write_joint_position_to_sim_index(position=q)
        if hasattr(self.sim, "forward"):
            self.sim.forward()
        self.scene.update(0.0)

    def tcp(self):
        p = T(self.r.data.body_link_pos_w)[self.e, self.bi]
        R = quat_xyzw_to_mat(T(self.r.data.body_link_quat_w)[self.e, self.bi])
        return p + R @ torch.tensor(TCP_OFFSET, device=self.dev), R, p

    def _err(self, p_des, R_des, w_rot):
        p_tcp, Rc, p_ee = self.tcp()
        ep = p_des - p_tcp
        Re = R_des @ Rc.T
        ang = math.acos(max(-1.0, min(1.0, (float(torch.trace(Re)) - 1) / 2)))
        if ang < 1e-6:
            er = torch.zeros(3, device=self.dev)
        elif ang > math.pi - 1e-3:
            er = torch.tensor([0.0, 0.0, ang], device=self.dev)
        else:
            ax = torch.tensor([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]], device=self.dev)
            er = ax / (2 * math.sin(ang)) * ang
        return ep, er, ang, float(ep.norm()) + w_rot * ang, p_tcp, p_ee

    def solve(self, p_des, R_des, q0: torch.Tensor | None = None, iters: int = 400, w_rot: float = 0.3):
        """Returns (q_full (num_envs, J) with the arm solved, pos_err, rot_err). Leaves the robot at the solution."""
        p_des = torch.as_tensor(p_des, dtype=torch.float32, device=self.dev)
        R_des = R_des.to(self.dev)
        q = (T(self.r.data.joint_pos) if q0 is None else q0).clone()
        jb = self.bi - (1 if self.r.is_fixed_base else 0)
        lims = T(self.r.data.joint_pos_limits)[self.e, self.arm]
        lo, hi = lims[:, 0], lims[:, 1]
        self.fk_set(q)
        ep, er, ang, cost, p_tcp, p_ee = self._err(p_des, R_des, w_rot)
        lam = 0.05
        W = torch.tensor([1, 1, 1, w_rot, w_rot, w_rot], device=self.dev).view(6, 1)
        for _ in range(iters):
            if float(ep.norm()) < 1e-4 and ang < 1e-3:
                break
            J = T(self.r.data.body_link_jacobian_w)[self.e, jb]
            off = J.shape[-1] - len(self.names)
            Ja = J[:, [off + i for i in self.arm]]
            r = p_tcp - p_ee
            rx = torch.tensor([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]], device=self.dev)
            Jt = W * torch.cat([Ja[:3] - rx @ Ja[3:], Ja[3:]], dim=0)
            e = W * torch.cat([ep, er]).unsqueeze(-1)
            dq = (Jt.T @ torch.linalg.solve(Jt @ Jt.T + lam ** 2 * torch.eye(6, device=self.dev), e)).squeeze(-1)
            step, accepted = 1.0, False
            while step > 1e-3:
                qn = q.clone()
                qn[self.e, self.arm] = torch.max(torch.min(q[self.e, self.arm] + step * dq.clamp(-0.3, 0.3), hi), lo)
                self.fk_set(qn)
                ep2, er2, ang2, cost2, p2, pe2 = self._err(p_des, R_des, w_rot)
                if cost2 < cost:
                    q, ep, er, ang, cost, p_tcp, p_ee, accepted = qn, ep2, er2, ang2, cost2, p2, pe2, True
                    break
                step *= 0.5
            if not accepted:
                lam *= 2.0
                self.fk_set(q)
                if lam > 5.0:
                    break
            else:
                lam = max(0.01, lam * 0.7)
        self.fk_set(q)
        return q, float(ep.norm()), ang
