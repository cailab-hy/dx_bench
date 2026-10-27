"""pad_grasp wheel-trap mask without the simulator: the vectorised mask equals the earlier per-env / per-wheel loop."""
import torch

from dxbench.embodiments.mdp.pad_grasp import wheel_patch_mask


def _loop_mask(nodes, wp_, f, l_, half_len, half_width, z_max):  # the implementation before 2026-10-27
    low = nodes[..., 2] <= z_max
    out = []
    for e in range(nodes.shape[0]):
        m = torch.zeros(nodes.shape[1], dtype=torch.bool)
        for k in range(wp_.shape[1]):
            d = nodes[e, :, :2] - wp_[e, k, :2]
            m |= ((d @ f[e]).abs() <= half_len) & ((d @ l_[e]).abs() <= half_width)
        out.append(m & low[e])
    return torch.stack(out)


def test_wheel_patch_mask_matches_loop():
    g = torch.Generator().manual_seed(0)
    N, V = 3, 4000
    yaw = torch.rand(N, generator=g) * 6.28
    f = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
    l_ = torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1)
    wp_ = torch.rand(N, 2, 2, generator=g) * 0.6 - 0.3
    nodes = torch.cat([torch.rand(N, V, 2, generator=g) * 0.8 - 0.4, torch.rand(N, V, 1, generator=g) * 0.08], -1)
    new = wheel_patch_mask(nodes, wp_, f, l_, 0.05, 0.035, 0.04)
    old = _loop_mask(nodes, wp_, f, l_, 0.05, 0.035, 0.04)
    assert new.shape == (N, V) and new.any()
    assert torch.equal(new, old)


def test_wheel_patch_mask_far_cloth_is_empty():
    nodes = torch.tensor([[[2.0, 2.0, 0.0], [0.0, 0.0, 0.10]]])  # far away / above the 4 cm band under a wheel
    wp_ = torch.zeros(1, 2, 2)
    f, l_ = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    assert not wheel_patch_mask(nodes, wp_, f, l_, 0.05, 0.035, 0.04).any()
