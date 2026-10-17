# Vendored from cailab-hy/lightwheel_rby1_transfer scripts/rby1_format.py @1178454 (same lab, internal use; DECISIONS D-2).
"""Mapping between the simulator's 18-D joint layout and the real RB-Y1 LeRobot format.

Reference: rainbowrobotics/icra_0526_compound_rel (LeRobot v3, robot_type "rby1", 15 fps):
  state/action (16): right_arm_0..6, left_arm_0..6, right_gripper_0, left_gripper_0
  - arm joints: absolute joint positions in rad; action = absolute target (a_t ~ s_{t+1});
    sign and zero match the simulator (the real episodes start at this repo's READY pose).
  - gripper: normalised opening, 1 = open, 0 = closed; actions are (mostly) binary 0/1.
  images: front 480x640 (head camera), right / left 640x480 portrait wrist cameras with the
  fingers at the bottom of the image; AV1 video.
"""
import numpy as np

FPS = 15
ROBOT_TYPE = "rby1"
NAMES = [f"right_arm_{i}" for i in range(7)] + [f"left_arm_{i}" for i in range(7)] + ["right_gripper_0", "left_gripper_0"]
FINGER_OPEN_M = 0.045   # simulator finger displacement when open (each finger; l1/r1 negative, l2/r2 positive)
# simulator camera -> RB-Y1 image key, and the portrait (height, width) the wrist cameras need
CAMERA_KEYS = {"first_person": "front", "right_hand": "right", "left_hand": "left"}
IMAGE_SHAPES = {"front": (480, 640), "right": (640, 480), "left": (640, 480)}
# collector render settings for the rby1 profile: (width, height), rotate 180 deg
RENDER = {"first_person": ((640, 480), False), "right_hand": ((480, 640), False), "left_hand": ((480, 640), True)}


def _idx(joint_names):
    j = list(joint_names)
    arms = [j.index(f"right_arm_{i}") for i in range(7)] + [j.index(f"left_arm_{i}") for i in range(7)]
    fingers = {s: (j.index(f"gripper_finger_{c}1"), j.index(f"gripper_finger_{c}2")) for s, c in (("right", "r"), ("left", "l"))}
    return arms, fingers


def opening(q, fingers, side):
    a, b = fingers[side]
    return np.clip(((q[..., b] - q[..., a]) / 2) / FINGER_OPEN_M, 0.0, 1.0)


def state16(q18, joint_names):
    """(T, 18) measured simulator joints -> (T, 16) RB-Y1 state."""
    arms, fingers = _idx(joint_names)
    q18 = np.asarray(q18, dtype=np.float64)
    return np.concatenate([q18[..., arms], opening(q18, fingers, "right")[..., None], opening(q18, fingers, "left")[..., None]], -1).astype(np.float32)


def gripper_commands(target_opening, eps=1e-6):
    """Recover the binary open(1)/close(0) command from a slew-limited finger target sequence."""
    cmd = np.empty(len(target_opening), dtype=np.float32)
    c = 1.0 if target_opening[0] > 0.5 else 0.0
    for t, v in enumerate(target_opening):
        if t > 0 and v < target_opening[t - 1] - eps:
            c = 0.0
        elif t > 0 and v > target_opening[t - 1] + eps:
            c = 1.0
        elif v >= 1.0 - 1e-4:
            c = 1.0
        elif v <= 1e-4:
            c = 0.0
        cmd[t] = c
    return cmd


def action16(a18, joint_names):
    """(T, 18) simulator joint targets -> (T, 16) RB-Y1 action (arm targets + binary gripper commands)."""
    arms, fingers = _idx(joint_names)
    a18 = np.asarray(a18, dtype=np.float64)
    grip = [gripper_commands(opening(a18, fingers, s)) for s in ("right", "left")]
    return np.concatenate([a18[..., arms], np.stack(grip, -1)], -1).astype(np.float32)


def to_sim18(v16, joint_names, template=None):
    """Inverse mapping (for replay of RB-Y1-format data): gripper opening -> symmetric finger positions."""
    arms, fingers = _idx(joint_names)
    v16 = np.asarray(v16, dtype=np.float64)
    out = np.zeros(v16.shape[:-1] + (len(joint_names),)) if template is None else np.array(template, dtype=np.float64)
    out[..., arms] = v16[..., :14]
    for k, s in ((14, "right"), (15, "left")):
        a, b = fingers[s]
        out[..., a] = -v16[..., k] * FINGER_OPEN_M
        out[..., b] = v16[..., k] * FINGER_OPEN_M
    return out


def resample_indices(n_raw, raw_fps, fps=FPS):
    """Frame indices of a raw sequence sampled at `fps`, and for each the last raw step before the next sample
    (whose joint target is the command in force when that next sample is taken)."""
    if raw_fps == fps:
        k = np.arange(n_raw)
        return k, k
    t = np.arange(int(np.floor((n_raw - 1) * fps / raw_fps)) + 1) / fps
    obs = np.minimum(np.round(t * raw_fps).astype(int), n_raw - 1)
    act = np.minimum(np.maximum(np.ceil((t + 1 / fps) * raw_fps).astype(int) - 1, obs), n_raw - 1)
    return obs, act


def adapt_image(img, camera, rendered_for_rby1):
    """Raw simulator frame -> RB-Y1 image. Frames recorded with the rby1 profile are already correct.
    Older landscape wrist frames are centre-cropped to 3:4 portrait, resized, and the left one flipped."""
    if rendered_for_rby1 or camera == "first_person":
        return img
    from PIL import Image
    h, w = img.shape[:2]
    cw = int(round(h * 3 / 4))
    x0 = (w - cw) // 2
    im = Image.fromarray(img[:, x0:x0 + cw]).resize((480, 640), Image.BICUBIC)
    if camera == "left_hand":
        im = im.rotate(180)
    return np.asarray(im)
