# T2 URDF ↔ USD 대조

- URDF: `/home/cai/dx_bench/tools/rby1_check/_ext/rby1-sdk/models/rby1a/urdf/model_v1.2.urdf` (root link `base`)
- USD(몸체 24 DOF): T1 `model_a` · 그리퍼: T1 `rbgripper_d405_collision`
- 단위: 각도 rad, 속도 rad/s 또는 m/s (USD의 deg 값을 환산), effort N·m 또는 N

## 차이 나는 항목만

| 관절 | 항목 | URDF | USD |
|---|---|---|---|
| right_wheel | effort↔maxForce | — | 100 |
| right_wheel | velocity↔maxJointVelocity | 15.71 | 17.45 |
| left_wheel | effort↔maxForce | — | 100 |
| left_wheel | velocity↔maxJointVelocity | 15.71 | 17.45 |
| torso_0 | effort↔maxForce | 270 | 841 |
| torso_1 | effort↔maxForce | 270 | 841 |
| torso_2 | effort↔maxForce | 270 | 841 |
| torso_3 | effort↔maxForce | 120 | 369 |
| torso_4 | effort↔maxForce | 120 | 369 |
| torso_5 | effort↔maxForce | 120 | 369 |
| right_arm_0 | effort↔maxForce | 70 | 143 |
| right_arm_1 | upper | 0.01745 | 0 |
| right_arm_1 | effort↔maxForce | 70 | 143 |
| right_arm_2 | effort↔maxForce | 70 | 143 |
| right_arm_3 | upper | 0.01745 | 0 |
| right_arm_3 | effort↔maxForce | 40 | 70 |
| right_arm_4 | effort↔maxForce | 10 | 70 |
| right_arm_5 | effort↔maxForce | 10 | 70 |
| right_arm_6 | effort↔maxForce | 8 | 140 |
| right_arm_6 | velocity↔maxJointVelocity | 2.094 | 6.283 |
| left_arm_0 | effort↔maxForce | 70 | 143 |
| left_arm_1 | lower | -0.01745 | 0 |
| left_arm_1 | effort↔maxForce | 70 | 143 |
| left_arm_2 | effort↔maxForce | 70 | 143 |
| left_arm_3 | upper | 0.01745 | 0 |
| left_arm_3 | effort↔maxForce | 40 | 70 |
| left_arm_4 | effort↔maxForce | 10 | 70 |
| left_arm_5 | effort↔maxForce | 10 | 70 |
| left_arm_6 | effort↔maxForce | 8 | 140 |
| left_arm_6 | velocity↔maxJointVelocity | 2.094 | 6.283 |
| head_0 | effort↔maxForce | 1000 | 70 |
| head_0 | velocity↔maxJointVelocity | 3.14 | 3.142 |
| head_1 | effort↔maxForce | 1000 | 70 |
| gripper_finger_l1 | effort↔maxForce | 1000 | 75 |
| gripper_finger_l1 | velocity↔maxJointVelocity | 0.05 | 0.04 |
| gripper_finger_l2 | velocity↔maxJointVelocity | 0.05 | 0.04 |
| gripper_finger_r1 | effort↔maxForce | 1000 | 75 |
| gripper_finger_r1 | velocity↔maxJointVelocity | 0.05 | 0.04 |
| gripper_finger_r2 | velocity↔maxJointVelocity | 0.05 | 0.04 |

## 영 자세 기구학 대조 (관절 원점 오차, 축 내적)

축 내적 1 = 같은 방향, −1 = 반대(양의 회전 방향이 반대), 0 근처 = 다른 축.

| 관절 | 원점 오차 mm | 축 내적 | USD axis 토큰 |
|---|---|---|---|
| right_wheel | 0.00 | +1.0000 | `Y` |
| left_wheel | 0.00 | +1.0000 | `Y` |
| torso_0 | 0.00 | +1.0000 | `X` |
| torso_1 | 0.00 | +1.0000 | `Y` |
| torso_2 | 0.00 | +1.0000 | `Y` |
| torso_3 | 0.00 | +1.0000 | `Y` |
| torso_4 | 0.00 | +1.0000 | `X` |
| torso_5 | 0.00 | +1.0000 | `Z` |
| right_arm_0 | 0.00 | +1.0000 | `` |
| right_arm_1 | 0.00 | +1.0000 | `X` |
| right_arm_2 | 0.00 | +1.0000 | `Z` |
| right_arm_3 | 0.00 | +1.0000 | `Y` |
| right_arm_4 | 0.00 | +1.0000 | `Z` |
| right_arm_5 | 0.00 | +1.0000 | `Y` |
| right_arm_6 | 0.00 | +1.0000 | `Z` |
| left_arm_0 | 0.00 | +1.0000 | `` |
| left_arm_1 | 0.00 | +1.0000 | `X` |
| left_arm_2 | 0.00 | +1.0000 | `Z` |
| left_arm_3 | 0.00 | +1.0000 | `Y` |
| left_arm_4 | 0.00 | +1.0000 | `Z` |
| left_arm_5 | 0.00 | +1.0000 | `Y` |
| left_arm_6 | 0.00 | +1.0000 | `Z` |
| head_0 | 0.00 | +1.0000 | `Z` |
| head_1 | 0.00 | +1.0000 | `Y` |
| gripper_finger_l1 ⚠ | 16.00 | +1.0000 | `X` |
| gripper_finger_l2 ⚠ | 16.00 | +1.0000 | `X` |
| gripper_finger_r1 ⚠ | 16.00 | +1.0000 | `X` |
| gripper_finger_r2 ⚠ | 16.00 | +1.0000 | `X` |
| head_base ⚠ | 0.00 | +0.0000 | `None` |

## 전체 값

| 관절 | URDF 종류 | USD 종류 | 하한 U/S | 상한 U/S | effort U / maxForce S | vel U/S | mimic |
|---|---|---|---|---|---|---|---|
| right_wheel | continuous | RevoluteJoint | — / — | — / — | — / 100 | 15.71 / 17.45 |  |
| left_wheel | continuous | RevoluteJoint | — / — | — / — | — / 100 | 15.71 / 17.45 |  |
| torso_0 | revolute | RevoluteJoint | -0.2618 / -0.2618 | 0.2618 / 0.2618 | 270 / 841 | 2.094 / 2.094 |  |
| torso_1 | revolute | RevoluteJoint | -0.5236 / -0.5236 | 1.571 / 1.571 | 270 / 841 | 2.094 / 2.094 |  |
| torso_2 | revolute | RevoluteJoint | -2.618 / -2.618 | 1.571 / 1.571 | 270 / 841 | 2.094 / 2.094 |  |
| torso_3 | revolute | RevoluteJoint | -0.7854 / -0.7854 | 1.571 / 1.571 | 120 / 369 | 3.142 / 3.142 |  |
| torso_4 | revolute | RevoluteJoint | -0.5236 / -0.5236 | 0.5236 / 0.5236 | 120 / 369 | 3.142 / 3.142 |  |
| torso_5 | revolute | RevoluteJoint | -2.356 / -2.356 | 2.356 / 2.356 | 120 / 369 | 3.142 / 3.142 |  |
| right_arm_0 | revolute | RevoluteJoint | -3.142 / -3.142 | 3.142 / 3.142 | 70 / 143 | 3.142 / 3.142 |  |
| right_arm_1 | revolute | RevoluteJoint | -3.142 / -3.142 | 0.01745 / 0 | 70 / 143 | 3.142 / 3.142 |  |
| right_arm_2 | revolute | RevoluteJoint | -3.142 / -3.142 | 3.142 / 3.142 | 70 / 143 | 3.142 / 3.142 |  |
| right_arm_3 | revolute | RevoluteJoint | -2.618 / -2.618 | 0.01745 / 0 | 40 / 70 | 3.142 / 3.142 |  |
| right_arm_4 | revolute | RevoluteJoint | -3.142 / -3.142 | 3.142 / 3.142 | 10 / 70 | 6.283 / 6.283 |  |
| right_arm_5 | revolute | RevoluteJoint | -1.571 / -1.571 | 1.92 / 1.92 | 10 / 70 | 6.283 / 6.283 |  |
| right_arm_6 | revolute | RevoluteJoint | -2.705 / -2.705 | 2.705 / 2.705 | 8 / 140 | 2.094 / 6.283 |  |
| left_arm_0 | revolute | RevoluteJoint | -3.142 / -3.142 | 3.142 / 3.142 | 70 / 143 | 3.142 / 3.142 |  |
| left_arm_1 | revolute | RevoluteJoint | -0.01745 / 0 | 3.142 / 3.142 | 70 / 143 | 3.142 / 3.142 |  |
| left_arm_2 | revolute | RevoluteJoint | -3.142 / -3.142 | 3.142 / 3.142 | 70 / 143 | 3.142 / 3.142 |  |
| left_arm_3 | revolute | RevoluteJoint | -2.618 / -2.618 | 0.01745 / 0 | 40 / 70 | 3.142 / 3.142 |  |
| left_arm_4 | revolute | RevoluteJoint | -3.142 / -3.142 | 3.142 / 3.142 | 10 / 70 | 6.283 / 6.283 |  |
| left_arm_5 | revolute | RevoluteJoint | -1.571 / -1.571 | 1.92 / 1.92 | 10 / 70 | 6.283 / 6.283 |  |
| left_arm_6 | revolute | RevoluteJoint | -2.705 / -2.705 | 2.705 / 2.705 | 8 / 140 | 2.094 / 6.283 |  |
| head_0 | revolute | RevoluteJoint | -1.57 / -1.571 | 1.57 / 1.571 | 1000 / 70 | 3.14 / 3.142 |  |
| head_1 | revolute | RevoluteJoint | -1.57 / -1.571 | 1.57 / 1.571 | 1000 / 70 | 3.14 / 3.14 |  |
| gripper_finger_l1 | prismatic | PrismaticJoint | -0.05 / -0.05 | 0 / 0 | 1000 / 75 | 0.05 / 0.04 |  |
| gripper_finger_l2 | prismatic | PrismaticJoint | 0 / 0 | 0.05 / 0.05 | 1000 / 1000 | 0.05 / 0.04 |  |
| gripper_finger_r1 | prismatic | PrismaticJoint | -0.05 / -0.05 | 0 / 0 | 1000 / 75 | 0.05 / 0.04 |  |
| gripper_finger_r2 | prismatic | PrismaticJoint | 0 / 0 | 0.05 / 0.05 | 1000 / 1000 | 0.05 / 0.04 |  |
| head_base | fixed | FixedJoint | — / — | — / — | — / — | — / — |  |
