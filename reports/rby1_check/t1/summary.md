# T1 USD 정적 검사

## model_a

- 파일: `/home/cai/dx_bench/submodules/rby1-sim-isaac/assets/model_v_1_2_a.usd`
- defaultPrim `/RBY1_A_v1_2` · metersPerUnit 1.0 · upAxis Z · kgPerUnit 1.0
- articulation root 1개: `/RBY1_A_v1_2`
- 관절 25개 (dof 24), 링크 26개, 충돌체 23개, 카메라 0개
- 사용 레이어 2개
- 충돌체 근사: Capsule/None×17, Cylinder/None×6
- articulation 속성 `/RBY1_A_v1_2`: {'physxArticulation:enabledSelfCollisions': False}

| 관절 | 종류 | 축 | 하한 | 상한 | 드라이브 | K | D | maxForce | 기타 physx |
|---|---|---|---|---|---|---|---|---|---|
| left_wheel | RevoluteJoint | Y | None | None | angular/force | 100.0 | 10.0 | 100.0 | {'maxActuatorVelocity': 900.0, 'maxJointVelocity': 1000.0} |
| right_wheel | RevoluteJoint | Y | None | None | angular/force | 100.0 | 10.0 | 100.0 | {'maxActuatorVelocity': 900.0, 'maxJointVelocity': 1000.0} |
| torso_0 | RevoluteJoint | X | -15.0 | 15.0 | angular/force | 100.0 | 10.0 | 841.0 | {'maxJointVelocity': 120.0} |
| torso_1 | RevoluteJoint | Y | -30.0 | 90.0 | angular/force | 100.0 | 10.0 | 841.0 | {'maxJointVelocity': 120.0} |
| torso_2 | RevoluteJoint | Y | -150.0 | 90.0 | angular/force | 100.0 | 10.0 | 841.0 | {'maxJointVelocity': 120.0} |
| torso_3 | RevoluteJoint | Y | -45.0 | 90.0 | angular/force | 100.0 | 10.0 | 369.0 | {'maxJointVelocity': 180.0} |
| torso_4 | RevoluteJoint | X | -30.0 | 30.0 | angular/force | 100.0 | 10.0 | 369.0 | {'maxJointVelocity': 180.0} |
| torso_5 | RevoluteJoint | Z | -135.0 | 135.0 | angular/force | 100.0 | 10.0 | 369.0 | {'maxJointVelocity': 180.0} |
| head_0 | RevoluteJoint | Z | -90.0 | 90.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| head_1 | RevoluteJoint | Y | -90.0 | 90.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 179.9087371826172} |
| left_arm_0 | RevoluteJoint |  | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_1 | RevoluteJoint | X | 0.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_2 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_3 | RevoluteJoint | Y | -150.0 | 0.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| left_arm_4 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| left_arm_5 | RevoluteJoint | Y | -90.0 | 110.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| left_arm_6 | RevoluteJoint | Z | -155.0 | 155.0 | angular/force | 100.0 | 10.0 | 140.0 | {'maxJointVelocity': 360.0} |
| right_arm_0 | RevoluteJoint |  | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_1 | RevoluteJoint | X | -180.0 | 0.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_2 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_3 | RevoluteJoint | Y | -150.0 | 0.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| right_arm_4 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| right_arm_5 | RevoluteJoint | Y | -90.0 | 110.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| right_arm_6 | RevoluteJoint | Z | -155.0 | 155.0 | angular/force | 100.0 | 10.0 | 140.0 | {'maxJointVelocity': 360.0} |

## rbgripper_d405

- 파일: `/home/cai/dx_bench/submodules/rby1-sim-isaac/assets/generated/rby1_v1_2_a_rbgripper_d405.usd`
- defaultPrim `/RBY1` · metersPerUnit 1.0 · upAxis Z · kgPerUnit 1.0
- articulation root 1개: `/RBY1`
- 관절 32개 (dof 18), 링크 32개, 충돌체 23개, 카메라 3개
- 사용 레이어 9개
- 충돌체 근사: Capsule/None×17, Cylinder/None×6
- articulation 속성 `/RBY1`: {'physxArticulation:enabledSelfCollisions': False}

| 관절 | 종류 | 축 | 하한 | 상한 | 드라이브 | K | D | maxForce | 기타 physx |
|---|---|---|---|---|---|---|---|---|---|
| left_arm_0 | RevoluteJoint |  | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_1 | RevoluteJoint | X | 0.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_2 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_3 | RevoluteJoint | Y | -150.0 | 0.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| left_arm_4 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| left_arm_5 | RevoluteJoint | Y | -90.0 | 110.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| left_arm_6 | RevoluteJoint | Z | -155.0 | 155.0 | angular/force | 100.0 | 10.0 | 140.0 | {'maxJointVelocity': 360.0} |
| right_arm_0 | RevoluteJoint |  | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_1 | RevoluteJoint | X | -180.0 | 0.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_2 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_3 | RevoluteJoint | Y | -150.0 | 0.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| right_arm_4 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| right_arm_5 | RevoluteJoint | Y | -90.0 | 110.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| right_arm_6 | RevoluteJoint | Z | -155.0 | 155.0 | angular/force | 100.0 | 10.0 | 140.0 | {'maxJointVelocity': 360.0} |
| tool_left | FixedJoint |  |  |  | None/None | None | None | None | {} |
| tool_right | FixedJoint |  |  |  | None/None | None | None | None | {} |
| gripper_finger_l1 | PrismaticJoint | X | -0.05000000074505806 | 0.0 | linear/force | 100.0 | 1.0 | 75.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_l2 | PrismaticJoint | X | 0.0 | 0.05000000074505806 | linear/force | 0.30934953689575195 | 0.00012373982463032007 | 1000.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_r1 | PrismaticJoint | X | -0.05000000074505806 | 0.0 | linear/force | 100.0 | 1.0 | 75.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_r2 | PrismaticJoint | X | 0.0 | 0.05000000074505806 | linear/force | 0.30934953689575195 | 0.00012373982463032007 | 1000.0 | {'maxJointVelocity': 0.03999999910593033} |

| 카메라 | f(mm) | hAp | vAp | fx@640 | hFOV° | clip |
|---|---|---|---|---|---|---|
| `/RBY1/link_head_2/first_person_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |
| `/RBY1/left_gripper/ee_left/left_hand_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |
| `/RBY1/right_gripper/ee_right/right_hand_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |

## rbgripper_d405_collision

- 파일: `/home/cai/dx_bench/submodules/rby1-sim-isaac/assets/generated/rby1_v1_2_a_rbgripper_d405_collision.usd`
- defaultPrim `/RBY1` · metersPerUnit 1.0 · upAxis Z · kgPerUnit 1.0
- articulation root 1개: `/RBY1`
- 관절 32개 (dof 18), 링크 32개, 충돌체 27개, 카메라 3개
- 사용 레이어 10개
- 충돌체 근사: Capsule/None×17, Cylinder/None×6, Mesh/convexDecomposition×4
- articulation 속성 `/RBY1`: {'physxArticulation:enabledSelfCollisions': False}

| 관절 | 종류 | 축 | 하한 | 상한 | 드라이브 | K | D | maxForce | 기타 physx |
|---|---|---|---|---|---|---|---|---|---|
| left_arm_0 | RevoluteJoint |  | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_1 | RevoluteJoint | X | 0.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_2 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_3 | RevoluteJoint | Y | -150.0 | 0.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| left_arm_4 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| left_arm_5 | RevoluteJoint | Y | -90.0 | 110.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| left_arm_6 | RevoluteJoint | Z | -155.0 | 155.0 | angular/force | 100.0 | 10.0 | 140.0 | {'maxJointVelocity': 360.0} |
| right_arm_0 | RevoluteJoint |  | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_1 | RevoluteJoint | X | -180.0 | 0.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_2 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_3 | RevoluteJoint | Y | -150.0 | 0.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| right_arm_4 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| right_arm_5 | RevoluteJoint | Y | -90.0 | 110.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| right_arm_6 | RevoluteJoint | Z | -155.0 | 155.0 | angular/force | 100.0 | 10.0 | 140.0 | {'maxJointVelocity': 360.0} |
| tool_left | FixedJoint |  |  |  | None/None | None | None | None | {} |
| tool_right | FixedJoint |  |  |  | None/None | None | None | None | {} |
| gripper_finger_l1 | PrismaticJoint | X | -0.05000000074505806 | 0.0 | linear/force | 100.0 | 1.0 | 75.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_l2 | PrismaticJoint | X | 0.0 | 0.05000000074505806 | linear/force | 0.30934953689575195 | 0.00012373982463032007 | 1000.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_r1 | PrismaticJoint | X | -0.05000000074505806 | 0.0 | linear/force | 100.0 | 1.0 | 75.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_r2 | PrismaticJoint | X | 0.0 | 0.05000000074505806 | linear/force | 0.30934953689575195 | 0.00012373982463032007 | 1000.0 | {'maxJointVelocity': 0.03999999910593033} |

| 카메라 | f(mm) | hAp | vAp | fx@640 | hFOV° | clip |
|---|---|---|---|---|---|---|
| `/RBY1/link_head_2/first_person_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |
| `/RBY1/left_gripper/ee_left/left_hand_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |
| `/RBY1/right_gripper/ee_right/right_hand_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |

## fullbody_collision

- 파일: `/home/cai/dx_bench/submodules/rby1-sim-isaac/assets/generated/rby1_fullbody_collision.usd`
- defaultPrim `/RBY1` · metersPerUnit 1.0 · upAxis Z · kgPerUnit 1.0
- articulation root 1개: `/RBY1`
- 관절 32개 (dof 18), 링크 32개, 충돌체 27개, 카메라 3개
- 사용 레이어 11개
- 충돌체 근사: Capsule/None×15, Cylinder/None×6, Mesh/convexDecomposition×4, Mesh/convexHull×2
- articulation 속성 `/RBY1`: {'physxArticulation:enabledSelfCollisions': False}

| 관절 | 종류 | 축 | 하한 | 상한 | 드라이브 | K | D | maxForce | 기타 physx |
|---|---|---|---|---|---|---|---|---|---|
| left_arm_0 | RevoluteJoint |  | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_1 | RevoluteJoint | X | 0.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_2 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_3 | RevoluteJoint | Y | -150.0 | 0.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| left_arm_4 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| left_arm_5 | RevoluteJoint | Y | -90.0 | 110.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| left_arm_6 | RevoluteJoint | Z | -155.0 | 155.0 | angular/force | 100.0 | 10.0 | 140.0 | {'maxJointVelocity': 360.0} |
| right_arm_0 | RevoluteJoint |  | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_1 | RevoluteJoint | X | -180.0 | 0.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_2 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_3 | RevoluteJoint | Y | -150.0 | 0.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| right_arm_4 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| right_arm_5 | RevoluteJoint | Y | -90.0 | 110.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| right_arm_6 | RevoluteJoint | Z | -155.0 | 155.0 | angular/force | 100.0 | 10.0 | 140.0 | {'maxJointVelocity': 360.0} |
| tool_left | FixedJoint |  |  |  | None/None | None | None | None | {} |
| tool_right | FixedJoint |  |  |  | None/None | None | None | None | {} |
| gripper_finger_l1 | PrismaticJoint | X | -0.05000000074505806 | 0.0 | linear/force | 100.0 | 1.0 | 75.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_l2 | PrismaticJoint | X | 0.0 | 0.05000000074505806 | linear/force | 0.30934953689575195 | 0.00012373982463032007 | 1000.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_r1 | PrismaticJoint | X | -0.05000000074505806 | 0.0 | linear/force | 100.0 | 1.0 | 75.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_r2 | PrismaticJoint | X | 0.0 | 0.05000000074505806 | linear/force | 0.30934953689575195 | 0.00012373982463032007 | 1000.0 | {'maxJointVelocity': 0.03999999910593033} |

| 카메라 | f(mm) | hAp | vAp | fx@640 | hFOV° | clip |
|---|---|---|---|---|---|---|
| `/RBY1/link_head_2/first_person_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |
| `/RBY1/left_gripper/ee_left/left_hand_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |
| `/RBY1/right_gripper/ee_right/right_hand_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |

## gripper_left

- 파일: `/home/cai/dx_bench/submodules/rby1-sim-isaac/assets/gripper/rb_gripper/rb_gripper_left.usd`
- defaultPrim `/Gripper_Left` · metersPerUnit 1.0 · upAxis Z · kgPerUnit 1.0
- articulation root 1개: `/Gripper_Left`
- 관절 2개 (dof 2), 링크 3개, 충돌체 0개, 카메라 0개
- 사용 레이어 2개
- 충돌체 근사: 
- articulation 속성 `/Gripper_Left`: {'physxArticulation:enabledSelfCollisions': False}

| 관절 | 종류 | 축 | 하한 | 상한 | 드라이브 | K | D | maxForce | 기타 physx |
|---|---|---|---|---|---|---|---|---|---|
| gripper_finger_l1 | PrismaticJoint | X | -0.05000000074505806 | 0.0 | linear/force | 100.0 | 1.0 | 75.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_l2 | PrismaticJoint | X | 0.0 | 0.05000000074505806 | None/None | None | None | None | {'maxJointVelocity': 0.03999999910593033} |

## gripper_right

- 파일: `/home/cai/dx_bench/submodules/rby1-sim-isaac/assets/gripper/rb_gripper/rb_gripper_right.usd`
- defaultPrim `/Gripper_Right` · metersPerUnit 1.0 · upAxis Z · kgPerUnit 1.0
- articulation root 1개: `/Gripper_Right`
- 관절 2개 (dof 2), 링크 3개, 충돌체 0개, 카메라 0개
- 사용 레이어 2개
- 충돌체 근사: 
- articulation 속성 `/Gripper_Right`: {'physxArticulation:enabledSelfCollisions': False}

| 관절 | 종류 | 축 | 하한 | 상한 | 드라이브 | K | D | maxForce | 기타 physx |
|---|---|---|---|---|---|---|---|---|---|
| gripper_finger_r1 | PrismaticJoint | X | -0.05000000074505806 | 0.0 | linear/force | 100.0 | 1.0 | 75.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_r2 | PrismaticJoint | X | 0.0 | 0.05000000074505806 | None/None | None | None | None | {'maxJointVelocity': 0.03999999910593033} |

## override_24dof

- 파일: `/home/cai/dx_bench/tools/rby1_check/overrides/rby1_24dof.usda`
- defaultPrim `/RBY1` · metersPerUnit 1.0 · upAxis Z · kgPerUnit 1.0
- articulation root 1개: `/RBY1/base`
- 관절 31개 (dof 28), 링크 32개, 충돌체 27개, 카메라 3개
- 사용 레이어 11개
- 충돌체 근사: Capsule/None×17, Cylinder/None×6, Mesh/convexDecomposition×4
- articulation 속성 `/RBY1/base`: {'physxArticulation:enabledSelfCollisions': False}

| 관절 | 종류 | 축 | 하한 | 상한 | 드라이브 | K | D | maxForce | 기타 physx |
|---|---|---|---|---|---|---|---|---|---|
| left_wheel | RevoluteJoint | Y | None | None | angular/force | 100.0 | 10.0 | 100.0 | {'maxActuatorVelocity': 900.0, 'maxJointVelocity': 1000.0} |
| right_wheel | RevoluteJoint | Y | None | None | angular/force | 100.0 | 10.0 | 100.0 | {'maxActuatorVelocity': 900.0, 'maxJointVelocity': 1000.0} |
| torso_0 | RevoluteJoint | X | -15.0 | 15.0 | angular/force | 100.0 | 10.0 | 841.0 | {'maxJointVelocity': 120.0} |
| torso_1 | RevoluteJoint | Y | -30.0 | 90.0 | angular/force | 100.0 | 10.0 | 841.0 | {'maxJointVelocity': 120.0} |
| torso_2 | RevoluteJoint | Y | -150.0 | 90.0 | angular/force | 100.0 | 10.0 | 841.0 | {'maxJointVelocity': 120.0} |
| torso_3 | RevoluteJoint | Y | -45.0 | 90.0 | angular/force | 100.0 | 10.0 | 369.0 | {'maxJointVelocity': 180.0} |
| torso_4 | RevoluteJoint | X | -30.0 | 30.0 | angular/force | 100.0 | 10.0 | 369.0 | {'maxJointVelocity': 180.0} |
| torso_5 | RevoluteJoint | Z | -135.0 | 135.0 | angular/force | 100.0 | 10.0 | 369.0 | {'maxJointVelocity': 180.0} |
| head_0 | RevoluteJoint | Z | -90.0 | 90.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| head_1 | RevoluteJoint | Y | -90.0 | 90.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 179.9087371826172} |
| left_arm_0 | RevoluteJoint |  | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_1 | RevoluteJoint | X | 0.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_2 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| left_arm_3 | RevoluteJoint | Y | -150.0 | 0.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| left_arm_4 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| left_arm_5 | RevoluteJoint | Y | -90.0 | 110.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| left_arm_6 | RevoluteJoint | Z | -155.0 | 155.0 | angular/force | 100.0 | 10.0 | 140.0 | {'maxJointVelocity': 360.0} |
| right_arm_0 | RevoluteJoint |  | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_1 | RevoluteJoint | X | -180.0 | 0.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_2 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 143.0 | {'maxJointVelocity': 180.0} |
| right_arm_3 | RevoluteJoint | Y | -150.0 | 0.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 180.0} |
| right_arm_4 | RevoluteJoint | Z | -180.0 | 180.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| right_arm_5 | RevoluteJoint | Y | -90.0 | 110.0 | angular/force | 100.0 | 10.0 | 70.0 | {'maxJointVelocity': 360.0} |
| right_arm_6 | RevoluteJoint | Z | -155.0 | 155.0 | angular/force | 100.0 | 10.0 | 140.0 | {'maxJointVelocity': 360.0} |
| tool_left | FixedJoint |  |  |  | None/None | None | None | None | {} |
| tool_right | FixedJoint |  |  |  | None/None | None | None | None | {} |
| gripper_finger_l1 | PrismaticJoint | X | -0.05000000074505806 | 0.0 | linear/force | 100.0 | 1.0 | 75.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_l2 | PrismaticJoint | X | 0.0 | 0.05000000074505806 | linear/force | 0.30934953689575195 | 0.00012373982463032007 | 1000.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_r1 | PrismaticJoint | X | -0.05000000074505806 | 0.0 | linear/force | 100.0 | 1.0 | 75.0 | {'maxJointVelocity': 0.03999999910593033} |
| gripper_finger_r2 | PrismaticJoint | X | 0.0 | 0.05000000074505806 | linear/force | 0.30934953689575195 | 0.00012373982463032007 | 1000.0 | {'maxJointVelocity': 0.03999999910593033} |

| 카메라 | f(mm) | hAp | vAp | fx@640 | hFOV° | clip |
|---|---|---|---|---|---|---|
| `/RBY1/link_head_2/first_person_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |
| `/RBY1/left_gripper/ee_left/left_hand_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |
| `/RBY1/right_gripper/ee_right/right_hand_camera` | 19.300 | 36.000 | 27.000 | 343.1 | 86.0 | [0.009999999776482582, 20.0] |

