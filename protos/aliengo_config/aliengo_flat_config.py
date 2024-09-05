# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import torch

class AliengoFlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 53         #原48
        num_actions = 12

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class commands( LeggedRobotCfg.commands ):
        curriculum = True #after change curriculum scale to 0.94 ,turn this bool var to true
        max_curriculum = 4.5
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 50.# 10. # time before command are changed[s]     
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            # lin_vel_x = [-0.5, 0.5] # min max [m/s]
            # lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            # ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]

            lin_vel_x = [-3., -3.] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]           
            ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.7854,     # [rad]
            'RL_thigh_joint': 0.7854,   # [rad]
            'FR_thigh_joint': 0.7854,     # [rad]
            'RR_thigh_joint': 0.7854,   # [rad]

            'FL_calf_joint': -1.5708,   # [rad]
            'RL_calf_joint': -1.5708,    # [rad]
            'FR_calf_joint': -1.5708,  # [rad]
            'RR_calf_joint': -1.5708,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 40., 'thigh': 55., 'calf': 55.}  # [N*m/rad]
        damping = {'hip': 2., 'thigh': 2., 'calf': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10


    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/aliengo/urdf/aliengo.urdf'
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        limit_joint = "hip_joint" #wms 限制外摆关节过于内收
        terminate_after_contacts_on = ["base","thigh"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        abd_pos = torch.tensor([[0.24,0.14],[0.24,-0.14],[-0.24,0.14],[-0.24,-0.14]],dtype=torch.float)

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = False
        friction_range = [0.3, 1.25]        #摩擦力的随机化范围，[最小摩擦力，最大摩擦力]
        added_mass_range = [-1., 1.]
        push_robots = True                  #是否用外力推动机器人
        push_interval_s = 10                #推动机器人的时间间隔
        max_push_vel_xy = 1.                #推动机器人的最大速度

    class sim(LeggedRobotCfg.sim):
        dt =  0.001  

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9      # 原0.9
        base_height_target = 0.36    # 原0.28
        max_contact_force = 200. # forces above this value are penalized,原100
        gait_full_period = 0.4
        stand_scale = 0.5 #0~1
        swing_height = 0.08
        sameSide_foot_distance_scale_x = 0.15 #0.366
        sameSide_foot_distance_scale_y = 0.04
        diffSide_foot_distance_scale_x = 0.08
        diffSide_foot_distance_scale_y = 0.2 #0.2642
        foot_x_distance_scale = 0.2
        foot_y_distance_scale = 0.1
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.5                 # 线速度跟踪奖励的比例因子，原1.0
            tracking_ang_vel = 0.7               # 角速度跟踪奖励的比例因子，原0.5
            ang_vel_xy = -0.09         # xy平面角速度奖励的比例因子，ang_vel_xy是求每个环境的（wx^2 + wy^2）之和，原-0.05
            stand_still = -0.9          # 静止奖励的比例因子，原-0.
            action_rate = -0.012        # 动作速率奖励的比例因子，action_rate是每个环境的（last_actions - actions）^ 2的和，原-0.01
            dof_acc = -4.2e-7          # 关节加速度奖励的比例因子，dof_acc是求关节加速度(（last_dof_vel - dof_vel）/ dt)^2的和，原-2.5e-7 #7_13 日以前参数 -3.2e-7
            orientation = -2.7          # 姿态奖励的比例因子，重力在x轴，y轴的分量的平方和，原-0.

            base_height = -7.
            feet_air_time_var = -0.0          # 自设奖励，奖励四条腿腾空时间方差小的情况
            gait_cycle_air_time_var = -0.0
            follow_foot_trajectoryZ = 0.3 #7_13 日以前参数 0.4
            foot_contact_like_trot = 0.3
            foot_contact_no_slip = -0.15
            dof_pos_limits = -100.
            limits_hip_pos = 0. #-0.23
            foot_distance_punishment = 50. #80.
            feet_ellipse_area_hold = -0.1 #-0.15 

class AliengoFlatCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_aliengo'
        max_iterations = 30000
    class policy:
        init_noise_std = 1.0                     # 浮点数属性，表示策略网络初始化时的噪声标准差。在初始化策略网络参数时，可以选择添加一定量的噪声以帮助策略网络更好地探索环境
        actor_hidden_dims = [512, 256, 128]      # 整数列表属性，表示策略网络中的隐藏层的维度。在这里，使用了一个包含三个元素的列表，分别表示了三个隐藏层的维度
        critic_hidden_dims = [512, 256, 128]     # 整数列表属性，表示评论者网络中的隐藏层的维度。与策略网络相似，这里也使用了一个包含三个元素的列表，分别表示了三个隐藏层的维度
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid   可以是 elu、relu、selu、crelu、lrelu、tanh 或 sigmoid

