from __future__ import absolute_import
import copy
from zoo.atari.envs.atari_wrappers import wrap_lightzero
from easydict import EasyDict
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray
from ding.envs import BaseEnv, BaseEnvTimestep
from typing import List

import numpy as np
import os
import sys
import math
import xml.dom.minidom
import traci
import sumolib
from gym import spaces
import gym
import uuid
import subprocess
import traceback
import re
import shutil
import wandb


def handle_exception(e):
    print(e)
    print(traceback.format_exc())


def copy_files(files, source_dir, destination_dir):
    for file in files:
        source_file = os.path.join(source_dir, file)
        destination_file = os.path.join(destination_dir, file)
        if os.path.exists(source_file):
            shutil.copy2(source_file, destination_file)
        else:
            print(f"warning {source_file} not exist")


def get_zone_index(angle, angle_boundaries):
    for i, (start, end) in enumerate(angle_boundaries):
        if start <= angle < end:
            return i
    return len(angle_boundaries) - 1


def get_lane_pos(vid):
    lane_index = traci.vehicle.getLaneIndex(vid)
    eid = traci.vehicle.getRoadID(vid)
    lid = traci.vehicle.getLaneID(vid)
    lane_width = traci.lane.getWidth(lid)
    lat = traci.vehicle.getLateralLanePosition(vid)
    lane_num = traci.edge.getLaneNumber(eid)
    res = ((lane_index + 1) * lane_width + lat) / (lane_num * lane_width)
    return res


class Highway_env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 90}

    def __init__(
        self,
        env="merge",
        start_edge="E2",
        end_junc="J3",
        end_edge="E3.123",
        gui=False,
        time_limit=400 * 10,
        param=[1, 10, 0.1],
        label=str(uuid.uuid4()),
    ):
        super().__init__()
        self.ego_id = "Auto"
        self.detect_range = 100.0
        self.end_junc = end_junc
        self.end_edge = end_edge
        self.start_edge = start_edge
        self.time_limit = time_limit
        self.start_vol = []
        self._max_episode_steps = 400 * 10

        self.max_acc = None
        self.max_lat_v = None
        self.maxSpeed = None
        self.max_angle = None
        self.x_goal, self.y_goal = None, None
        self.max_dis_navigation = None
        self.reset_times = 0
        self.work_id, self.work_dir = self._init_work_space(env_name=env)
        self.config_path = self.work_dir + "/highway.sumocfg"
        self.env_name = env
        self.gui = gui
        self.navigation_precent = 0
        self.time_step = 0
        self._obs_key = "image"
        self._act_key = "action"
        self._size = (64, 64)
        self._obs_is_dict = False

        self.end_road = end_junc
        self.angle_boundaries = [
            (0.0, 60.0),  # 0~60
            (60.0, 120.0),  # 60~120
            (120.0, 180.0),  # 120~180
            (-180.0, -120.0),  # -180~-120
            (-120.0, -60.0),  # -120~-60
            (-60.0, 0.0),  # -60~0
        ]

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )
        self.observation_space_ = spaces.Box(
            low=-1, high=1, shape=(29,), dtype=np.float32
        )
        self.time_step = 0
        self.task_level = 1
        self.consecutive_finish = 0
        self.param = param
        self.label = label
        self.start_sumo(gui=gui)

    @property
    def observation_space(self):
        spaces = {self._obs_key: self.observation_space_}
        return gym.spaces.Dict(
            {
                **spaces,
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )

    def _init_work_space(self, env_name):
        """Create tmp work space for sumo"""
        task_id = uuid.uuid4()
        work_dir = f"/home/ujs/TPAMI/tmp/{task_id}"

        os.mkdir(work_dir)
        files_to_copy = [
            "background.jpg",
            "stop.xml",
            "background.xml",
            "highway.net.xml",
            "highway.sumocfg",
            "autoGenTraffic.sh",
            "bus.trips.xml",
            "bus.config.txt",
            "car.config.txt",
            "car.trips.xml",
            "vTypeDistributions.add.xml",
            "auto.trips.xml",
        ]
        copy_files(
            files_to_copy,
            f"/home/ujs/TPAMI/env/{env_name}",
            work_dir,
        )
        script_path = os.path.join(work_dir, "autoGenTraffic.sh")
        with open(script_path, "r") as file:
            script_content = file.read()
            self._extract_starting_volumes(script_content)
        return task_id, work_dir

    def render(self, mode="human"):
        pass

    def raw_obs(self, veh_list):  # dimension: 24+5
        obs = []
        if self.ego_id in veh_list:
            obs_space = [[[], [], []] for _ in range(6)]  # 3*6

            ego_x, ego_y = traci.vehicle.getPosition(self.ego_id)
            dis_goal_ego = np.linalg.norm(
                np.array([self.x_goal - ego_x, self.y_goal - ego_y])
            )

            for vid in veh_list:
                veh_x, veh_y = traci.vehicle.getPosition(vid)
                dis2veh = np.linalg.norm(
                    np.array([veh_x - ego_x, veh_y - ego_y]))

                if vid != self.ego_id and dis2veh < self.detect_range:
                    angle2veh = math.degrees(
                        math.atan2(veh_y - ego_y, veh_x - ego_x))

                    obs_direction_index = get_zone_index(
                        angle2veh, self.angle_boundaries
                    )

                    obs_space[obs_direction_index][0].append(vid)
                    obs_space[obs_direction_index][1].append(dis2veh)
                    obs_space[obs_direction_index][2].append(angle2veh)

            for direction_space in obs_space:
                if len(direction_space[0]) == 0:
                    obs.append(self.detect_range)
                    obs.append(0.0)
                    obs.append(0.0)
                    obs.append(0.0)
                else:
                    mindis_v_index = direction_space[1].index(
                        min(direction_space[1]))
                    obs.append(min(direction_space[1]))
                    obs.append(direction_space[2][mindis_v_index])
                    obs.append(
                        traci.vehicle.getSpeed(
                            direction_space[0][mindis_v_index])
                    )
                    obs.append(
                        traci.vehicle.getAngle(
                            direction_space[0][mindis_v_index])
                    )

            obs.append(traci.vehicle.getSpeed(self.ego_id))
            obs.append(traci.vehicle.getAngle(self.ego_id))
            obs.append(get_lane_pos(self.ego_id))
            obs.append(traci.vehicle.getLateralSpeed(self.ego_id))
            obs.append(dis_goal_ego)
            pos = [ego_x, ego_y]

        else:
            zeros = [0.0] * 3
            detect_range_repeat = [self.detect_range] + zeros
            obs = detect_range_repeat * 6 + [
                0.0,  # ego speed
                0.0,  # ego angle
                0.0,  # ego pos lateral
                0.0,  # ego speed lateral
                self.max_dis_navigation,
            ]
            pos = [0.0, 0.0]

        return obs, pos

    def norm_obs(self, veh_list):
        obs, pos = self.raw_obs(veh_list)
        state = []
        for i in range(6):  # 6 directions
            base_index = i * 4
            state.extend(
                [
                    obs[base_index] / self.detect_range,
                    obs[base_index + 1] / self.max_angle,
                    obs[base_index + 2] / self.maxSpeed,
                    obs[base_index + 3] / self.max_angle,
                ]
            )
        # Adding the last specific elements
        state.extend(
            [
                obs[24] / self.maxSpeed,  # ego speed
                obs[25] / self.max_angle,  # ego angle
                obs[26] / self.detect_range,  # ego pos lateral
                obs[27] / self.max_lat_v,  # ego speed lateral
                obs[28] / self.max_dis_navigation,
            ]
        )

        return np.array(state, dtype=np.float32), pos

    def get_reward(self, veh_list):
        cost = 0.0
        overtime_check = False
        navigation_check = False
        idle_check = False
        idle_step_limit = 10 * 10
        idle_dis_threshold = 10
        obs, _ = self.raw_obs(veh_list)
        dis_goal_ego = obs[28]
        v_ego = obs[24]
        collision_check = self.check_collision()

        if collision_check:
            # pass
            cost += 10

        if self.time_step % idle_step_limit == 0:
            if (
                abs(self.last_travel_dis -
                    abs(dis_goal_ego - self.max_dis_navigation))
                < idle_dis_threshold
            ):
                idle_check = True
                cost += 0.1
                print(">>>>>>>>>>>>>>>>>>>>>>>>> Idle too long:",
                      self.time_step, "\n")
            self.last_travel_dis = abs(dis_goal_ego - self.max_dis_navigation)

        if dis_goal_ego < 30.0:
            navigation_check = True
            self.navigation_precent = 1
            cost -= (1 + self.all_speed / (self.time_step + 1))
            self.consecutive_finish += 1
            print("=========================== Finish!" + "\n")
        else:
            self.navigation_precent = 1 - dis_goal_ego / self.max_dis_navigation

        if self.time_step > self.time_limit:
            overtime_check = True
            cost += 1 - self.navigation_precent
            print("+++++++++++++++++++++> over time:",
                  self.navigation_precent, "\n")

        speed_reward = v_ego / self.maxSpeed
        # speed_reward = v_ego - 0.1
        self.all_speed += v_ego
        reward = (
            self.param[0] * speed_reward
            - self.param[1] * cost
            # + self.param[2] * (-1 - np.log(1.0 - self.navigation_precent))
            + self.param[2] * self.navigation_precent
        )
        self.total_reward += reward
        return (
            reward,
            collision_check,
            self.navigation_precent,
            overtime_check,
            navigation_check,
            idle_check,
        )

    def check_collision(self):
        collision_check = False
        vlist = traci.simulation.getCollidingVehiclesIDList()
        if self.ego_id in vlist:
            collision_check = True
            print("====================> Collision!")
        return collision_check

    def close(self):
        traci.close(self.label)
        if os.path.exists(
            f"/home/ujs/TPAMI/tmp/{self.work_id}"
        ):
            shutil.rmtree(
                f"/home/ujs/TPAMI/tmp/{self.work_id}"
            )

    def step(self, action):
        traci.switch(self.label)
        try:
            self.time_step += 1
            if not hasattr(self, 'prev_action'):
                self.prev_action = action
            smoothed_action = 0.5 * self.prev_action + 0.5 * action
            self.prev_action = smoothed_action
            acc, lane_change = smoothed_action[0].item(
            ), smoothed_action[1].item()
            control_acc = self.max_acc * acc
            traci.vehicle.changeSublane(self.ego_id, lane_change)
            traci.vehicle.setAcceleration(
                self.ego_id, control_acc, duration=0.03)
            traci.simulationStep()
            if self.time_step % (100 * 30) == 0:
                print("Step time:", int(self.time_step) / 30)
            veh_list = traci.vehicle.getIDList()

            (
                reward,
                collision_check,
                self.navigation_precent,
                overtime_check,
                navigation_check,
                idle_check,
            ) = self.get_reward(veh_list)
            next_state, pos = self.norm_obs(veh_list)
            terminated = navigation_check or collision_check
            truncated = overtime_check
            info = {
                "mean_speed": self.all_speed / (self.time_step + 1),
                "time_step": self.time_step,
                "navigation": self.navigation_precent,
                "task_level": self.task_level,
                "total_reward": self.total_reward,
                "violation": collision_check,
                "collision": collision_check,
                "idle": idle_check,
                "overtime": overtime_check,
                "navigation_check": navigation_check,
            }
            if terminated or truncated:
                wandb.log(info)
                print(info)
                pass
            done = terminated or truncated
            if collision_check or overtime_check:
                self.consecutive_finish = 0

            return next_state, reward, done, info
        except Exception as e:
            handle_exception(e)

    def _compute_soomth_loss(self, action, reward):
        if self.time_step > 1:
            last_action = self.last_action
            action_smoothness_loss = (
                action[0] - last_action[0])**2 + (action[1] - last_action[1])**2
            self.action_smoothness_loss = action_smoothness_loss
        else:
            self.action_smoothness_loss = 0
        self.last_action = action
        return self.action_smoothness_loss

    def check_done(self, states):
        return self.check_violation(states)

    def check_violation(self, states):
        traci.switch(self.label)
        res = self.check_collision()
        return [res]

    def reset(self, seed=None, options=None):
        traci.switch(self.label)
        print(self.navigation_precent)
        print(self.time_step)
        # wandb.log(
        #     {
        #         "navigation": self.navigation_precent,
        #         "time_step": self.time_step,
        #     }
        # )
        self._initialize_reset_vals()
        self._create_traffic_config()
        self._load_simulation()
        self._wait_for_auto_car()
        VehicleIds = traci.vehicle.getIDList()
        obs, info = self.norm_obs(VehicleIds)
        return obs

    def _initialize_reset_vals(self):
        """Initialize variables and parse config."""
        self.time_step = 0
        self.last_travel_dis = 20
        self.dom = xml.dom.minidom.parse(
            "/home/ujs/TPAMI/tmp/"
            + str(self.work_id)
            + "/highway.sumocfg"
        )

        self.root = self.dom.documentElement
        self.all_speed = 0
        self.total_reward = 0
        self.navigation_precent = 0

    def _create_traffic_config(self):
        """Handles trip XML file creation or updating."""
        config_dir = os.path.dirname(self.config_path)
        trip_path = os.path.join(config_dir, "auto.trips.xml")

        if not os.path.exists(trip_path):
            trip_dom = xml.dom.minidom.Document()
            trips_element = trip_dom.createElement("trips")
            trip_dom.appendChild(trips_element)
        else:
            trip_dom = xml.dom.minidom.parse(trip_path)
            trips_element = trip_dom.documentElement

        if not self._auto_trip_exists(trips_element):
            self._create_traffic(trip_dom, trips_element)
        else:
            self._insert_auto_trip(trips_element)
        self._save_trip_file(trip_dom, trip_path)

        if self.consecutive_finish >= 10 or self.reset_times == 0:
            os.chdir(self.work_dir)
            script_path = os.path.join(os.getcwd(), "autoGenTraffic.sh")
            if not os.access(script_path, os.X_OK):
                os.chmod(script_path, 0o755)
            if self.task_level < 1:
                return
            new_volumes = [
                float(start_vol) * (self.task_level + 1) for start_vol in self.start_vol
            ]
            self.task_level -= 1
            self.consecutive_finish = 0
            print("new vol:", new_volumes)
            with open(script_path, "r") as file:
                content = file.read()
                matches = re.findall(r"-p\s+\d+\.\d+", content)

                if len(matches) >= 2:
                    content = content.replace(
                        matches[0], f"-p {new_volumes[0]}", 1)
                    content = content.replace(
                        matches[1], f"-p {new_volumes[1]}", 1)
                with open(script_path, "w") as file:
                    file.write(content)

            try:
                result = subprocess.run(
                    [script_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.stderr:
                    print("Generate Traffic Error Output:")
                    print(result.stderr)
            except subprocess.CalledProcessError as e:
                handle_exception(e)
            finally:
                os.chdir("..")

    def _auto_trip_exists(self, trips_element):
        """Check if the 'Auto' trip already exists."""
        trip_elements = trips_element.getElementsByTagName("trip")
        return any(trip.getAttribute("id") == "Auto" for trip in trip_elements)

    def _create_traffic(self, trip_dom, trips_element):
        """Create a new 'Auto' trip element."""
        attributes = {
            "id": "Auto",
            "depart": "20",
            "departLane": "best",
            "departSpeed": "2.00",
            "color": "red",
            "from": self.start_edge,
            "to": self.end_edge,
        }
        new_trip_element = trip_dom.createElement("trip")
        for key, value in attributes.items():
            new_trip_element.setAttribute(key, value)
        trips_element.insertBefore(new_trip_element, trips_element.firstChild)

    def _insert_auto_trip(self, trips_element):
        """Update the existing 'Auto' trip element."""
        trip_elements = trips_element.getElementsByTagName("trip")
        auto_trip = next(
            trip for trip in trip_elements if trip.getAttribute("id") == "Auto"
        )
        attributes = {
            "id": "Auto",
            "depart": "20",
            "departLane": "best",
            "departSpeed": "2.00",
            "color": "red",
            "from": self.start_edge,
            "to": self.end_edge,
        }
        for key, value in attributes.items():
            auto_trip.setAttribute(key, value)

    def _save_trip_file(self, trip_dom, trip_path):
        """Save the updated trips XML file."""
        with open(trip_path, "w") as trip_file:
            trip_dom.writexml(trip_file)

    def _update_config(self, seed):
        """Update and save the configuration with the new seed value."""
        if not seed:
            seed = np.random.randint(0, 10000)
        random_seed_element = self.root.getElementsByTagName("seed")[0]
        # super().reset(seed=seed)
        try:
            if self.reset_times % 2 == 0:
                random_seed_element.setAttribute("value", str(seed))

            with open(self.config_path, "w") as file:
                self.dom.writexml(file)
        except Exception as e:
            handle_exception(e)

    def _load_simulation(self):
        """Load the simulation with the new configuration."""
        traci.load(["-c", self.config_path, "--seed",
                   str(np.random.randint(0, 10000))])
        print(
            f"============= Resetting the env {self.reset_times}============")
        self.reset_times += 1

    def _extract_starting_volumes(self, script_content):
        # Regex to find all instances of -p followed by a number
        pattern = re.compile(r"-p (\d+\.\d+)")
        matches = pattern.findall(script_content)
        self.start_vol.extend(matches)

    def _wait_for_auto_car(self):
        """Wait for the 'Auto' car to appear in the simulation."""
        AutoCarAvailable = False
        while not AutoCarAvailable:
            traci.simulationStep()
            VehicleIds = traci.vehicle.getIDList()
            if self.ego_id in VehicleIds:
                AutoCarAvailable = True
                if self.gui:
                    traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
                    traci.gui.trackVehicle(traci.gui.DEFAULT_VIEW, self.ego_id)
                traci.vehicle.setSpeedFactor(self.ego_id, 3)
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)
                traci.vehicle.setSpeedMode(self.ego_id, 0)

                self.maxSpeed = traci.vehicle.getMaxSpeed(self.ego_id)
                self.max_angle = 360.0
                self.x_goal, self.y_goal = traci.junction.getPosition(
                    self.end_junc)
                self.max_dis_navigation = sum(
                    traci.lane.getLength(v + "_0")
                    for v in traci.vehicle.getRoute(self.ego_id)
                )
                self.max_acc = traci.vehicle.getAccel(self.ego_id)
                self.max_lat_v = traci.vehicle.getMaxSpeedLat(self.ego_id)

    def start_sumo(
        self,
        gui=False,
    ):

        sumoBinary = "sumo-gui" if gui else "sumo"
        traci.start(
            [sumoBinary, "-c", self.config_path,
                "--collision.check-junctions", "true"],
            label=self.label,
        )


@ENV_REGISTRY.register('highway')
class HighwayENV(BaseEnv):
    config = dict(
        # (int) The number of environment instances used for data collection.
        collector_env_num=8,
        # (int) The number of environment instances used for evaluator.
        evaluator_env_num=2,
        # (int) The number of episodes to evaluate during each evaluation period.
        n_evaluator_episode=2,
        # (str) The name of the Atari game environment.
        # env_id='PongNoFrameskip-v4',
        # (str) The type of the environment, here it's Atari.
        env_type='Atari',
        # (tuple) The shape of the observation space, which is a stacked frame of 4 images each of 96x96 pixels.
        observation_shape=(29,),
        # (int) The maximum number of steps in each episode during data collection.
        collect_max_episode_steps=int(1.08e6),
        # (int) The maximum number of steps in each episode during evaluation.
        eval_max_episode_steps=int(1.08e6),
        # (bool) If True, the game is rendered in real-time.
        render_mode_human=False,
        # (bool) If True, a video of the game play is saved.
        save_replay=False,
        # replay_path (str or None): The path to save the replay video. If None, the replay will not be saved.
        # Only effective when env_manager.type is 'base'.
        replay_path=None,
        # (bool) If set to True, the game screen is converted to grayscale, reducing the complexity of the observation space.
        gray_scale=True,
        # (int) The number of frames to skip between each action. Higher values result in faster simulation.
        frame_skip=2,
        manager=dict(shared_memory=True, ),
        # (int) The value of the cumulative reward at which the training stops.
        stop_value=int(1e6),
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        """
        Overview:
            Return the default configuration for the Atari LightZero environment.
        Arguments:
            - cls (:obj:`type`): The class AtariEnvLightZero.
        Returns:
            - cfg (:obj:`EasyDict`): The default configuration dictionary.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict) -> None:
        """
        Overview:
            Initialize the Atari LightZero environment with the given configuration.
        Arguments:
            - cfg (:obj:`EasyDict`): The configuration dictionary.
        """
        self.cfg = cfg
        self._init_flag = False

    def reset(self) -> dict:
        if not self._init_flag:
            env = Highway_env(
                env="merge",
                start_edge="E2",
                end_junc="J3",
                end_edge="E3.123",
                gui=False,
                time_limit=400 * 30,
                param=[1, 10, 0.1],
                label=str(uuid.uuid4()),
            )
            self._env = wrap_lightzero(env,
                                       self.cfg, episode_life=self.cfg.episode_life, clip_rewards=self.cfg.clip_rewards)
            self._observation_space = self._env.env.observation_space
            self._action_space = self._env.env.action_space
            self._reward_space = gym.spaces.Box(
                low=-10000, high=10000, shape=(1,), dtype=np.float32
            )
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            # self._env.env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.env.seed(self._seed)

        result = self._env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result

        self.obs = to_ndarray(obs)
        self._eval_episode_return = 0.
        obs = self.observe()
        return obs

    def step(self, action: int) -> BaseEnvTimestep:
        """
        Overview:
            Execute the given action and return the resulting environment timestep.
        Arguments:
            - action (:obj:`int`): The action to be executed.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): The environment timestep after executing the action.
        """
        obs, reward, done, info = self._env.step(action)
        self.obs = to_ndarray(obs)
        self.reward = np.array(reward).astype(np.float32)
        self._eval_episode_return += self.reward
        observation = self.observe()
        if done:
            info['eval_episode_return'] = self._eval_episode_return

        return BaseEnvTimestep(observation, self.reward, done, info)

    def observe(self) -> dict:
        """
        Overview:
            Return the current observation along with the action mask and to_play flag.
        Returns:
            - observation (:obj:`dict`): The dictionary containing current observation, action mask, and to_play flag.
        """
        observation = self.obs

        action_mask = np.ones(2, 'int8')
        return {'observation': observation, 'action_mask': action_mask, 'to_play': -1}

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment's random number generator. Can handle both static and dynamic seeding.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    @property
    def legal_actions(self):
        return np.arange(self._action_space.n)

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def close(self) -> None:
        """
        Close the environment, and set the initialization flag to False.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Property to access the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Property to access the action space of the environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        """
        Property to access the reward space of the environment.
        """
        return self._reward_space

    def __repr__(self) -> str:
        return "LightZero Atari Env({})".format(self.cfg.env_id)

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.collect_max_episode_steps
        cfg.episode_life = True
        cfg.clip_rewards = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.eval_max_episode_steps
        cfg.episode_life = False
        cfg.clip_rewards = False
        return [cfg for _ in range(evaluator_env_num)]
