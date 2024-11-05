from __future__ import absolute_import

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

lane_change_times = 2
speed_change_times = 2


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
    ego_edge_pos = traci.vehicle.getLanePosition(vid)
    lid = traci.vehicle.getLaneID(vid)
    edge_len = traci.lane.getLength(lid)
    edge_proportion = ego_edge_pos / edge_len
    lane_width = traci.lane.getWidth(lid)
    lat = traci.vehicle.getLateralLanePosition(vid)
    lane_num = traci.edge.getLaneNumber(eid)
    res = ((lane_index + 1) * lane_width + lat) / (lane_num * lane_width)
    return res, edge_proportion


ROOT_PATH = "/home/ujs/TPAMI"


class Highway_env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 90}

    def __init__(
        self,
        env="merge",
        start_edge="E2",
        end_junc="J3",
        end_edge="E3.123",
        gui=False,
        time_limit=200 * 10,
        param=[1, 10, 0.0],
        label="",
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

        self._initialize_reset_vals()
        self.reset_times = 0
        self.work_id, self.work_dir = self._init_work_space(env_name=env)
        self.config_path = self.work_dir + "/highway.sumocfg"
        self.task_level = 2
        self.end_road = end_junc
        self.angle_boundaries = [
            (-5.0, 60.0),  # 0~60
            (60.0, 120.0),  # 60~120
            (120.0, 180.0),  # 120~180
            (-180.0, -120.0),  # -180~-120
            (-120.0, -60.0),  # -120~-60
            (-60.0, 5.0),  # -60~0
        ]
        self.env = env
        self.gui = gui
        self._obs_key = "image"
        self._act_key = "action"
        self._size = (64, 64)
        self._obs_is_dict = False

        self.action_space = spaces.Box(
            # low=-1, high=1, shape=(2,), dtype=np.float32
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.observation_space_ = spaces.Box(
            low=-1, high=1, shape=(37,), dtype=np.float32
        )
        self.param = param
        self.label = label

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
        work_dir = f"{ROOT_PATH}/tmp/{task_id}"
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
            f"{ROOT_PATH}/env/{env_name}",
            work_dir,
        )
        script_path = os.path.join(work_dir, "autoGenTraffic.sh")
        with open(script_path, "r") as file:
            script_content = file.read()
            self._extract_starting_volumes(script_content)
        return task_id, work_dir

    def render(self, mode="human"):
        pass

    def calculate_ttc(self, veh_list):
        ego_speed = traci.vehicle.getSpeed(self.ego_id)
        ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
        ttc_values = []

        for vid in veh_list:
            if vid == self.ego_id:
                continue

            veh_speed = traci.vehicle.getSpeed(vid)
            veh_pos = np.array(traci.vehicle.getPosition(vid))
            relative_speed = ego_speed - veh_speed
            if relative_speed <= 0:
                continue  # 前车比自车快或速度相同，无需考虑

            distance = np.linalg.norm(veh_pos - ego_pos)

            # 根据方向确定是否在前方（假设车道方向与自车方向一致）
            angle = traci.vehicle.getAngle(
                vid) - traci.vehicle.getAngle(self.ego_id)
            if -15 <= angle <= 15: 
                ttc = distance / \
                    relative_speed if relative_speed > 0.01 else 300
                ttc_values.append(ttc)

        if ttc_values:
            min_ttc = min(ttc_values)
        else:
            min_ttc = float('inf')

        return min_ttc

    def raw_obs(self, veh_list):
        obs = []
        pos = [0.0, 0.0]

        if self.ego_id in veh_list:
            ego_x, ego_y = traci.vehicle.getPosition(self.ego_id)
            dis_goal_ego = np.linalg.norm(
                np.array([self.x_goal - ego_x, self.y_goal - ego_y])
            )
            ego_speed = traci.vehicle.getSpeed(self.ego_id)
            ego_angle = traci.vehicle.getAngle(self.ego_id)
            ego_acc = traci.vehicle.getAcceleration(self.ego_id)
            ego_lane_pos, ego_edge_pos = get_lane_pos(self.ego_id)
            ego_lateral_speed = traci.vehicle.getLateralSpeed(self.ego_id)
            ego_length = traci.vehicle.getLength(self.ego_id)

            pos = [ego_x, ego_y]

            # 获取其他车辆的相对距离和角度
            other_vehicles = []
            for vid in veh_list:
                if vid != self.ego_id:
                    veh_x, veh_y = traci.vehicle.getPosition(vid)
                    dis2veh = np.linalg.norm(
                        np.array([veh_x - ego_x, veh_y - ego_y]))
                    if dis2veh < self.detect_range:
                        angle2veh = math.degrees(
                            math.atan2(veh_y - ego_y, veh_x - ego_x))
                        other_vehicles.append((dis2veh, angle2veh, vid))

            # 按距离排序，选择最近的5辆车
            other_vehicles.sort(key=lambda x: x[0])
            nearest_vehicles = other_vehicles[:5]

            for vehicle in nearest_vehicles:
                dis, angle, vid = vehicle
                speed = traci.vehicle.getSpeed(vid)
                acc = traci.vehicle.getAcceleration(vid)
                lateral_speed = traci.vehicle.getLateralSpeed(vid)
                length = traci.vehicle.getLength(vid)

                obs.append(speed)
                obs.append(angle)
                obs.append(acc)
                obs.append(lateral_speed)
                obs.append(dis)
                obs.append(length)

            num_missing = 5 - len(nearest_vehicles)
            for _ in range(num_missing):
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            obs.append(ego_speed)  # 31
            obs.append(ego_angle)  # 32
            obs.append(ego_acc)  # 33
            obs.append(ego_lane_pos)  # 34
            obs.append(ego_lateral_speed)  # 35
            obs.append(ego_length)  # 36
            obs.append(dis_goal_ego)  # 37
            obs.append(dis_goal_ego)  # 39
            obs.append(ego_edge_pos)  # 39

        else:
            obs = [0.0] * (5 * 6)
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        self.max_dis_navigation])

        return obs, pos

    def norm_obs(self, veh_list):
        obs, pos = self.raw_obs(veh_list)
        state = []

        for i in range(5):
            base_index = i * 6
            state.extend([
                obs[base_index] / self.maxSpeed,          # 速度归一化
                obs[base_index + 1] / self.max_angle,     # 相对角度归一化
                obs[base_index + 2] / self.max_acc,       # 加速度归一化
                obs[base_index + 3] / self.max_lat_v,     # 横向速度归一化
                obs[base_index + 4] / self.detect_range,  # 距离归一化
                obs[base_index + 5] / 20,    # 车辆长度归一化
            ])

        base_self = 30
        state.extend([
            obs[base_self] / self.maxSpeed,               # 自车速度归一化
            obs[base_self + 1] / self.max_angle,          # 自车角度归一化
            obs[base_self + 2] / self.max_acc,            # 自车加速度归一化
            obs[base_self + 3],
            obs[base_self + 4] / self.max_lat_v,          # 自车横向速度归一化
            obs[base_self + 5] / 20,         # 自车车辆长度归一化
            obs[base_self + 6] / self.max_dis_navigation  # 距离目标归一化
        ])

        return np.array(state), pos

    def get_reward(self, veh_list):
        cost = 0.0
        overtime_check = False
        navigation_check = False
        obs, _ = self.raw_obs(veh_list)
        dis_goal_ego = obs[36]
        v_ego = obs[30]
        collision_check = self.check_collision()

        if collision_check:
            cost += (10 + self.all_speed / (self.time_step))

        min_ttc = self.calculate_ttc(veh_list)
        ttc_penalty = 0.0
        ttc_threshold = 3.0
        if min_ttc < ttc_threshold:
            ttc_penalty = (ttc_threshold - min_ttc) / ttc_threshold
            cost += ttc_penalty
            # print("TTT:", min_ttc, ttc_penalty)
        # if self.time_step % idle_step_limit == 0:
        #     if (
        #         abs(self.last_travel_dis -
        #             abs(dis_goal_ego - self.max_dis_navigation))
        #         < idle_dis_threshold
        #     ):
        #         idle_check = True
        #         # cost += 0.1
        #         print(">>>>>>>>>>>>>>>>>>>>>>>>> Idle too long:",
        #               self.time_step, "\n")
        #     self.last_travel_dis = abs(dis_goal_ego - self.max_dis_navigation)

        if dis_goal_ego < 30.0:
            navigation_check = True
            self.navigation_precent = 1
            cost -= (10 + self.all_speed / (self.time_step))
            print("=========================== Finish!" + "\n")
        else:
            self.navigation_precent = 1 - dis_goal_ego / self.max_dis_navigation

        if self.time_step > self.time_limit:
            overtime_check = True
            cost += 10*(1 - self.navigation_precent)
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
        )

    def check_collision(self):
        collision_check = False
        vlist = traci.simulation.getCollidingVehiclesIDList()
        if self.ego_id in vlist:
            collision_check = True
            print("====================> Collision!" + "\n")
        return collision_check

    def close(self):
        traci.switch(self.label)
        traci.close(self.label)
        if os.path.exists(f"{ROOT_PATH}/tmp/{self.work_id}"):
            shutil.rmtree(f"{ROOT_PATH}/tmp/{self.work_id}")

    def step(self, action):
        traci.switch(self.label)


        risk = False
        try:
            self.time_step += 1
            if "Auto" in traci.vehicle.getIDList():
                current_lane_pos, _ = get_lane_pos(self.ego_id)
                real_lane_change = current_lane_pos - self.last_lat_pos

                if self.reset_times >= lane_change_times:
                    # traci.vehicle.setLaneChangeMode(ego_id, 0)
                    # if abs(self.action[1]*real_lane_change) <= 0:
                    #     risk = True
                    #     print("risk", self.action[1], real_lane_change)
                    lane_change = action[1].item()
                    traci.vehicle.changeSublane(self.ego_id, lane_change)
                    if self.reset_times >= speed_change_times:
                        # traci.vehicle.setSpeedFactor(ego_id, 3)
                        acc = action[0].item()
                        traci.vehicle.setAcceleration(self.ego_id, acc, 0.1)
                    else:
                        action[0] = traci.vehicle.getAcceleration(
                            self.ego_id)/self.max_acc
                else:
                    action[0] = traci.vehicle.getAcceleration(
                        self.ego_id)/self.max_acc
                    action[1] = 100*real_lane_change
                self.last_lat_pos = current_lane_pos

                traci.simulationStep()
            else:
                navigation_check = done = True

            self.action = action
            veh_list = traci.vehicle.getIDList()
            (
                reward,
                collision_check,
                self.navigation_precent,
                overtime_check,
                navigation_check,
            ) = self.get_reward(veh_list)
            next_state, _ = self.norm_obs(veh_list)
            terminated = navigation_check or collision_check
            truncated = overtime_check
            info = {
                "mean_speed": self.all_speed / (self.time_step + 1),
                "time_step": self.time_step,
                "navigation": self.navigation_precent,
                "task_level": self.task_level,
                "total_reward": self.total_reward,
                "reset_times": self.reset_times,
                "overtime": overtime_check,
                "action": self.action,
                "navigation_check": navigation_check,
                "risk":  1.0 if (collision_check or risk) else 0.0
            }
            if terminated or truncated:
                print(info)
            done = terminated or truncated

            next_state = {self._obs_key: next_state}
            next_state["is_first"] = False
            next_state["is_last"] = done
            next_state["is_terminal"] = terminated
            next_state["risk"] = 1.0 if collision_check else 0.0
            return next_state, reward, done, info
        except Exception as e:
            handle_exception(e)

    def get_info(self):
        return {
            "mean_speed": self.all_speed / (self.time_step + 1),
            "time_step": self.time_step,
            "navigation": self.navigation_precent,
            "task_level": self.task_level,
            "total_reward": self.total_reward,
        }

    def _compute_soomth_loss(self, action, reward):
        if self.time_step > 1:
            last_action = self.last_action
            action_smoothness_loss = (action[0] - last_action[0]) ** 2 + (
                action[1] - last_action[1]
            ) ** 2
            self.action_smoothness_loss = action_smoothness_loss
        else:
            self.action_smoothness_loss = 0
        self.last_action = action
        return self.action_smoothness_loss

    def reset(self, seed=None, options=None):
        if self.time_step == 0:
            self.start_sumo(gui=self.gui)
        traci.switch(self.label)
        self._initialize_reset_vals()
        self._create_traffic_config()
        self._load_simulation()
        self._wait_for_auto_car()
        VehicleIds = traci.vehicle.getIDList()
        obs, _ = self.norm_obs(VehicleIds)
        obs = {self._obs_key: obs}
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        obs["risk"] = 0.0
        # NOTE policy project
        if self.reset_times >= lane_change_times:
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)
            if self.reset_times >= speed_change_times:
                traci.vehicle.setSpeedMode(self.ego_id, 0)
        return obs

    def _initialize_reset_vals(self):
        """Initialize variables and parse config."""

        self.max_acc = None
        self.max_lat_v = None
        self.maxSpeed = None
        self.max_angle = None
        self.x_goal, self.y_goal = None, None
        self.max_dis_navigation = None
        self.time_step = 0
        self.all_speed = 0
        self.total_reward = 0
        self.navigation_precent = 0
        self.last_lat_pos = 0
        self.action = [0, 0]

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

        if self.reset_times == 0:
            os.chdir(self.work_dir)
            script_path = os.path.join(os.getcwd(), "autoGenTraffic.sh")
            if not os.access(script_path, os.X_OK):
                os.chmod(script_path, 0o755)
            # if self.task_level < 2:
            #     return
            new_volumes = [
                float(start_vol) * (self.task_level + 1) for start_vol in self.start_vol
            ]
            # self.task_level -= 1
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
                print("Generate Traffic Success!", result.stdout)
                if result.stderr:
                    print("Generate Traffic Error Output:")
                    print(result.stderr)
            except subprocess.CalledProcessError as e:
                handle_exception(e)
            finally:
                pass

    def _auto_trip_exists(self, trips_element):
        """Check if the 'Auto' trip already exists."""
        trip_elements = trips_element.getElementsByTagName("trip")
        return any(trip.getAttribute("id") == "Auto" for trip in trip_elements)

    def _create_traffic(self, trip_dom, trips_element):
        """Create a new 'Auto' trip element."""
        attributes = {
            "id": "Auto",
            "depart": str(np.random.randint(30, 100)),
            "departLane": "best",
            "departSpeed": "2.00",
            "minGap": "0.1",
            "lcStrategic": "100",
            "lcSpeedGain": "100000",
            "lcOpposite": "100",
            "lcSpeedGainRight": "1",
            "lcSpeedGainLookahead": "0.1",
            "lcAssertive": "100000",
            "lcImpatience": "1.5",
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
            "depart": str(np.random.randint(30, 100)),
            "departLane": "best",
            "departSpeed": "2.00",
            "minGap": "0.1",
            "lcStrategic": "100",
            "lcSpeedGain": "100000",
            "lcOpposite": "100",
            "lcSpeedGainRight": "1",
            "lcSpeedGainLookahead": "0.1",
            "lcAssertive": "100000",
            "lcImpatience": "1.5",
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

    def _load_simulation(self):
        """Load the simulation with the new configuration."""
        traci.load(["-c", self.config_path, "--seed",
                   str(np.random.randint(0, 10000))])
        print(f"============= Resetting the env {self.reset_times}============")
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
                # traci.vehicle.setLaneChangeMode(self.ego_id, 0)
                # traci.vehicle.setSpeedMode(self.ego_id, 0)

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
