from pettingzoo import ParallelEnv
import traci
import traci.connection
import numpy as np

class TrafficEnv(ParallelEnv):
    metadata = {"name": "traffic_marl_env"}

    def __init__(self):
        self.sumo_ids = ['A1', 'B0', 'B1', 'B2', 'C1']
        self.agents = [f"agent_{i}" for i in range(len(self.sumo_ids))]
        self.agent_to_sumo = {f"agent_{i}": sid for i, sid in enumerate(self.sumo_ids)}
        self.last_waiting_times = {agent: 0 for agent in self.agents}

    def reset(self, seed=None, options=None):
        try:
            if "default" in traci.connection._connections:
                traci.close()
        except Exception:
            pass
        finally:
            if "default" in traci.connection._connections:
                del traci.connection._connections["default"]

        traci.start(["sumo", "-c", "/content/sumo/config.sumocfg"])
        self.last_waiting_times = {agent: 0 for agent in self.agents}
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        return observations

    def step(self, actions):
        for agent, action in actions.items():
            self._apply_action(agent, action)

        traci.simulationStep()

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        rewards = {agent: self._compute_reward(agent) for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent):
        sumo_id = self.agent_to_sumo[agent]
        lanes = traci.trafficlight.getControlledLanes(sumo_id)
        unique_lanes = list(dict.fromkeys(lanes))
        
        obs = []
        for lane in unique_lanes[:3]: # Limit to first 3 incoming lanes for fixed state size
            length = traci.lane.getLength(lane)
            # Metric 1: Traffic Density (n_vehicles / length)
            density = traci.lane.getLastStepVehicleNumber(lane) / length
            # Metric 2: Queuing Density (n_queued / length)
            queuing = traci.lane.getLastStepHaltingNumber(lane) / length
            obs.extend([density, queuing])
        
        while len(obs) < 6: obs.append(0.0)
        return np.array(obs, dtype=np.float32)

    def _apply_action(self, agent, action):
        sumo_id = self.agent_to_sumo[agent]
        phases = [0, 1, 2, 3]
        traci.trafficlight.setPhase(sumo_id, phases[action % len(phases)])

    def _compute_reward(self, agent):
        sumo_id = self.agent_to_sumo[agent]
        lanes = traci.trafficlight.getControlledLanes(sumo_id)
        
        # Weighted Reward: Negative Delay + Positive Throughput proxy
        current_waiting_time = sum([traci.lane.getWaitingTime(l) for l in lanes])
        # Throughput proxy: vehicles that were in the lane but left
        throughput = sum([traci.lane.getLastStepVehicleNumber(l) for l in lanes])
        
        delay_component = (self.last_waiting_times[agent] - current_waiting_time) / 100.0
        reward = delay_component + (throughput * 0.1)
        
        self.last_waiting_times[agent] = current_waiting_time
        return float(reward)
