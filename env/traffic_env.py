from pettingzoo import ParallelEnv
import traci
import traci.connection
import numpy as np
from env.observation import MARLObservation

class TrafficEnv(ParallelEnv):
    metadata = {"name": "traffic_marl_env"}

    def __init__(self):
        self.sumo_ids = ['A1', 'B0', 'B1', 'B2', 'C1']
        self.agents = [f"agent_{i}" for i in range(len(self.sumo_ids))]
        self.agent_to_sumo = {f"agent_{i}": sid for i, sid in enumerate(self.sumo_ids)}
        
        self.min_green = 10
        self.max_green = 60
        self.delta_t = 5
        
        self.last_waiting_times = {agent: 0 for agent in self.agents}
        self.current_phase_timer = {agent: 0 for agent in self.agents}
        # Initialize observation objects for each agent
        self.observations = {agent: MARLObservation(self.agent_to_sumo[agent]) for agent in self.agents}

    def reset(self, seed=None, options=None):
        try:
            if "default" in traci.connection._connections: traci.close()
        except Exception: pass
        finally:
            if "default" in traci.connection._connections: del traci.connection._connections["default"]

        traci.start(["sumo", "-c", "/content/MARL-TSC-SUMO-Group/sumo/config.sumocfg"])
        
        self.last_waiting_times = {agent: 0 for agent in self.agents}
        self.current_phase_timer = {agent: 0 for agent in self.agents}
        return {agent: self.observations[agent].get() for agent in self.agents}

    def step(self, actions):
        for agent, action in actions.items():
            self._apply_action(agent, action)
            
        traci.simulationStep()
        for agent in self.agents:
            self.current_phase_timer[agent] += 1
        
        obs = {agent: self.observations[agent].get() for agent in self.agents}
        rewards = {agent: self._compute_reward(agent) for agent in self.agents}
        return obs, rewards, {a: False for a in self.agents}, {a: False for a in self.agents}, {a: {} for a in self.agents}

    def _apply_action(self, agent, action):
        sumo_id = self.agent_to_sumo[agent]
        current_phase = traci.trafficlight.getPhase(sumo_id)
        
        if current_phase % 2 != 0:
            if self.current_phase_timer[agent] >= 3:
                traci.trafficlight.setPhase(sumo_id, (current_phase + 1) % 4)
                self.current_phase_timer[agent] = 0
            return

        if action == 1:
            new_duration = min(self.max_green, traci.trafficlight.getPhaseDuration(sumo_id) + self.delta_t)
            traci.trafficlight.setPhaseDuration(sumo_id, new_duration)
        elif action == 2:
            new_duration = max(self.min_green, traci.trafficlight.getPhaseDuration(sumo_id) - self.delta_t)
            traci.trafficlight.setPhaseDuration(sumo_id, new_duration)

        if self.current_phase_timer[agent] >= traci.trafficlight.getPhaseDuration(sumo_id):
            traci.trafficlight.setPhase(sumo_id, (current_phase + 1) % 4)
            self.current_phase_timer[agent] = 0

    def _compute_reward(self, agent):
        sumo_id = self.agent_to_sumo[agent]
        lanes = traci.trafficlight.getControlledLanes(sumo_id)
        current_waiting_time = sum([traci.lane.getWaitingTime(l) for l in lanes])
        throughput = sum([traci.lane.getLastStepVehicleNumber(l) for l in lanes])
        delay_component = (self.last_waiting_times[agent] - current_waiting_time) / 100.0
        reward = delay_component + (throughput * 0.1)
        self.last_waiting_times[agent] = current_waiting_time
        return float(reward)
