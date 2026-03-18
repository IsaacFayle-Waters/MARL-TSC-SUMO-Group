"""
TrafficEnv: Updated with exact paper parameters from Section 4.1.

Changes based on user-provided text:
- Action Interval (delta_t): 12s (reassessment between two consecutive actions).
- Adjustment Step (delta_t_tilde): 4s (green light adjustment time).
- q (Scaling factor): 3.
"""
from pettingzoo import ParallelEnv
import traci
import traci.connection
import sumolib
import numpy as np
from env.observation import MARLObservation

class TrafficEnv(ParallelEnv):
    metadata = {"name": "traffic_marl_env"}

    def __init__(self, net_file="/content/MARL-TSC-SUMO-Group/sumo/network.net.xml"):
        self.sumo_ids = ['A1', 'A2', 'A3', 'B0', 'B1', 'B2', 'B3', 'B4', 'C0', 'C1', 'C2', 'C3', 'C4', 'D0', 'D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3']
        self.agents = [f"agent_{i}" for i in range(len(self.sumo_ids))]
        self.agent_to_sumo = {f"agent_{i}": sid for i, sid in enumerate(self.sumo_ids)}
        
        # --- UPDATED MARL SGAT HYPERPARAMETERS (Section 4.1) ---
        self.q_scaling = 3.0 # scaling factor q = 3
        self.min_green = 10
        self.max_green = 60
        self.delta_t_tilde = 4 # Adjustment step = 4s
        self.reassessment_interval = 12 # Action interval = 12s

        self.net = sumolib.net.readNet(net_file)
        self.structural_data = self._parse_network_structure()

        self.last_waiting_times = {agent: 0 for agent in self.agents}
        self.current_phase_timer = {agent: 0 for agent in self.agents}
        self.observations_manager = {}

    def _parse_network_structure(self):
        data = {}
        for agent in self.agents:
            sumo_id = self.agent_to_sumo[agent]
            node = self.net.getNode(sumo_id)
            m_p_i = sum([edge.getLaneNumber() for edge in node.getIncoming()])
            dist = 500.0
            incoming_edges = node.getIncoming()
            if incoming_edges:
                dist = incoming_edges[0].getLength()
            data[agent] = {'m_p': max(1, m_p_i), 'dist': max(1.0, dist)}
        return data

    def reset(self, seed=None, options=None):
        try:
            import traci.connection
            if 'default' in traci.connection._connections: traci.close()
        except: pass
        traci.start(["sumo", "-c", "/content/MARL-TSC-SUMO-Group/sumo/config.sumocfg"])
        self.observations_manager = {agent: MARLObservation(self.agent_to_sumo[agent]) for agent in self.agents}
        self.last_waiting_times = {agent: 0 for agent in self.agents}
        self.current_phase_timer = {agent: 0 for agent in self.agents}
        return {agent: self.observations_manager[agent].get() for agent in self.agents}

    def step(self, actions):
        # 1. Apply actions
        for agent, action in actions.items():
            self._apply_action(agent, action)
        
        # 2. Advance simulation by reassessment_interval (12s)
        for _ in range(self.reassessment_interval):
            traci.simulationStep()
            for agent in self.agents: 
                self.current_phase_timer[agent] += 1
                # Automatic transition to yellow if phase exceeds programmed duration
                sumo_id = self.agent_to_sumo[agent]
                curr_phase = traci.trafficlight.getPhase(sumo_id)
                if self.current_phase_timer[agent] >= traci.trafficlight.getPhaseDuration(sumo_id):
                    traci.trafficlight.setPhase(sumo_id, (curr_phase + 1) % 4)
                    self.current_phase_timer[agent] = 0

        obs = {agent: self.observations_manager[agent].get() for agent in self.agents}
        rewards = {agent: self._compute_reward(agent) for agent in self.agents}
        return obs, rewards, {a: False for a in self.agents}, {a: False for a in self.agents}, {a: {} for a in self.agents}

    def _get_hetero_index(self, agent):
        m_p_i = self.structural_data[agent]['m_p']
        dist = self.structural_data[agent]['dist']
        cor_forward = m_p_i / (m_p_i * self.q_scaling * np.sqrt(dist))
        cor_backward = m_p_i / (m_p_i * self.q_scaling * np.sqrt(dist))
        return cor_forward, cor_backward

    def _compute_reward(self, agent):
        sumo_id = self.agent_to_sumo[agent]
        lanes = traci.trafficlight.getControlledLanes(sumo_id)
        current_waiting_time = sum([traci.lane.getWaitingTime(l) for l in lanes])
        delay = (self.last_waiting_times[agent] - current_waiting_time) / 100.0
        cor_f, cor_b = self._get_hetero_index(agent)
        reward = 1.0 * cor_f * cor_b * delay
        self.last_waiting_times[agent] = current_waiting_time
        return float(reward)

    def _apply_action(self, agent, action):
        sumo_id = self.agent_to_sumo[agent]
        current_phase = traci.trafficlight.getPhase(sumo_id)
        if current_phase % 2 != 0: return # Skip if yellow
        
        dur = traci.trafficlight.getPhaseDuration(sumo_id)
        if action == 1: 
            traci.trafficlight.setPhaseDuration(sumo_id, min(self.max_green, dur + self.delta_t_tilde))
        elif action == 2: 
            traci.trafficlight.setPhaseDuration(sumo_id, max(self.min_green, dur - self.delta_t_tilde))
