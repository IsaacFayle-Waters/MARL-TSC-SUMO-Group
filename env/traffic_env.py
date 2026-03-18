"""
TrafficEnv: Multi-Agent Traffic Signal Control Environment with Heterogeneous Correlation Rewards.

This environment implements the MARL_SGAT framework for traffic signal control using SUMO.
It features a custom reward function based on the paper's 'Heterogeneous Correlation Index',
which accounts for the structural relationships between upstream and downstream intersections.

ACTION SPACE (Phase Duration Adjustment):
- Action 0: Maintain current phase duration.
- Action 1: Extend current green phase by delta_t (+5s).
- Action 2: Reduce current green phase by delta_t (-5s).

REWARD LOGIC:
Incorporates Equation (8) from the paper, using correlation indices (cor_forward, cor_backward)
to weight the impact of local delay and throughput based on intersection heterogeneity.
"""
from pettingzoo import ParallelEnv
import traci
import traci.connection
import numpy as np
from env.observation import MARLObservation

class TrafficEnv(ParallelEnv):
    metadata = {"name": "traffic_marl_env"}

    def __init__(self):
        """
        Initialize the environment parameters and agent mappings.
        """
        self.sumo_ids = ['A1', 'B0', 'B1', 'B2', 'C1']
        self.agents = [f"agent_{i}" for i in range(len(self.sumo_ids))]
        self.agent_to_sumo = {f"agent_{i}": sid for i, sid in enumerate(self.sumo_ids)}
        
        # --- MARL SGAT HYPERPARAMETERS ---
        self.q_scaling = 1.0 # Scaling factor 'q' from the paper to unify numerical ranges.
        self.min_green = 10  # Minimum green time allowed.
        self.max_green = 60  # Maximum green time allowed.
        self.delta_t = 5     # Time increment for duration adjustments.

        # State tracking
        self.last_waiting_times = {agent: 0 for agent in self.agents}
        self.current_phase_timer = {agent: 0 for agent in self.agents}
        self.observations_manager = {}

    def reset(self, seed=None, options=None):
        """
        Restart the SUMO simulation and re-initialize agent observation handlers.
        """
        try:
            if "default" in traci.connection._connections: traci.close()
        except Exception: pass
        finally:
            if "default" in traci.connection._connections: del traci.connection._connections["default"]

        # Start SUMO via TraCI
        traci.start(["sumo", "-c", "/content/MARL-TSC-SUMO-Group/sumo/config.sumocfg"])
        
        # Observation managers must be created after TraCI is initialized
        self.observations_manager = {agent: MARLObservation(self.agent_to_sumo[agent]) for agent in self.agents}
        self.last_waiting_times = {agent: 0 for agent in self.agents}
        self.current_phase_timer = {agent: 0 for agent in self.agents}
        
        return {agent: self.observations_manager[agent].get() for agent in self.agents}

    def step(self, actions):
        """
        Execute actions, advance simulation, and compute SGAT-based rewards.
        """
        for agent, action in actions.items():
            self._apply_action(agent, action)
            
        traci.simulationStep()
        
        for agent in self.agents: 
            self.current_phase_timer[agent] += 1

        obs = {agent: self.observations_manager[agent].get() for agent in self.agents}
        rewards = {agent: self._compute_reward(agent) for agent in self.agents}
        
        # Standardization for PettingZoo Parallel API
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return obs, rewards, terminations, truncations, infos

    def _get_hetero_index(self, agent):
        """
        Implements Equations (6) and (7) from the paper.
        Calculates correlation indices based on lane counts and distances between intersections.
        """
        # Placeholder values for structural parameters (m_p and x_p)
        # In production, these should be parsed from the .net.xml file.
        m_p_i = 3           # Lane count at current intersection
        m_p_upstream = 3    # Lane count at upstream intersection
        m_p_downstream = 3  # Lane count at downstream intersection
        dist_upstream = 500.0
        dist_downstream = 500.0
        
        # Eq (6): Upstream Heterogeneous Correlation
        cor_forward = m_p_upstream / (m_p_i * self.q_scaling * np.sqrt(dist_upstream))
        
        # Eq (7): Downstream Heterogeneous Correlation
        cor_backward = m_p_downstream / (m_p_i * self.q_scaling * np.sqrt(dist_downstream))
        
        return cor_forward, cor_backward

    def _compute_reward(self, agent):
        """
        Implements Equation (8) for Reward Calculation.
        Multiplies the local delay impact by the product of forward and backward correlation indices.
        """
        sumo_id = self.agent_to_sumo[agent]
        lanes = traci.trafficlight.getControlledLanes(sumo_id)
        
        # Calculate local average delay (change in waiting time)
        current_waiting_time = sum([traci.lane.getWaitingTime(l) for l in lanes])
        delay = (self.last_waiting_times[agent] - current_waiting_time) / 100.0
        
        # Fetch Heterogeneity indices
        cor_f, cor_b = self._get_hetero_index(agent)
        
        # Final SGAT Reward Formulation:
        # Reward = (Volume_Ratio) * Cor_Forward * Cor_Backward * Delay
        # We use a default Volume_Ratio of 1.0 for the off-peak (u=0) scenario.
        reward = 1.0 * cor_f * cor_b * delay
        
        self.last_waiting_times[agent] = current_waiting_time
        return float(reward)

    def _apply_action(self, agent, action):
        """
        Apply phase duration adjustments and handle logic for yellow phase transitions.
        """
        sumo_id = self.agent_to_sumo[agent]
        current_phase = traci.trafficlight.getPhase(sumo_id)
        
        # Handle yellow phase logic (fixed 3s duration, no adjustments allowed)
        if current_phase % 2 != 0:
            if self.current_phase_timer[agent] >= 3:
                traci.trafficlight.setPhase(sumo_id, (current_phase + 1) % 4)
                self.current_phase_timer[agent] = 0
            return
        
        # Apply adjustments to green phase durations (Actions 1 and 2)
        dur = traci.trafficlight.getPhaseDuration(sumo_id)
        if action == 1: 
            traci.trafficlight.setPhaseDuration(sumo_id, min(self.max_green, dur + self.delta_t))
        elif action == 2: 
            traci.trafficlight.setPhaseDuration(sumo_id, max(self.min_green, dur - self.delta_t))

        # Check if phase has reached its programmed end
        if self.current_phase_timer[agent] >= traci.trafficlight.getPhaseDuration(sumo_id):
            traci.trafficlight.setPhase(sumo_id, (current_phase + 1) % 4)
            self.current_phase_timer[agent] = 0
