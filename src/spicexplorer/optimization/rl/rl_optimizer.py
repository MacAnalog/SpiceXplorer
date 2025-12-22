

from spicexplorer.core.domains import Project_Setup

from ..base import Spice_Constraint_Satisfaction, Spice_Single_Objective
from .circuit_env import SpiceGymEnv


# Optimization interface

class RL_Spice_Optimizer(Spice_Constraint_Satisfaction):
    """
    RL-based optimizer using Stable-Baselines3.
    Inherits simulation and scoring logic from Spice_Constraint_Satisfaction.
    """
    def __init__(self, setup_obj: Project_Setup, spicelib_wrapper: NGSpice_Wrapper):
        super().__init__(setup_obj=setup_obj, spicelib_wrapper=spicelib_wrapper)
        self.env = None
        self.model = None

    # --- 1. Setup Phase ---
    def parameterize(self) -> Any:
        # RL uses the Gym Env spaces, so we just return the param definitions for reference
        return self.setup_obj.dut_params

    def _create_optimizer_obj(self) -> bool:
        """Initialize the Gym Environment and the PPO Agent."""
        logger.info("Initializing RL Environment and Agent...")
        
        # Create the Env, passing 'self.evaluate' as the callback
        self.env = SpiceGymEnv(setup_obj=self.setup_obj, eval_callback=self.evaluate_adapter)
        
        # Initialize Agent (PPO is generally robust for continuous control)
        self.optimizer = ...
        return True

    # --- 2. Adaptation Layer ---
    def evaluate_adapter(self, action_dict: Dict[str, float], is_normalized_input: bool = False):
        """
        Adapts the Gym call to the Base_Optimizer evaluate() signature.
        Handles the denormalization if the input comes from the RL agent.
        """
        # 1. Denormalize if coming from RL Agent ([-1, 1] -> [1u, 10u])
        if is_normalized_input:
            # We must map [-1, 1] to the parameter ranges
            # NOTE: Your base denormalize expects 0..1 or raw values depending on implementation
            # You might need a specific helper here to map [-1, 1] to your config's range
            real_params = self._map_gym_action_to_physical(action_dict)
        else:
            real_params = action_dict

        # 2. Call the parent class's evaluate logic
        return super().evaluate(real_params)

    def _map_gym_action_to_physical(self, norm_params: Dict[str, float]) -> Dict[str, float]:
        """Maps Gym's [-1, 1] range to physical values."""
        phys_params = {}
        for name, val in norm_params.items():
            param = self.setup_obj.get_param_by_name(name)
            # Linear mapping: -1 -> min, +1 -> max
            # x_real = min + (val + 1)/2 * (max - min)
            phys_params[name] = param.min_val + (val + 1.0)/2.0 * (param.max_val - param.min_val)
        return phys_params

    # --- 3. Execution Phase ---
    def optimization_step(self) -> Tuple[Dict[str, np.floating], np.floating, Dict[str, Any]]:
        """
        In RL, one 'step' in the main loop can correspond to a block of training.
        """
        # Train the agent for a few timesteps (e.g. 1 episode worth)
        steps_per_trial = 100 
        self.model.learn(total_timesteps=steps_per_trial, reset_num_timesteps=False)
        
        # Return the last recorded point from the log so the main loop can track progress
        last_entry = self.optimization_log[-1]
        return last_entry.point.params, last_entry.point.score, last_entry.fit_summary