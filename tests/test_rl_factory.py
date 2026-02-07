import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

# --- 1. MOCKING DEPENDENCIES (If SB3 is not installed) ---
# This allows you to test the factory logic without installing PyTorch/SB3 yet.
try:
    import stable_baselines3
except ImportError:
    print("⚠️  Stable Baselines3 not found. Mocking it for testing purposes.")
    sb3_mock = MagicMock()
    sys.modules["stable_baselines3"] = sb3_mock
    sys.modules["stable_baselines3.common.base_class"] = MagicMock()
    # Mock specific agent classes
    sb3_mock.PPO = MagicMock(__name__="PPO")
    sb3_mock.SAC = MagicMock(__name__="SAC")
    sb3_mock.DDPG = MagicMock(__name__="DDPG")
    sb3_mock.TD3 = MagicMock(__name__="TD3")

# --- 2. IMPORT USER MODULES ---
# We assume your files are named 'rl_factory.py' and 'domain.py'
try:
    from spicexplorer.optimization.rl.domain import (
        AgentType, AgentConfig, DDPGConfig, SACConfig, 
        NetworkConfig, RLTrainingConfig, ReplayBufferConfig, 
        SACAlphaConfig, NoiseConfig
    )
    from rl_factory import RLAgentFactory
except ImportError as e:
    print(f"❌ Critical Error: Could not import your modules. {e}")
    print("Ensure 'domain.py' and 'rl_factory.py' are in the current folder.")
    sys.exit(1)

# --- 3. TEST SUITE ---
class TestRLFactory(unittest.TestCase):
    
    def setUp(self):
        """Setup generic dummy configs for testing."""
        self.dummy_env = MagicMock()
        
        # Create a basic generic config
        self.base_config = AgentConfig(
            actor=NetworkConfig(lr=1e-4, hidden_units=(64, 64)),
            training=RLTrainingConfig(gamma=0.95, batch_size=32),
            memory=ReplayBufferConfig(batch_size=32)
        )

    def test_ppo_adapter_translation(self):
        """Test if AgentConfig is correctly translated to PPO kwargs."""
        # PPO usually uses the standard AgentConfig
        agent = RLAgentFactory.create_agent(
            AgentType.PPO, 
            self.dummy_env, 
            self.base_config
        )
        
        # Verify the agent was initialized with translated params
        # PPO class should have been called with learning_rate=1e-4, etc.
        from stable_baselines3 import PPO
        PPO.assert_called_once()
        
        call_kwargs = PPO.call_args[1]
        self.assertEqual(call_kwargs['learning_rate'], 1e-4)
        self.assertEqual(call_kwargs['gamma'], 0.95)
        self.assertEqual(call_kwargs['batch_size'], 32)
        print("✅ PPO Adapter translation verified.")

    def test_sac_adapter_translation(self):
        """Test SAC specific translation (Tau, Ent Coef, etc)."""
        # SAC requires specific SACConfig
        sac_config = SACConfig(
            actor=NetworkConfig(lr=3e-4, hidden_units=(256, 256)),
            training=RLTrainingConfig(tau=0.01, policy_update_freq=2),
            alpha=SACAlphaConfig(learn_alpha=False, alpha_init=0.5)
        )
        
        agent = RLAgentFactory.create_agent(
            AgentType.SAC, 
            self.dummy_env, 
            sac_config
        )
        
        from stable_baselines3 import SAC
        call_kwargs = SAC.call_args[1]
        
        # Check specific mappings
        self.assertEqual(call_kwargs['tau'], 0.01)
        self.assertEqual(call_kwargs['ent_coef'], 0.5) # alpha_init because learn_alpha is False
        self.assertEqual(call_kwargs['train_freq'], 2)
        print("✅ SAC Adapter translation verified.")

    def test_custom_agent_registration(self):
        """Test registering and creating a user-defined custom agent."""
        
        # 1. Define a Mock Custom Agent
        class MyGeneticAgent:
            def __init__(self, env, mutation_rate, **kwargs):
                self.env = env
                self.mutation_rate = mutation_rate
        
        # 2. Define a Custom Config & Adapter
        @dataclass
        class GeneticConfig(AgentConfig):
            mutation_rate: float = 0.9

        def adapter_genetic(cfg: GeneticConfig) -> Dict[str, Any]:
            return {
                "mutation_rate": cfg.mutation_rate,
                "verbose": 1
            }

        # 3. Register it
        RLAgentFactory.register(AgentType.CUSTOM, MyGeneticAgent, adapter_genetic)
        
        # 4. Create it
        gen_config = GeneticConfig(mutation_rate=0.75)
        agent = RLAgentFactory.create_agent(
            AgentType.CUSTOM, 
            self.dummy_env, 
            gen_config
        )
        
        # 5. Verify
        self.assertIsInstance(agent, MyGeneticAgent)
        self.assertEqual(agent.mutation_rate, 0.75)
        print("✅ Custom Agent registration verified.")

    def test_unknown_agent_error(self):
        """Ensure factory raises error for unregistered enums."""
        # Hack an invalid enum into existence for testing
        class FakeType:
            value = "fake"
        
        with self.assertRaises(ValueError):
            RLAgentFactory.create_agent(FakeType(), self.dummy_env, self.base_config)
        print("✅ Unknown agent error handling verified.")

if __name__ == "__main__":
    unittest.main()