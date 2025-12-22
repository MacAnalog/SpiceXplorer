import logging
from typing import Type, Callable, Dict, Any, Union

# SB3 Imports
from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

# SpiceXplorer Imports
from spicexplorer.core.domains import AgentType
from spicexplorer.optimization.rl.utils.hyperparameters import AgentConfig, DDPGConfig, SACConfig

# ------------------ (1) Module Logger ------------------

logger = logging.getLogger("spicexplorer.optimization.rl.rl_factory")

# ------------------ (2) Classes ------------------

# Type hint for the registry value
ApiTuple = KeyError  # Placeholder, usually Tuple[Type, Callable]

class RLAgentFactory:
    """
    A registry and factory for creating RL agents. 
    Handles the translation between Domain Configs and Agent Framework (SB3) kwargs.
    """
    
    # Registry stores: { AgentType: (AgentClass, ConfigAdapterFunction) }
    _registry: Dict[AgentType, ApiTuple] = {}

    @classmethod
    def register(cls, 
                 agent_type: AgentType, 
                 agent_class: Type[BaseAlgorithm], 
                 adapter_func: Callable[[AgentConfig], Dict[str, Any]]):
        """
        Register a new agent type (standard or custom).
        
        Args:
            agent_type: Enum identifier.
            agent_class: The class to instantiate (e.g. PPO, or MyCustomAgent).
            adapter_func: A function that takes your domain Config object and 
                          returns a dictionary of kwargs for the agent_class constructor.
        """
        cls._registry[agent_type] = (agent_class, adapter_func)
        logger.info(f"Registered RL Agent: {agent_type.value} -> {agent_class.__name__}")

    @classmethod
    def create_agent(cls, 
                     agent_type: AgentType, 
                     env: Any, 
                     config: AgentConfig, 
                     **kwargs) -> BaseAlgorithm:
        """
        Instantiates an agent based on the type and configuration.
        """
        if agent_type not in cls._registry:
            raise ValueError(f"Agent type '{agent_type}' is not registered. Available: {list(cls._registry.keys())}")

        agent_class, adapter_func = cls._registry[agent_type]

        # 1. Adapt the Domain Config to specific Agent Kwargs (e.g., actor.lr -> learning_rate)
        #    This keeps your generic config clean and the agent implementation specific.
        agent_kwargs = adapter_func(config)

        # 2. Merge with any runtime overrides (like tensorboard paths passed from the optimizer)
        agent_kwargs.update(kwargs)

        # 3. Instantiate
        logger.info(f"Creating agent {agent_type.value} with params: {agent_kwargs.keys()}")
        return agent_class(env=env, **agent_kwargs)

# ------------------ (3) Adapter Functions for Each Agent Type ------------------

def _adapter_sb3_general(config: AgentConfig) -> Dict[str, Any]:
    """Common mappings for all SB3 agents."""
    return {
        "policy": "MlpPolicy",
        "learning_rate": config.actor.lr,  # SB3 usually uses one LR, or a schedule
        "gamma": config.training.gamma,
        "batch_size": config.memory.batch_size,
        "verbose": 1,
        # "seed": config.seed # If you added seed to config
    }

def _adapter_ppo(config: AgentConfig) -> Dict[str, Any]:
    params = _adapter_sb3_general(config)
    # PPO specific mappings
    # params['n_steps'] = config.training.update_every * ... 
    return params

def _adapter_sac(config: SACConfig) -> Dict[str, Any]:
    params = _adapter_sb3_general(config)
    params.update({
        "tau": config.training.tau,
        "train_freq": config.training.policy_update_freq,
        "ent_coef": "auto" if config.alpha.learn_alpha else config.alpha.alpha_init,
        # SAC specific network architecture passing
        "policy_kwargs": dict(net_arch=list(config.actor.hidden_units)) 
    })
    return params

def _adapter_ddpg(config: DDPGConfig) -> Dict[str, Any]:
    params = _adapter_sb3_general(config)
    params.update({
        "tau": config.training.tau,
        # DDPG Noise handling could be complex, might need to instantiate an ActionNoise object here
    })
    return params

# --- Bootstrapping the Registry ---
RLAgentFactory.register(AgentType.PPO, PPO, _adapter_ppo)
RLAgentFactory.register(AgentType.SAC, SAC, _adapter_sac)
RLAgentFactory.register(AgentType.DDPG, DDPG, _adapter_ddpg)
RLAgentFactory.register(AgentType.TD3, TD3, _adapter_ddpg) # TD3 uses similar config to DDPG