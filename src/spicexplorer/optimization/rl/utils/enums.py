from enum import Enum

# --------------------------------------------
# General
# --------------------------------------------
class NoiseType(Enum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"

# --------------------------------------------
# RL Specific
# --------------------------------------------
class AgentType(Enum):
    DDPG = "ddpg"
    PPO = "ppo"
    SAC = "sac"
    TD3 = "td3"
    A2C = "a2c"

class ActorCriticType(Enum):
    MLP = "mlp"
    CNN = "cnn"
    RNN = "rnn"
    GAT = "gat"
    GCN = "gcn"

class ReplayBufferType(Enum):
    STANDARD = "standard"
    PRIORITIZED = "prioritized"

class ExplorationStrategy(Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    NOISE_ADDITION = "noise_addition"
    PARAMETER_NOISE = "parameter_noise"

# --------------------------------------------
# Simulation Specific
# --------------------------------------------
class SpiceSimulatorType(Enum):
    SPECTRE = "spectre"
    HSPICE  = "hspice"
    NGSPICE = "ngspice"

class SpiceAnalysisType(Enum):
    DC      = "dc"
    AC      = "ac"
    TRANSIENT = "transient"
    NOISE  = "noise"

