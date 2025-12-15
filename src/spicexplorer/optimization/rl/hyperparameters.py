import yaml
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Type, TypeVar, Tuple

T = TypeVar("T")

def _from_dict(cls: Type[T], data: dict) -> T:
    """Recursively constructs a dataclass instance from a dictionary."""
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            if is_dataclass(f.type):
                kwargs[f.name] = _from_dict(f.type, data[f.name])
            else:
                kwargs[f.name] = data[f.name]
    return cls(**kwargs)

@dataclass
class BaseHyperparameters:
    @classmethod
    def from_yaml(cls: Type[T], file_path: str) -> T:
        """Loads hyperparameters from a YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return _from_dict(cls, data)

# --------------------------------------------
# Hyperparameters by Agent Components
# --------------------------------------------
@dataclass
class ActorHyperparameters(BaseHyperparameters):
    lr: float = 0.001
    hidden_units: Tuple[int, ...] = (256, 128)

@dataclass
class CriticHyperparameters(BaseHyperparameters):
    lr: float = 0.001
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    hidden_units: Tuple[int, ...] = (256, 128)

@dataclass
class NoiseHyperparameters(BaseHyperparameters):
    type: str = "gaussian"
    sigma_initial: float = 0.2
    sigma_min: float = 0.01
    sigma_decay: float = 0.995

@dataclass
class MemoryHyperparameters(BaseHyperparameters):
    buffer_size: int = 100000
    batch_size: int = 64

@dataclass
class TrainingHyperparameters(BaseHyperparameters):
    gamma: float = 0.99
    tau: float = 0.005
    update_every: int = 1
    initial_random_steps: int = 1000

# --------------------------------------------
# Hyperparameters by Agent Type
# --------------------------------------------

# DDPG Hyperparameters
@dataclass
class DDPGHyperparameters(BaseHyperparameters):
    actor: ActorHyperparameters = field(default_factory=ActorHyperparameters)
    critic: CriticHyperparameters = field(default_factory=CriticHyperparameters)
    noise: NoiseHyperparameters = field(default_factory=NoiseHyperparameters)
    memory: MemoryHyperparameters = field(default_factory=MemoryHyperparameters)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)

    def to_dict(self) -> dict:
        """Converts the hyperparameters to a dictionary for compatibility."""
        return {
            "lr_actor": self.actor.lr,
            "hidden_units_actor": self.actor.hidden_units,
            "lr_critic": self.critic.lr,
            "weight_decay_critic": self.critic.weight_decay,
            "grad_clip_critic": self.critic.grad_clip,
            "hidden_units_critic": self.critic.hidden_units,
            "noise_type": self.noise.type,
            "noise_sigma_initial": self.noise.sigma_initial,
            "noise_sigma_min": self.noise.sigma_min,
            "noise_sigma_decay": self.noise.sigma_decay,
            "buffer_size": self.memory.buffer_size,
            "batch_size": self.memory.batch_size,
            "gamma": self.training.gamma,
            "tau": self.training.tau,
            "update_every": self.training.update_every,
            "initial_random_steps": self.training.initial_random_steps,
        }