from spicexplorer.optimization.rl.hyperparameters import DDPGHyperparameters

# Load hyperparameters from the YAML file
try:
    hyperparams = DDPGHyperparameters.from_yaml("tests/ddpg_hyperparameters.yaml")

    # Print the loaded hyperparameters
    print("Loaded DDPG Hyperparameters:")
    print(hyperparams)

    # Access nested parameters
    print("\nActor learning rate:", hyperparams.actor.lr)
    print("Actor hidden units:", hyperparams.actor.hidden_units)
    print("Critic grad clip:", hyperparams.critic.grad_clip)
    print("Memory buffer size:", hyperparams.memory.buffer_size)
    print("Training gamma:", hyperparams.training.gamma)

    # Convert to dictionary for compatibility with the agent
    hyperparams_dict = hyperparams.to_dict()
    print("\nHyperparameters as dictionary:")
    print(hyperparams_dict)

except FileNotFoundError:
    print("Error: 'ddpg_hyperparameters.yaml' not found. Make sure the file exists in the 'example' directory.")
except Exception as e:
    print(f"An error occurred: {e}")