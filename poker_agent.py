import rlcard
from rlcard.agents import RandomAgent, CFRAgent
from rlcard.envs.registration import register, make
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch.nn as nn
import torch
import optuna
import gymnasium
from gymnasium import spaces

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable tensor cores for better performance if available
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

class HoldemEnvWrapper(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.env = make('limit-holdem')
        
        # RLCard's limit hold'em uses a 72-dimensional observation space
        # and 4 actions (fold, check/call, raise, all-in)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(72,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        state, _ = self.env.reset()
        return self._extract_state(state), {}

    def step(self, action):
        # RLCard returns (next_state, reward) tuple
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 2:
            next_state, reward = result
            done = False  # RLCard doesn't return done status for limit holdem
        else:
            next_state, reward, done = result
        
        return self._extract_state(next_state), reward, done, done, {}

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        np.random.seed(seed)

    def _extract_state(self, state):
        """Convert the RLCard state dict into a flat numpy array"""
        # RLCard's observation is a dict with an 'obs' key containing the actual observation
        obs = state['obs']
        if isinstance(obs, tuple):
            # If observation is a tuple, convert it to a flat array
            obs = np.concatenate([o.flatten() if isinstance(o, np.ndarray) else np.array([o]) for o in obs])
        return obs.astype(np.float32)

# Define a more granular action space (example)
def get_action_fraction(action_id, min_raise, max_raise):
    if action_id == 0:
      return "fold"
    if action_id == 1:
      return "check"
    if action_id == 2:
      return "call"
    
    # Define raise sizes as fractions of the pot (or multiples of the big blind)
    fracs = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] 
    raise_frac = fracs[action_id - 3] # because 0,1,2 are fold, check, call

    # Make sure we don't raise less than the minimum allowed raise or the call amount
    raise_amount = max(min_raise, int(raise_frac * min_raise))

    # Make sure we don't raise more than our stack or the maximum allowed raise
    raise_amount = min(raise_amount, max_raise)
    return ("raise", raise_amount)

# Custom Neural Network Architecture
class CustomHoldemNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        n_input = np.prod(observation_space.shape)
        
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Convert observations to tensor if they're not already
        if isinstance(observations, np.ndarray):
            observations = torch.FloatTensor(observations)
        return self.net(observations.float())

# Create Opponent Pool
def create_opponent_pool(num_random=2, num_cfr=1):
    """Create a pool of opponents"""
    # Create a temporary environment to get the number of actions
    temp_env = make('limit-holdem')
    opponent_pool = []
    
    # Add random agents
    for _ in range(num_random):
        opponent_pool.append(RandomAgent(num_actions=temp_env.num_actions))
    
    # Add CFR agents
    for _ in range(num_cfr):
        opponent_pool.append(CFRAgent(env=temp_env))
    
    return opponent_pool

class PokerEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, opponent_pool=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.opponent_pool = opponent_pool
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate against each opponent in the pool
            rewards = []
            for opponent in self.opponent_pool:
                self.eval_env.env.set_agents([None, opponent])
                episode_rewards = []
                
                for _ in range(5):  # Run 5 episodes per opponent
                    obs, _ = self.eval_env.reset()
                    done = False
                    episode_reward = 0
                    
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                        done = terminated or truncated
                        episode_reward += reward
                        
                    episode_rewards.append(episode_reward)
                
                mean_reward = np.mean(episode_rewards)
                rewards.append(mean_reward)
            
            mean_reward = np.mean(rewards)
            self.last_mean_reward = mean_reward
            
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f}")
                print("==========================================")
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print("New best mean reward!")
                
        return True

# Hyperparameter Tuning with Optuna (Example)
def objective(trial):
    # Calculate optimal batch size (power of 2)
    n_steps = trial.suggest_int("n_steps", 1024, 8192, 1024)
    n_envs = 8  # Increased number of environments
    total_timesteps = n_steps * n_envs
    batch_size = 2 ** trial.suggest_int("batch_size_exp", 6, 11)  # 64 to 2048
    batch_size = min(batch_size, total_timesteps)  # Ensure batch size isn't larger than buffer
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    policy_kwargs = dict(
        features_extractor_class=CustomHoldemNetwork,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    
    model = PPO("MlpPolicy", env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                ent_coef=ent_coef,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=1)
    
    try:
        model.learn(total_timesteps=200000, callback=PokerEvalCallback(eval_env=HoldemEnvWrapper(), eval_freq=10000, opponent_pool=create_opponent_pool(), verbose=0))  # Increase training budget
    except Exception as e:
        print(e)
        return -float('inf')  # Return a bad value if training fails

    # Evaluate the final agent
    total_reward = 0
    for _ in range(100):
        obs, info = HoldemEnvWrapper().reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = HoldemEnvWrapper().step(action)
            done = terminated or truncated
        total_reward += info['payoffs'][0]

    avg_reward = total_reward / 100
    return avg_reward

def make_env(rank=0):
    """
    Create a wrapped environment for vectorized environments
    """
    def _thunk():
        env = HoldemEnvWrapper()
        np.random.seed(rank)  # Set the seed for the environment
        return env
    return _thunk  # Return the function, not the result

if __name__ == "__main__":
    # Number of environments should be a multiple of CPU cores
    n_envs = 8
    
    # Create vectorized environment using SubprocVecEnv for parallel processing
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    # Create evaluation environment
    eval_env = HoldemEnvWrapper()
    opponent_pool = create_opponent_pool()
    eval_env.env.set_agents([None, opponent_pool[0]])
    
    # Create the model with optimized parameters
    policy_kwargs = dict(
        features_extractor_class=CustomHoldemNetwork,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,  # Increased steps for better learning
        batch_size=256,  # Power of 2 for better GPU utilization
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1
    )
    
    # Create callback for evaluation
    eval_callback = PokerEvalCallback(
        eval_env=eval_env,
        eval_freq=10000,
        opponent_pool=opponent_pool,
        verbose=1
    )
    
    # Run optimization with increased trials
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train final model with best parameters
    best_params = study.best_params
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=best_params["learning_rate"],
        n_steps=best_params["n_steps"],
        batch_size=best_params["batch_size"],
        ent_coef=best_params["ent_coef"],
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1
    )
    
    # Train with increased timesteps
    model.learn(total_timesteps=500000, callback=eval_callback)
    
    # Save the model
    model.save("poker_model_final")
    
    # Clean up
    env.close()
    eval_env.close()