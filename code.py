import rlcard
from rlcard.agents import RandomAgent, CFRAgent
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import torch.nn as nn
import numpy as np
import optuna
from gymnasium import spaces

class HoldemEnvWrapper(rlcard.envs.holdem.HoldemEnv):
    def __init__(self, env_config):
        super().__init__(env_config)
        # RLCard's observation space is a dict, but SB3 expects a flattened space
        # You may need a more complex observation space depending on your features
        self.observation_space = spaces.Box(low=-1, high=1, shape=(52*2 + 3,))  # Example: cards + pot + stack

        # Define your action space
        self.action_space = spaces.Discrete(10)  # Example: 10 actions (fold, call, raise levels)

    def _extract_state(self, state):
        # This is where you'll need to adapt to your specific needs.
        # How do you want to represent the game state for your agent?

        # Example: One-hot encode cards and concatenate with pot and stack size
        encoded_cards = np.zeros(52*2)  # 52 cards x 2 (player and community)
        for card in state['hand']:
            encoded_cards[self._encode_card(card)] = 1
        for card in state['public_cards']:
            encoded_cards[52 + self._encode_card(card)] = 1
        
        return np.concatenate((encoded_cards, [state['pot'], state['my_chips']]))

    def _encode_card(self, card):
        # Map card string (e.g., 'SA' for Ace of Spades) to a number between 0-51
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
        return rank_map[card[1]] * 4 + suit_map[card[0]]

    def _decode_action(self, action_id):
        # Assuming action_id is an integer from 0-9
        if action_id == 0:
            return 'fold'
        elif action_id == 1:
            return 'check' if 'check' in [a for a, _ in self.actions] else 'call'
        else:
            # Map remaining action_ids to raise amounts
            raise_fraction = (action_id - 2) / 7  # Scale from 0 to 1
            min_raise, max_raise = self.actions[-1][1]
            raise_amount = min_raise + (max_raise - min_raise) * raise_fraction
            return (self.actions[-1][0], int(raise_amount))
            
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
    def __init__(self, observation_space, features_dim=512):
        super(CustomHoldemNetwork, self).__init__(observation_space, features_dim)

        self.shared_net = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.policy_net = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

        self.value_net = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        shared_output = self.shared_net(observations)
        return self.policy_net(shared_output), self.value_net(shared_output)

# Create Opponent Pool
def create_opponent_pool(num_random=2, num_cfr=1):
    opponent_pool = []
    for _ in range(num_random):
        opponent_pool.append(RandomAgent(num_actions=env.num_actions))
    for _ in range(num_cfr):
        opponent_pool.append(CFRAgent(env)) # Note: CFR may require pre-training
    return opponent_pool

# Evaluation Callback with Exploitability Estimation (Simplified)
class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, opponent_pool, deterministic=True, verbose=0):
        super(EvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.opponent_pool = opponent_pool

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            # Evaluate against random agent
            total_reward_random = 0
            for _ in range(100):
                obs, info = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, rewards, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                total_reward_random += info['payoffs'][0]

            avg_reward_random = total_reward_random / 100
            print(f"Eval vs Random at timestep {self.num_timesteps}: Average Reward = {avg_reward_random}")

            # Simplified exploitability estimation (play against CFR)
            if any(isinstance(opp, CFRAgent) for opp in self.opponent_pool):
                total_reward_cfr = 0
                cfr_agent = [opp for opp in self.opponent_pool if isinstance(opp, CFRAgent)][0]
                self.eval_env.set_agents([None, cfr_agent])
                for _ in range(100):
                    obs, info = self.eval_env.reset()
                    done = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=self.deterministic)
                        obs, rewards, terminated, truncated, info = self.eval_env.step(action)
                        done = terminated or truncated
                    total_reward_cfr += info['payoffs'][0]

                avg_reward_cfr = total_reward_cfr / 100
                print(f"Eval vs CFR at timestep {self.num_timesteps}: Average Reward = {avg_reward_cfr}")

        return True

# Hyperparameter Tuning with Optuna (Example)
def objective(trial):
    env = make_vec_env('holdem-v2', n_envs=4, env_class=HoldemEnvWrapper)
    opponent_pool = create_opponent_pool()
    eval_env = rlcard.make('holdem-v2', config={'env_class': HoldemEnvWrapper})
    eval_env.set_agents([None, opponent_pool[0]])

    callback = EvalCallback(eval_env, eval_freq=10000, opponent_pool=opponent_pool)

    policy_kwargs = dict(
        features_extractor_class=CustomHoldemNetwork,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],  # Example architecture
    )

    model = PPO("MlpPolicy", env,
                policy_kwargs=policy_kwargs,
                learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                n_steps=trial.suggest_int("n_steps", 1024, 8192),
                batch_size=trial.suggest_int("batch_size", 64, 512),
                ent_coef=trial.suggest_float("ent_coef", 0.005, 0.05, log=True),
                verbose=0)

    try:
        model.learn(total_timesteps=200000, callback=callback)  # Increase training budget
    except Exception as e:
        print(e)
        return -float('inf')  # Return a bad value if training fails

    # Evaluate the final agent
    total_reward = 0
    for _ in range(100):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        total_reward += info['payoffs'][0]

    avg_reward = total_reward / 100
    return avg_reward

# Wrap RLCard environment for Stable Baselines3 compatibility
# Create the environment using the wrapper
env = make_vec_env('holdem-v2', n_envs=4, env_class=HoldemEnvWrapper)
# Instantiate the agent with the custom network
opponent_pool = create_opponent_pool()
eval_env = rlcard.make('holdem-v2', config={'env_class': HoldemEnvWrapper})
eval_env.set_agents([None, opponent_pool[0]])  # Start with a random opponent

# Define the evaluation callback
callback = EvalCallback(eval_env, eval_freq=10000, opponent_pool=opponent_pool)

policy_kwargs = dict(
    features_extractor_class=CustomHoldemNetwork,
    features_extractor_kwargs=dict(features_dim=512),
    net_arch=[dict(pi=[256,256], vf=[256,256])],
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.0005, n_steps=4096, batch_size=256, ent_coef=0.01)
# Train the agent with the callback
# model.learn(total_timesteps=100000, callback=callback)

# Run hyperparameter optimization (optional)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Train the final agent with the best hyperparameters
best_params = study.best_params
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, **best_params)
model.learn(total_timesteps=200000, callback=callback)  # Train longer with best params