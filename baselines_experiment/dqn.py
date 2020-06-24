import gym
import gym_sokoban

from stable_baselines.deepq.policies import MlpPolicy 
from stable_baselines.common.vec_env import DummyVecEnv 
from stable_baselines import DQN

env = gym.make('Sokoban-small-v0')

model = DQN(MlpPolicy, env, 
        verbose=1, 
        tensorboard_log=".dqn/", 
        double_q=True, 
        prioritized_replay=True
        )

model.learn(total_timesteps=1_000_000)
model.save(".dqn/dqn_boxoban")

del model # remove to demonstrate saving and loading

model = DQN.load(".dqn/dqn_boxoban")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode='human')
