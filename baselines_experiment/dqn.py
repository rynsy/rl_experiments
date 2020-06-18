import gym
import gym_sokoban

from stable_baselines.deepq.policies import LnCnnPolicy 
from stable_baselines.common.vec_env import DummyVecEnv 
from stable_baselines import DQN

# Parallel environments
env = gym.make('Boxoban-Train-v0')

model = DQN(LnCnnPolicy, env, verbose=1, tensorboard_log=".dqn/")
model.learn(total_timesteps=1_000_000)
model.save(".dqn/dqn_boxoban")

del model # remove to demonstrate saving and loading

model = DQN.load(".dqn/dqn_boxoban")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode='human')
