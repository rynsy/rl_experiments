import gym
import gym_sokoban
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines.common.policies import CnnLstmPolicy 
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

# Parallel environments
env = make_vec_env('Sokoban-small-v0', n_envs=8)

model = PPO2(CnnLstmPolicy, env, verbose=1, tensorboard_log=".ppo/")
model.learn(total_timesteps=10_000_000)
model.save(".ppo/ppo_boxoban")

del model # remove to demonstrate saving and loading

model = PPO2.load(".ppo/ppo_boxoban")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode='human')
