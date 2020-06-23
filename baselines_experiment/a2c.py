import gym
import gym_sokoban

from stable_baselines.common.policies import MlpPolicy 
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# Parallel environments
env = make_vec_env('Sokoban-small-v0', n_envs=9)

model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=".a2c/")
model.learn(total_timesteps=1_000_000)
model.save(".a2c/a2c_boxoban")

del model # remove to demonstrate saving and loading

model = A2C.load(".a2c/a2c_boxoban")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode='human')
