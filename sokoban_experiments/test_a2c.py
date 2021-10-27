import gym
import gym_sokoban

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# Parallel environments
env = make_vec_env('Sokoban-small-v0', n_envs=9)

model = A2C.load(".a2c/MODEL/sokoban_small_a2c")

while True:
    obs = env.reset()
    t = 0
    while t < 200:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render(mode='human')
        t += 1
        if dones.any():
            print("SOLVED!")
