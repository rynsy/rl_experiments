import gym
import gym_minigrid
from gym_minigrid.wrappers import *

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

env_id = 'MiniGrid-DoorKey-16x16-v0'
max_steps = 1_000_000
log_dir = ".ppo/"
num_cpu = 8
parallel = True

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        env.seed(seed+rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == "__main__":
    # Parallel environments
    if parallel:
        env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    else:
        env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=max_steps)
    model.save(log_dir + "ppo_minigrid")
    del model # remove to demonstrate saving and loading

    model = PPO2.load(log_dir + "ppo_minigrid")
    env = make_env(env_id, 0)()

    mean_reward, std_reward = evaluate_policy(model, env)
    print("Mean Reward: {}, std_dev: {}".format(mean_reward, std_reward))
    demo = input("Watch model? (q to quit)")
    if demo != "q":
        for _ in range(1000):
            obs = env.reset()
            t = 0
            while t < 200:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                print("Action: {}\tTimestep: {}".format(action,t))
                env.render(mode='human')
                t += 1
                if done:
                    break
