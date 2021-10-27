import gym
import gym_minigrid
from gym_minigrid.wrappers import *

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import A2C
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

env_id = 'MiniGrid-DoorKey-5x5-v0'
max_steps = 10_000_000
log_dir = ".a2c/"
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
        reward_env = SubprocVecEnv([make_env(env_id, i) for i in range(1)])
    else:
        env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])
        reward_env = DummyVecEnv([make_env(env_id, i) for i in range(1)])

    eval_env = make_env(env_id, 0)()

    model = A2C(CnnPolicy, env, verbose=1, tensorboard_log=log_dir)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
    eval_callback = EvalCallback(reward_env, callback_on_new_best=callback_on_best, eval_freq=10_000, verbose=1)

    model.learn(total_timesteps=max_steps, callback=eval_callback)
    model.save(log_dir + "a2c_minigrid")
    del model # remove to demonstrate saving and loading

    model = A2C.load(log_dir + "a2c_minigrid")

    eval_env.reset()
    mean_reward, std_reward = evaluate_policy(model, eval_env)
    print("Mean Reward: {}, std_dev: {}".format(mean_reward, std_reward))


    demo = input("Watch model? (q to quit)")
    if demo != "q":
        for _ in range(1000):
            obs = eval_env.reset()
            t = 0
            while t < 200:
                action, _states = model.predict(obs)
                obs, rewards, done, info = eval_env.step(action)
                print("Action: {}\tTimestep: {}".format(action,t))
                eval_env.render(mode='human')
                t += 1
                if done:
                    break
