import gym
import gym_sokoban
import tensorflow as tf
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines import A2C
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy 

path = F".a2c/" 
model_path = path + "MODEL/"
graph_path = path + "GRAPH/"
pre_trained_path = model_path + "sokoban_small_a2c"
checkpoint_path = model_path + "sokoban_small_a2c_checkpoint"
MAX_STEPS = 200_000

env_id = 'Sokoban-small-v0'

def get_model(vec_env):
    return A2C(CnnPolicy, 
                vec_env,
                tensorboard_log=graph_path,
                verbose=1
            )

def get_vec_env():
    return make_vec_env(env_id, n_envs=9)

def get_eval_env():
    return gym.make(env_id)

def single_boards():
    env = get_vec_env()
    model = get_model(env)
    for i in range(100):
        model.learn(total_timesteps=MAX_STEPS)
        model.save(checkpoint_path)
        env = get_vec_env()
        model.set_env(env)

if __name__ == "__main__":
    single_boards()
