import gym
import gym_minigrid
from gym_minigrid.wrappers import *

import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import A2C
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

env_ids = [
        'MiniGrid-DoorKey-5x5-v0',
        'MiniGrid-DoorKey-6x6-v0',
        'MiniGrid-DoorKey-8x8-v0',
        'MiniGrid-DoorKey-16x16-v0',
        ]

num_cpu = 4
max_steps = 500_000
log_dir = ".a2c/"
data_dir = ".a2c/"
model_name = "latest_model"
verbose = 0
parallel = True
terminate_early = False
pretrained = False
pretrained_model = ".a2c/a2c_6x6.zip"
performance_data = {}
performance_file = "training-output.json"
output_file = "training-output.txt"
eval_step = 0
THRESHOLD = 0.9

if __name__ == "__main__":
    def write_out(text):
        print(text)
        with open(data_dir + output_file, "a") as outfile:
            outfile.write(text)

    def make_env(env_id, rank, seed=0):
        def _init():
            env = gym.make(env_id)
            env = RGBImgPartialObsWrapper(env)
            env = ImgObsWrapper(env)
            env.seed(seed+rank)
            return env
        set_global_seeds(seed)
        return _init

    def init_env(env_id):
        if parallel:
            env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
            reward_env = SubprocVecEnv([make_env(env_id, i) for i in range(1)])
        else:
            env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])
            reward_env = DummyVecEnv([make_env(env_id, i) for i in range(1)])
        if terminate_early:
            callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.85, verbose=verbose)
            eval_callback = EvalCallback(reward_env, callback_on_new_best=callback_on_best, eval_freq=10_000, verbose=verbose)
            return env, reward_env, eval_callback
        else:
            return env, reward_env, None

    def eval_model(model, test_env_id):
        global eval_step, THRESHOLD
        test_success, curriculum_success = True, True
        performance_data[eval_step] = {}
        for env_id in env_ids:
            write_out("[MODEL EVAL]\tTesting learner on env: {}".format(env_id))
            env, eval_env, eval_callback = init_env(env_id)

            fresh_model = A2C(CnnPolicy, env, verbose=verbose)
            fresh_model.learn(total_timesteps=max_steps, callback=eval_callback)
            
            fresh_mean, fresh_std = evaluate_policy(fresh_model, eval_env, n_eval_episodes=100)
            model_mean, model_std = evaluate_policy(model, eval_env, n_eval_episodes=100)
            performance_data[eval_step][env_id] = {
                                            'baseline_mean'                 : fresh_mean,
                                            'baseline_std'                  : fresh_std,
                                            'model_mean'                    : model_mean,
                                            'model_std'                     : model_std,
                                            'baseline_training_steps'       : max_steps,
                                            'eval_episodes'                 : 100
                                            } 
            write_out("[MODEL EVAL: LEARNER] \t env_id: {}, Mean Reward: {}, std_dev: {}".format(env_id, model_mean, model_std))
            write_out("[MODEL EVAL: BASELINE]\t env_id: {}, Mean Reward: {}, std_dev: {}".format(env_id, fresh_mean, fresh_std))
            
            pass_test = round(model_mean - model_std, 3) >= round(fresh_mean - fresh_std, 3)
            diff = abs(round(model_mean - model_std, 3) - round(fresh_mean - fresh_std, 3))
            if pass_test:
                write_out("[TEST RESULT]\tmodel out-performs fresh model for env: {}, diff: {}".format(env_id, diff))
            else:
                write_out("[TEST RESULT]\tmodel DID NOT out-perform fresh model for env: {}, diff: {}".format(env_id, diff))
                if env_id == test_env_id:
                    test_success = False

        curriculum_success = sum([performance_data[eval_step][env_id]['baseline_mean'] > THRESHOLD for env_id in env_ids]) == len(env_ids)
        eval_step += 1
        return test_success, curriculum_success

    i = 0
    env_id = env_ids[i]
    env, reward_env, eval_callback = init_env(env_id)
    if not pretrained:
        model = A2C(CnnPolicy, env, verbose=verbose)
    else:
        model = A2C.load(pretrained_model)

    curriculum_learned = False
    while i < len(env_ids) or not curriculum_learned:
        env_id = env_ids[i % len(env_ids)]
        env, reward_env, eval_callback = init_env(env_id)
        model.set_env(env) 
        write_out("[TRAINING: START]\tlearning env: {}".format(env_id))
        model.learn(total_timesteps=max_steps, callback=eval_callback)
        eval_env = make_env(env_id, 0)()
        mean_reward, std_reward = evaluate_policy(model, eval_env)
        write_out("[TRAINING: FINISH]\tID: {}, Mean Reward: {}, std_dev: {}".format(env_id, mean_reward, std_reward))
        test_res, curriculum_learned = eval_model(model, env_id)
        if test_res:
            model.save(data_dir + model_name)
            with open(data_dir + performance_file, 'w') as outfile:
                json.dump(performance_data, outfile)
            i += 1
        elif curriculum_learned:
            write_out("[SUCCESS]\t\tSuccessfully learned the full curriculum, model beating reward threshold on all environments")
            sys.exit(0)
        else:
            i = 0
