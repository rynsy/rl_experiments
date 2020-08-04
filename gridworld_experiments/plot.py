import json
import matplotlib.pyplot as plt
import numpy as np

env_ids = [
        'MiniGrid-DoorKey-5x5-v0',
        'MiniGrid-DoorKey-6x6-v0',
        'MiniGrid-DoorKey-8x8-v0',
        'MiniGrid-DoorKey-16x16-v0',
        ]
env_colors = [
        'blue',
        'green',
        'red',
        'black'
        ]

eval_step = 0

file_path = ".a2c/training-output.json"
with open(file_path, "r") as file:
    data = json.load(file)

x = np.linspace(0, len(data), len(data))
lines = [
        np.zeros(len(data)),
        np.zeros(len(data)),
        np.zeros(len(data)),
        np.zeros(len(data))
    ]

def passing_test(data):
    return round(data['model_mean'] - data['model_std'], 3) >= round(data['baseline_mean'] - data['baseline_std'], 3)

for idx, env_data in data.items():
    success = True
    step = int(idx)
    last_trained = eval_step % len(env_ids)

    for i in env_ids:
        lines[env_ids.index(i)][eval_step] = env_data[i]['model_mean']

    for env_name, data in env_data.items():
        point_color = env_colors[env_ids.index(env_name)]
        
        
        print("Round {} on environment: {}".format(idx, env_name))
        print("\tThe learner was trained on environment: {}".format(env_ids[last_trained]))
        if passing_test(data):
            print("\t\tThe test was a success")
        else:
            if env_name in env_ids[0:last_trained]:
                success = False
            print("\t\tThe test failed")
        print("\n")
    if success:
        eval_step += 1
    else:
        eval_step = 0
for i in range(len(lines)):
    plt.plot(x, lines[i], marker='', color=env_colors[i], linewidth=1, alpha=0.9, label=env_ids[i])
plt.show()
