import gym
import highway_env
from matplotlib import pyplot as plt
import numpy as np

config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted",
    }
}
temp = []
times = 400
during = 40
agg_rate = 20
for i in range(times):
    print(i)
    env = gym.make('highway-v0')
    env.configure(config)
    obs = env.reset()
    # print("ori", obs)
    temp.append(obs)
    for _ in range(during):
        action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, info = env.step(action)
        #print(_, ":", obs)
        temp.append(obs)
        env.render()
# print("temp:", temp[1])
# print(type(temp))
# obs, info = env.reset()
# print(obs)
a = np.array(temp)
# print(a)
np.save('./npy/agg_'+ str(agg_rate) +'%_'+ str(during) +'s_'+ str(times) +'times.npy', a)
# a = np.load('aggressive.npy')
# a = a.tolist()
# print("a:", a)
# print(type(a))