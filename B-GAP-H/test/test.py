import gym
import highway_env

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h", "pre_x", "pre_y"],
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
env = gym.make('highway-v0')
env.configure(config)
obs = env.reset()
# print(obs)
# print(type(obsfornn_all))
for _ in range(30):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    print(_, ":", obs)
    # temp.append(obs)
    env.render()
