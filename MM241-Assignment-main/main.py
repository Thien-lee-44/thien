import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx
import numpy as np
# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
   render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 10

if __name__ == "__main__":
    # Reset the environment

    observation, info = env.reset(seed=42)
    st_policy= Policy2210xxx()
    ep = 0
    while ep < NUM_EPISODES:
        action = st_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info)
            observation, info = env.reset(seed=ep)
            observation["products"]=sorted(observation["products"],key=lambda x: x["size"][0]**2+x["size"][1]**2,reverse=True)
            ep += 1


    observation, info = env.reset(seed=42)
    gd_policy = GreedyPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = gd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        observation["products"]=sorted(observation["products"],key=lambda x: x["size"][0]**2+x["size"][1]**2,reverse=True)
        if terminated or truncated:
            print(info)
            observation, info = env.reset(seed=ep)
            observation["products"]=sorted(observation["products"],key=lambda x: x["size"][0]**2+x["size"][1]**2,reverse=True)
            ep += 1

    # Reset the environment

    # Test GreedyPolicy
   

    # Test RandomPolicy
    
    # Uncomment the following code to test your policy
    # # Reset the environment
    # observation, info = env.reset(seed=42)
    # print(info)

    # policy2210xxx = Policy2210xxx()
    # for _ in range(200):
    #     action = policy2210xxx.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(info)

    #     if terminated or truncated:
    #         observation, info = env.reset()

env.close()
