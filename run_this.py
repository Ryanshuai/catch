"""
Sarsa is a online updating method for Reinforcement learning.
Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.
You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(1000):
        observation = env.reset()# initial observation
        while True:
            env.render()# fresh env xuan ran
            action = RL.choose_action(observation)# RL choose action based on observation
            observation_, reward, done = env.step(action)# RL take action and get next observation and reward
            RL.store_transition(observation, action, reward, observation_)#save the memory
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            observation = observation_ # move to next state
            if done:# break while loop when end of this episode
                break
            step += 1
    print('game over')# end of game
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()