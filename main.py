from environment import Game
from agent import Agent
from utils_2020 import plotLearning

import numpy as np

if __name__ == "__main__":
    env = Game(human=False, grid=True, infos=False)
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=env.action_space,
        eps_end=0.01,
        input_dims=[env.state_space],
        lr=0.003,
    )
    scores, eps_history = [], []
    episode = 0

    while env.running:
        score = 0
        done = False
        state = env.reset()
        
        while not done:
            if not env.running:
                break
            
            env.render()
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transitions(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
            
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        print(
            "episode ",
            episode,
            "score %.2f" % score,
            "average score %.2f" % avg_score,
            "epsilon %.2f" % agent.epsilon,
        )
        
        episode += 1
