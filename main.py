import pygame as pg

from environment import Game
from agent import Agent

if __name__ == "__main__":
    env = Game(human=True, grid=True, infos=True, progress_bars=True)
    agent = Agent(
        gamma=0.9,
        epsilon=1.0,
        batch_size=64,
        n_actions=env.action_space,
        min_epsilon=0.01,
        input_dims=[env.state_space],
        lr=0.001,
    )

    while env.running:
        score = 0
        done = False
        state = env.reset()
        
        while not done:
            if not env.running:
                break
                        
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            agent.store_transitions(state, action, reward, new_state, done)
            agent.learn()
            state = new_state
            env.render(agent)

        agent.last_decision = agent.current_decision
        agent.n_exploration = 0
        agent.n_exploitation = 0
        
        env.rewards.append(env.reward_episode)
        env.n_episode += 1
    
    pg.quit()
