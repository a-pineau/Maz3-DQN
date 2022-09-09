"""Implements the game loop and handles the user's events."""

import os
import random
import numpy as np
import pygame as pg
import constants as const

from utils import message, distance

vec = pg.math.Vector2
n_snap = 0

# Manually places the window
os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (50, 50)

STATE_SPACE = 6
ACTION_SPACE = 4

REWARD_EXIT = 1.0
REWARD_FREE = 0.5

PENALTY_WANDER = -1
PENALTY_OCCUPIED = -0.75
PENALTY_OUT = -0.8
PENALTY_VISITED = -0.25

THRESHOLD_REWARD = -2 * const.CONFIGURATION.size

CELL_COLORS = {
    0: const.OCCUPIED_CELL_COLOR,
    1: const.FREE_CELL_COLOR,
    2: const.VISITED_CELL_COLOR,
    3: const.PLAYER_COLOR,
    4: const.TARGET_COLOR,
}


class Game:
    def __init__(self, human=False, grid=False, infos=True) -> None:
        pg.init()
        self.human = human
        self.grid = grid
        self.infos = infos
        self.screen = pg.display.set_mode([const.TOTAL_WIDTH, const.TOTAL_HEIGHT])
        self.clock = pg.time.Clock()
        self.running = True

        pg.display.set_caption(const.TITLE)

        self.maze = const.CONFIGURATION.copy()
        self.position = (0, 0)

        self.state_space = STATE_SPACE
        self.action_space = ACTION_SPACE

        self.n_episode = 0
        self.reward_episode = 0
        self.rewards = [0]

    ####### Methods #######

    def reset(self) -> np.array:
        """Resets the game and return its corresponding state."""
        self.score = 0
        self.reward_episode = 0
        self.position = (0, 0)
        self.maze = const.CONFIGURATION.copy()

        return self.get_state()

    def move(self, action) -> None:
        """
        Moves player according to the action chosen by the model.

        args:
            action (int, required): action chosen by the human/agent to move the player
        """
        self.exit = self.visited = self.out = self.occupied = False
        i, j = self.position
        old_position = i, j

        if action == 0:
            j += 1
        elif action == 1:
            j -= 1
        elif action == 2:
            i -= 1
        elif action == 3:
            i += 1

        # move out of bounds
        if i == self.maze.shape[0] or j == self.maze.shape[1] or j < 0 or i < 0:
            self.out = True
        # move to free/visited cell
        elif self.maze[i, j] in (1, 2):
            if self.maze[i, j] == 2:
                self.visited = True

            self.maze[old_position] = 2
            self.maze[i, j] = 3
            self.position = i, j
        # trying to move to an occupied cell
        elif self.maze[i, j] == 0:
            self.occupied = True
        # move to exit (win)
        elif self.maze[i, j] == 4:
            self.position = i, j
            self.exit = True

    def step(self, action):
        self.events()
        self.move(action)

        reward, done = self.get_reward()
        self.reward_episode += reward

        return self.get_state(), reward, done, False

    def get_state(self) -> np.ndarray:
        """Returns the current state of the game."""
        state = [
            self.position[0],
            self.position[1],
        ]
        state.extend(self.get_values_neighbours())

        return np.array(state, dtype=np.float32)

    def get_reward(self) -> tuple:

        # stops episode if the player does nothing but wonder around
        if self.reward_episode < THRESHOLD_REWARD:
            return PENALTY_WANDER, True
        # player moves out of bounds
        elif self.out:
            return PENALTY_OUT, False
        # player moves to a visited cell
        elif self.visited:
            return PENALTY_VISITED, False
        elif self.occupied:
            return PENALTY_OCCUPIED, False
        # player finds the exit
        elif self.exit:
            return REWARD_EXIT, True

        # player moves to a free cell
        return REWARD_FREE, False

    def get_values_neighbours(self):
        offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        values_neighbours = []

        for offset in offsets:
            i, j = self.position[0], self.position[1]
            i, j = i + offset[0], j + offset[1]
            if 0 <= i < self.maze.shape[0] and 0 <= j < self.maze.shape[1]:
                values_neighbours.append(self.maze[i, j])
            else:
                values_neighbours.append(-1) # out of bounds

        return values_neighbours

    def events(self):
        for event in pg.event.get():
            if (
                event.type == pg.QUIT
                or event.type == pg.KEYDOWN
                and event.key == pg.K_q
            ):
                self.running = False

    def render(self, agent):
        """TODO"""

        self.screen.fill(const.BACKGROUND_COLOR)

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                value = self.maze[i, j]
                color = CELL_COLORS[value]
                x, y = const.INFOS_WIDTH + j * const.BLOCK_SIZE, i * const.BLOCK_SIZE
                w, h = const.BLOCK_SIZE, const.BLOCK_SIZE

                pg.draw.rect(self.screen, color, (x, y, w, h))

        if self.grid:
            self.draw_grid()
        if self.infos:
            self.draw_infos(agent)

        self.draw_progress_bar()

        pg.display.flip()
        self.clock.tick(const.FPS)

    def draw_entities(self):
        """TODO"""
        self.player.draw(self.screen)
        self.food.draw(self.screen)

        for enemy in self.enemies:
            enemy.draw(self.screen)

    def draw_grid(self):
        """TODO"""
        for i in range(1, const.PLAY_WIDTH // const.BLOCK_SIZE):
            # vertical lines
            p_v1 = const.INFOS_WIDTH + const.BLOCK_SIZE * i, 0
            p_v2 = const.INFOS_WIDTH + const.BLOCK_SIZE * i, const.PLAY_HEIGHT

            # horizontal lines
            p_h1 = const.INFOS_WIDTH, const.BLOCK_SIZE * i
            p_h2 = const.TOTAL_WIDTH, const.BLOCK_SIZE * i

            pg.draw.line(self.screen, const.GRID_COLOR, p_v1, p_v2)
            pg.draw.line(self.screen, const.GRID_COLOR, p_h1, p_h2)

    def draw_infos(self, agent):
        """Draws game informations"""

        perc_exploration = (
            agent.n_exploration / (agent.n_exploration + agent.n_exploitation) * 100
        )
        perc_exploitation = (
            agent.n_exploitation / (agent.n_exploration + agent.n_exploitation) * 100
        )
        perc_threshold = int((self.reward_episode / THRESHOLD_REWARD) * 100)

        infos = [
            f"Episode: {self.n_episode}",
            f"Episode reward: {round(self.reward_episode, 1)}",
            f"Mean reward: {round(np.mean(self.rewards), 1)}",
            f"Initial Epsilon: {agent.max_epsilon}",
            f"Epsilon: {round(agent.epsilon, 4)}",
            f"Epsilon decay: {agent.epsilon_decay}",
            f"Exploration: {round(perc_exploration, 3)}%",
            f"Exploitation: {round(perc_exploitation, 3)}%",
            f"Last decision: {agent.last_decision}",
            f"Threshold: {perc_threshold}%",
            f"Time: {int(pg.time.get_ticks() / 1e3)}s",
            f"FPS: {int(self.clock.get_fps())}",
        ]

        # Drawing infos
        for i, info in enumerate(infos):
            message(
                self.screen,
                info,
                const.INFOS_SIZE,
                const.INFOS_COLOR,
                (5, 5 + i * const.Y_OFFSET_INFOS),
            )

    def draw_progress_bar(self):
        x_bg, y_bg = const.INFOS_WIDTH - const.PROGRESS_BAR_WIDTH, 0
        w_bg, h_bg = const.PROGRESS_BAR_WIDTH, const.TOTAL_HEIGHT
        x_fg, y_fg = x_bg, y_bg
        w_fg, h_fg = w_bg, (self.reward_episode / THRESHOLD_REWARD) * h_bg

        pg.draw.rect(
            self.screen, const.PROGRESS_BAR_BACKGROUND, (x_bg, y_bg, w_bg, h_bg)
        )
        pg.draw.rect(
            self.screen, const.PROGRESS_BAR_FOREGROUND, (x_fg, y_fg, w_fg, h_fg)
        )
        pg.draw.line(self.screen, pg.Color("Black"), (x_bg, y_bg), (x_bg, h_bg))
        pg.draw.line(
            self.screen,
            pg.Color("Black"),
            (x_bg + const.PROGRESS_BAR_WIDTH, y_bg),
            (x_bg + const.PROGRESS_BAR_WIDTH, h_bg),
        )


def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, numpy and random.

    Args:
        seed: random seed
    """

    try:
        import torch
    except ImportError:
        print("Module PyTorch cannot be imported")
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)


def main():
    pass


if __name__ == "__main__":
    main()
