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

STATE_SPACE = 2
ACTION_SPACE = 4

MAX_FRAME = 500

REWARD_EXIT = 10

PENALTY_WANDER = -1
PENALTY_OCCUPIED = -2
PENALTY_OUT = -5
PENALTY_VISITED = -1

MOVES = {0: "right", 1: "left"}
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
        self.screen = pg.display.set_mode([const.PLAY_WIDTH, const.PLAY_HEIGHT])
        self.clock = pg.time.Clock()
        self.running = True

        pg.display.set_caption(const.TITLE)

        self.maze = const.CONFIGURATION.copy()
        self.position = (0, 0)

        self.state_space = STATE_SPACE
        self.action_space = ACTION_SPACE
        self.n_games = 0
        self.n_frames_threshold = 0

    ####### Methods #######

    def reset(self) -> np.array:
        """Resets the game and return its corresponding state."""
        self.score = 0
        self.n_frames_threshold = 0
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
        if (
            j * const.BLOCK_SIZE == const.PLAY_WIDTH
            or j * const.BLOCK_SIZE < 0
            or i * const.BLOCK_SIZE < 0
            or i * const.BLOCK_SIZE == const.PLAY_HEIGHT
        ):
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
            self.exit = True
        
        # if self.human:
        #     keys = pg.key.get_pressed()
        #     i, j = self.position 
        #     old_position = i, j
            
        #     if keys[pg.K_RIGHT]:
        #         j += 1
        #     elif keys[pg.K_LEFT]:
        #         j -= 1
        #     elif keys[pg.K_UP]:
        #         i -= 1
        #     elif keys[pg.K_DOWN]:
        #         i += 1

        #     # move out of bounds
        #     if (
        #         j * const.BLOCK_SIZE == const.PLAY_WIDTH
        #         or j * const.BLOCK_SIZE < 0
        #         or i * const.BLOCK_SIZE < 0
        #         or i * const.BLOCK_SIZE == const.PLAY_HEIGHT
        #     ):
        #         self.out = True
        #     # move to free/visited cell
        #     elif self.maze[i, j] in (0, 2):
        #         if self.maze[i, j] == 2:
        #             self.visited = True
                    
        #         self.maze[old_position] = 2
        #         self.maze[i, j] = 3
                
        #         self.position = i, j
        #     # move to exit (win)
        #     elif self.maze[i, j] == 4:
        #         self.exit = True

    def step(self, action):
        self.n_frames_threshold += 1

        self.events()
        self.move(action)

        reward, done = self.get_reward()
        state = self.get_state()

        return state, reward, done

    def get_state(self) -> np.array:
        return np.array([self.position[0], self.position[1]], dtype=np.float32)

    def get_reward(self) -> tuple:

        # stops episode if the player does nothing but wonder around
        if self.n_frames_threshold > MAX_FRAME:
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
        return 1, False

    def events(self):
        for event in pg.event.get():
            if (
                event.type == pg.QUIT
                or event.type == pg.KEYDOWN
                and event.key == pg.K_q
            ):
                self.running = False

    def render(self):
        """TODO"""

        self.screen.fill(const.BACKGROUND_COLOR)

        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                value = self.maze[i, j]
                color = CELL_COLORS[value]
                x, y = j * const.BLOCK_SIZE, i * const.BLOCK_SIZE
                w, h = const.BLOCK_SIZE, const.BLOCK_SIZE

                pg.draw.rect(self.screen, color, (x, y, w, h))

        self.draw_grid()

        if self.infos:
            self.draw_infos()

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
            p_v1 = const.BLOCK_SIZE * i, 0
            p_v2 = const.BLOCK_SIZE * i, const.PLAY_HEIGHT

            # horizontal lines
            p_h1 = 0, const.BLOCK_SIZE * i
            p_h2 = const.PLAY_WIDTH, const.BLOCK_SIZE * i

            pg.draw.line(self.screen, const.GRID_COLOR, p_v1, p_v2)
            pg.draw.line(self.screen, const.GRID_COLOR, p_h1, p_h2)

    def draw_infos(self):
        """Draws game informations"""

        if self.score > self.highest_score:
            self.highest_score = self.score

        perc_exploration = (
            self.agent.n_exploration
            / (self.agent.n_exploration + self.agent.n_exploitation)
            * 100
        )
        perc_exploitation = (
            self.agent.n_exploitation
            / (self.agent.n_exploration + self.agent.n_exploitation)
            * 100
        )
        perc_threshold = int((self.n_frames_threshold / MAX_FRAME) * 100)

        infos = [
            f"Game: {self.n_games}",
            f"Reward game: {round(self.reward_episode, 1)}",
            f"Mean reward: {round(self.mean_rewards[-1], 1)}",
            f"Score: {self.score}",
            f"Highest score: {self.highest_score}",
            f"Mean score: {round(self.mean_scores[-1], 1)}",
            f"Initial Epsilon: {self.agent.max_epsilon}",
            f"Epsilon: {round(self.agent.epsilon, 4)}",
            f"Exploration: {round(perc_exploration, 3)}%",
            f"Exploitation: {round(perc_exploitation, 3)}%",
            f"Last decision: {self.agent.last_decision}",
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

        # sep line
        pg.draw.line(
            self.screen,
            const.SEP_LINE_COLOR,
            (const.INFO_WIDTH, 0),
            (const.INFO_WIDTH, const.INFO_HEIGHT),
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