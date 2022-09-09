import os
import pygame as pg
import numpy as np
from gen_maze import random_maze_generator

vec = pg.math.Vector2

N = 11
M = 11
P0 = (0, 0)
# P1 = (N-1, M-1)
P1 = (4, 0)
maze = random_maze_generator(N, M, P0, P1)

# Configuration
CONFIGURATION = np.array(maze)

# Main window
TITLE = "Maz3-DQN"
BLOCK_SIZE = 40
INFOS_WIDTH = 275
INFOS_HEIGHT = CONFIGURATION.shape[0] * BLOCK_SIZE
PLAY_WIDTH = CONFIGURATION.shape[1] * BLOCK_SIZE
PLAY_HEIGHT = INFOS_HEIGHT
TOTAL_WIDTH = INFOS_WIDTH + PLAY_WIDTH
TOTAL_HEIGHT = INFO_HEIGHT = PLAY_HEIGHT
FPS = 40


# Directories
FILE_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(FILE_DIR, "../imgs")
SNAP_FOLDER = os.path.join(FILE_DIR, "../snapshots")

# Colors
GRID_COLOR = (40, 40, 40)
BACKGROUND_COLOR = (30, 30, 30)
FREE_CELL_COLOR = (220, 220, 220)
OCCUPIED_CELL_COLOR = (70, 70, 70)
VISITED_CELL_COLOR = pg.Color("Yellow")
PLAYER_COLOR = pg.Color("Red")
TARGET_COLOR = pg.Color("Green")
SEP_LINE_COLOR = (60, 60, 60)
INFOS_COLOR = (255, 255, 255)
PROGRESS_BAR_BACKGROUND = (20, 20, 20)
PROGRESS_BAR_FOREGROUND = (186, 3, 252)

# Miscs
INFOS_SIZE = 20
Y_OFFSET_INFOS = 25
PROGRESS_BAR_WIDTH = 25
