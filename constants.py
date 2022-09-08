import os
import pygame as pg
import numpy as np
from gen_maze import random_maze_generator

vec = pg.math.Vector2

N = 15
M = 15
P0 = (0, 0)
P1 = (N-1, M-1)
maze = random_maze_generator(N, M, P0, P1)

# Configuration
CONFIGURATION = np.array(maze)
print(CONFIGURATION)

# Main window
TITLE = "maz3-solver"
BLOCK_SIZE = 40
PLAY_HEIGHT = CONFIGURATION.shape[0] * BLOCK_SIZE
PLAY_WIDTH = CONFIGURATION.shape[1] * BLOCK_SIZE
FPS = 30

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

# Target

# Miscs
INFOS_SIZE = 20
Y_OFFSET_INFOS = 25
