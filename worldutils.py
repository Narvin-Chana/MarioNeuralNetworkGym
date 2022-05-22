import math

import numpy as np
from collections import namedtuple
from ramutils import *

# RAM Addresses
ACTIVE_ENEMIES = 0xF
ENEMIES_LEVEL_POS = 0x6E
ENEMIES_X_SCREEN = 0x87
ENEMIES_Y_SCREEN = 0xCF

MARIO_X = 0x0086
MARIO_X_SCREEN_OFFSET = 0x03AD
MARIO_Y = 0x00CE
MARIO_CURRENT_SCREEN = 0x006D
CURRENT_SCREEN = 0x071A
SCREEN_EDGE_X = 0x071C

CURRENT_BLOCKS = 0x0500

# Constants
NB_ENEMY = 5
BLOCK_TYPE_EMPTY = 0x00
BLOCK_TYPE_FULL = 0x01


def get_enemy_positions(env):
    """
    Returns all the active enemies positions.

    :param env: The ram environment (needs to contain a 'ram' variable or wrap an environment that does)
    :return:
    """
    enemy_locations = []

    left_edge_pos = read_ram(env, CURRENT_SCREEN)[0] * 0x100 + read_ram(env, SCREEN_EDGE_X)[0]

    for enemy_num in range(NB_ENEMY):
        enemy = read_ram(env, ACTIVE_ENEMIES + enemy_num)[0]
        # RAM locations 0x0F through 0x13 are 0 if no enemy
        # drawn or 1 if drawn to screen
        if enemy:
            # Grab the enemy location
            x_pos_level = read_ram(env, ENEMIES_LEVEL_POS + enemy_num)[0]
            x_pos_screen = read_ram(env, ENEMIES_X_SCREEN + enemy_num)[0]

            # The width of one Screen in pixels is 256. 0x100 == 256.

            enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen
            y_pos_screen = read_ram(env, ENEMIES_Y_SCREEN + enemy_num)[0]

            location = (enemy_loc_x - left_edge_pos, y_pos_screen)
            enemy_locations.append(location)

    return enemy_locations


def get_mario_position(env):
    """
    Returns Mario's position inside the level (on screen).
    :param env: The ram environment (needs to contain a 'ram' variable or wrap an environment that does)
    :return: X and Y value of Mario's position on the screen (X: [0,256], Y: [0,240])
    """
    x_pos = env.ram[MARIO_X] + env.ram[MARIO_CURRENT_SCREEN] * 256 - (
            env.ram[SCREEN_EDGE_X] + (env.ram[CURRENT_SCREEN] * 256))
    y_pos = env.ram[MARIO_Y]

    return x_pos, y_pos


def read_blocks(env):
    """
    Reads the blocks state on the current screen.
    :param env: The ram environment (needs to contain a 'ram' variable or wrap an environment that does)
    :return: A two-sided array
    """
    blockList = np.zeros((15, 16), dtype=np.int32)

    for row in range(0, 15):
        for col in range(0, 16):
            if row < 2:
                blockList[row, col] = BLOCK_TYPE_EMPTY
            else:
                x_start = env.ram[MARIO_X] + env.ram[MARIO_CURRENT_SCREEN] * 256 - env.ram[MARIO_X_SCREEN_OFFSET]
                x, y = col * 16 + x_start, row * 16
                page = (x // 256) % 2
                sub_x = (x % 256) // 16
                sub_y = (y - 32) // 16

                addr = CURRENT_BLOCKS + page * 208 + sub_y * 16 + sub_x

                blockList[row, col] = BLOCK_TYPE_EMPTY if env.ram[addr] == BLOCK_TYPE_EMPTY else BLOCK_TYPE_FULL

    return blockList


EMPTY_MASK_VALUE = 0
FULL_MASK_VALUE = 1
MARIO_MASK_VALUE = 2
ENEMY_MASK_VALUE = 3


def get_simplified_world(env):
    world = np.zeros((15, 16, 4), dtype=np.byte)
    blocks = read_blocks(env)

    for row in range(0, 15):
        for col in range(0, 16):
            if blocks[row, col] == BLOCK_TYPE_EMPTY:
                world[row, col, EMPTY_MASK_VALUE] = 1
            elif blocks[row, col] == BLOCK_TYPE_FULL:
                world[row, col, FULL_MASK_VALUE] = 1

    mario_x, mario_y = get_mario_position(env)
    mario_row = round(mario_y / 16 + 0.5)
    mario_col = round(mario_x / 16 + 0.5)

    if 0 <= mario_row <= 14 and 0 <= mario_col <= 15:
        world[mario_row, mario_col, MARIO_MASK_VALUE] = 1

    enemy = get_enemy_positions(env)

    for e in enemy:
        e_x, e_y = e[0], e[1]
        if e_x < 0 or e_y < 0 or e_x > 255:
            break
        e_row = min(round(e_y / 16 + 0.5), 14)
        e_col = min(round(e_x / 16 + 0.5), 15)
        world[e_row, e_col, ENEMY_MASK_VALUE] = 1

    # To visualize a specific layer:
    # for k in range(0, 4):
    #     a = np.zeros((15, 16), dtype=np.int32)
    #     for i in range(0, 15):
    #         for j in range(0, 16):
    #             a[i, j] = world[i, j, k]
    #     if k == LAYER_YOU_WANT_TO_SEE:
    #         print(f"value of dimension {k}: ")
    #         print(a)

    return world
