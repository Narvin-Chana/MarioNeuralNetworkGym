import numpy as np
from collections import namedtuple
from ramutils import *

NB_ENEMY = 5

Point = namedtuple('Point', ['x', 'y'])
# Set bins. Blocks are 16x16 so we create bins offset by 16
ybins = list(range(16, 240, 16))
xbins = list(range(16, 256, 16))


def get_enemy_positions(env):
    """
    Returns all the active enemies positions.

    This function comes from Chrispresso who found a way to easily get all these positions.
    (https://chrispresso.io/AI_Learns_To_Play_SMB_Using_GA_And_NN#results)
    :param env:
    :return:
    """
    enemy_locations = []

    for enemy_num in range(NB_ENEMY):
        enemy = read_ram(env, 0xF + enemy_num)[0]
        # RAM locations 0x0F through 0x13 are 0 if no enemy
        # drawn or 1 if drawn to screen
        if enemy:
            # Grab the enemy location
            x_pos_level = read_ram(env, 0x6E + enemy_num)[0]
            x_pos_screen = read_ram(env, 0x87 + enemy_num)[0]
            # The width in pixels is 256. 0x100 == 256.
            # Multiplying by x_pos_level gets you the
            # screen that is actually displayed, and then
            # add the offset from x_pos_screen
            enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen
            enemy_loc_y = read_ram(env, 0xCF + enemy_num)[0]
            # Get row/col
            row = enemy_loc_y // 16
            col = enemy_loc_x // 16
            # Add location.
            # col moves in x-direction
            # row moves in y-direction
            location = Point(col, row)
            enemy_locations.append(location)

    return enemy_locations


Mario_X_Position = 0x0086
Mario_Screen_Value = 0x006D
Screen_Value = 0x071A
Screen_Edge_X = 0x071C
Mario_Y_Position = 0x00CE


def get_mario_position(env):
    """
    Returns Mario's position inside the level (on screen).
    :param env: The ram environment (needs to contain a 'ram' variable or wrap an environment that does)
    :return: X and Y value of Mario's position on the screen (X: [0,256], Y: [0,240])
    """
    x_pos = env.ram[Mario_X_Position] + env.ram[Mario_Screen_Value] * 256 - (env.ram[Screen_Edge_X] + (env.ram[Screen_Value] * 256))
    y_pos = env.ram[Mario_Y_Position]

    print(x_pos, y_pos)
    return x_pos, y_pos


Empty = 0x00
Full = 0x01
Blocks_Address = 0x0500


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
                blockList[row, col] = Empty
            else:
                x, y = col * 16, row * 16
                page = (x // 256) % 2
                sub_x = (x % 256) // 16
                sub_y = (y - 32) // 16

                addr = Blocks_Address + page * 208 + sub_y * 16 + sub_x

                blockList[row, col] = Empty if env.ram[addr] == Empty else Full

    return blockList
