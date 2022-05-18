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

    This function comes from Chrispresso that found a way to easily get all these positions.
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
            x_pos_level  = read_ram(env, 0x6E + enemy_num)[0]
            x_pos_screen = read_ram(env, 0x87 + enemy_num)[0]
            # The width in pixels is 256. 0x100 == 256.
            # Multiplying by x_pos_level gets you the
            # screen that is actually displayed, and then
            # add the offset from x_pos_screen
            enemy_loc_x = (x_pos_level * 0x100) + x_pos_screen
            enemy_loc_y = read_ram(env, 0xCF + enemy_num)
            # Get row/col
            row = np.digitize(enemy_loc_y, ybins)
            col = np.digitize(enemy_loc_x, xbins)
            # Add location.
            # col moves in x-direction
            # row moves in y-direction
            location = Point(col, row)
            enemy_locations.append(location)

    return enemy_locations


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
