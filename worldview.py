import pygame
import numpy as np

WIDTH = 256
HEIGHT = 256

FILL_COLOR = (0, 0, 0)


class WorldViewer:
    """
    Frame to visualize the simplified world.
    """
    def __init__(self, width=WIDTH, height=HEIGHT):
        pygame.init()
        self.size = (width, height)
        self.screen = pygame.display.set_mode((width, height))
        self.fill_color = FILL_COLOR
        self.colormap = [
            (0, 0, 0),              # Empty blocks
            (255, 255, 255),        # Full blocks
            (0, 255, 0),            # Mario
            (255, 0, 0)             # Enemy
        ]

    def render(self, world):
        """
        Updates the frame with new world data.
        :param world: Simplified world matrix (15 x 16 x 4)
        """
        self.screen.fill(self.fill_color)

        nb_tiles_x = len(world)
        nb_tiles_y = len(world[0])

        tile_x_size = self.size[0] / nb_tiles_x
        tile_y_size = self.size[1] / nb_tiles_y

        for y in range(nb_tiles_y):
            for x in range(nb_tiles_x):
                rect = pygame.Rect((y * tile_x_size, x * tile_y_size), (tile_x_size, tile_y_size))

                color_id = 0
                for d in range(world.shape[2]):
                    if world[x][y][d] == 1:
                        color_id = d

                pygame.draw.rect(self.screen, self.colormap[color_id], rect)

        pygame.display.flip()