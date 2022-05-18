import pygame

WIDTH = 256
HEIGHT = 256

FILL_COLOR = (0, 0, 0)


class WorldViewer:

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
        self.screen.fill(self.fill_color)

        nb_tiles_x = len(world[0])
        nb_tiles_y = len(world)

        tile_x_size = self.size[0] / nb_tiles_x
        tile_y_size = self.size[1] / nb_tiles_y

        for y in range(nb_tiles_y):
            for x in range(nb_tiles_x):
                rect = pygame.Rect((x * tile_x_size, y * tile_y_size), (tile_x_size, tile_y_size))
                pygame.draw.rect(self.screen, self.colormap[world[x][y]], rect)

        pygame.display.flip()