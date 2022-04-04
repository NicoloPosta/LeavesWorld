import pygame
import numpy as np

class GUI(object):
    leaf_color = (255, 153, 51)
    col_grid = (0, 0, 0)
    col_background = (255, 255, 255)
    goal_color = (0, 0, 255)
    agent_color = (75, 0, 150)

    def __init__(self, matrix: np.ndarray, cell_size=65):
        self.matrix = matrix
        self.x, self.y = self.matrix.shape

        # definizione della posizione goal per mostrarlo a schermo
        self.goal = (self.x - 1, self.y - 1)
        self.cell_size = cell_size
        # inizializza la finestra di pygame
        self.surface = None

    # Inizializza la finestra di pygame.
    def __init_pygame(self) -> None:
        pygame.init()
        self.surface = pygame.display.set_mode((self.y * self.cell_size, self.x * self.cell_size))
        pygame.display.set_caption("QML Learning")

    # stampa la matrice a schermo
    def draw(self, agent_pos: tuple[int, int], grid) -> bool:
        self.__init_pygame()
        self.matrix = grid

        agent_x, agent_y = agent_pos

        for i in range(self.x):
            for j in range(self.y):
                if self.matrix[i][j] == 1:
                    col = self.leaf_color
                else:
                    col = self.col_background
                if i == agent_x and j == agent_y:
                    col = self.agent_color
                elif i == self.goal[0] and j == self.goal[1]:
                    col = self.goal_color
                # stampa la cella corrente
                pygame.draw.rect(self.surface, col, (j * self.cell_size, i * self.cell_size, self.cell_size - 1, self.cell_size - 1))

        pygame.display.update()

        # controlla se l'utente chiude la finestra
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        return True
