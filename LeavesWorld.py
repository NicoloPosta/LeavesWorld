import numpy as np
import random
import GUI
from Randomizer import Randomizer

class LeavesWorld(object):
    def __init__(self, rs: np.random.RandomState, m = 7, n = 7, p_leaves=40):
        self.rs = rs
        self.grid = np.zeros((m,n))
        self.grid_for_reset = np.zeros((m,n))
        self.iterator = 0
        self.leaves_sucked = 0
        self.m = m
        self.n = n
        self.p_leaves = p_leaves
        self.goal = self.n * self.m - 1
        self.leaves = int((float(n*m)/100) * float(p_leaves))
        self.randomGrid()
        self.stateSpace = [i for i in range(19571)] # spazio degli stati, grandezza della matrice
        self.stateGoals = [ self.goal ]
        self.actionSpace = {'U': -self.n, 'D': self.n, 'L': -1, 'R': 1, 'S': 0}
        self.possibleActions = ['U', 'D', 'L', 'R', 'S']
        self.agentPosition = 0
        self.gui = GUI.GUI(self.grid)
    
    @property
    def percentageLeavesSucked(self) -> float:
        if self.leaves == 0:
            return 0
        return float(self.leaves_sucked) / float(self.leaves) * 100

    # contorllo del goal state
    def isTerminalState(self, state: int) -> bool:  
        return state in self.stateGoals

    # trova la posizione dell'agente
    def getAgentRowAndColumn(self) -> tuple[int, int]: 
        x = int(self.agentPosition / self.n)
        y = self.agentPosition % self.n
        return x, y

    # restituisce il vicinato di 9-Moore basandosi sulle coordinate dell'agente
    def getNeighbours(self,  nextPosition = "default") -> np.ndarray:
        if nextPosition != "default":
            x = int(nextPosition / self.n)
            y = nextPosition % self.n
        else:
            x, y = self.getAgentRowAndColumn()

        neighbours = []
        #               UL         U       UR       L       NM       R      DL        D      DR
        for move in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
            new_x, new_y = x + move[0], y + move[1]
            neighbour = 0

            if new_x < 0:
                neighbour = 2
            elif new_y < 0:
                neighbour = 2
            elif new_x > self.m - 1:
                neighbour = 2
            elif new_y > self.n - 1:
                neighbour = 2
            else:
                neighbour = self.grid[new_x][new_y]

            neighbours.append(neighbour)
        return np.array(neighbours, dtype=np.int8)
    
    # controllo se il movimento è off grid
    def offGridMove(self, newState: int, oldState: int) -> bool: 
        # se ci muoviamo su una cella non nella griglia
        if newState not in range(self.n*self.m):
            return True
        # controlliamo se x_new e y_new sono entrambi variati allora è comunque off grid
        x_old = int(oldState / self.n)
        y_old = oldState % self.n
        x_new = int(newState / self.n)
        y_new = newState % self.n
        if x_old != x_new and y_old != y_new:
            return True

    # converte il vicinato in un intero in base 3
    def neighboursToInt(self, state: int) -> int:
        tmp = np.array2string(state, separator="")
        return int(tmp[1:-1], 3)

    # calcola lo stato successivo
    def calculateNextState(self, action: str, nextPosition: int) -> int: 
        neighbours = self.getNeighbours()
        if action == "S":
            neighbours[4] = 0
            return self.neighboursToInt(neighbours)
        else:
            return self.neighboursToInt(self.getNeighbours(nextPosition))

    # esegue uno step in base all'azione passatogli
    def step(self, action: str) -> tuple[int, int, bool, dict]:
        # incremento il contatore per muovere le foglie
        if self.iterator == 2:
            self.iterator = 0
            self.moveLeaves()
        self.iterator += 1 
        current_neighbours = self.getNeighbours()
        resultingPosition = self.agentPosition + self.actionSpace[action]
        x, y = self.getAgentRowAndColumn()
        if action == 'S' and self.grid[x][y] == 1:
            self.grid[x][y] = 0
            reward = 5
            self.leaves_sucked += 1
        elif action == 'S' and self.grid[x][y] != 1:
            reward = -5
        elif not self.isTerminalState(resultingPosition):
            reward = -1
        else:
            reward = 5*self.leaves_sucked
        # controllo se il movimento è off grid
        if not self.offGridMove(resultingPosition, self.agentPosition):
            resultingState = self.calculateNextState(action, resultingPosition)
            self.agentPosition = resultingPosition
            return resultingState, reward, self.isTerminalState(resultingPosition), None
        else:
            reward = -10
            return self.neighboursToInt(current_neighbours), reward, self.isTerminalState(resultingPosition), None

    # genera una composizione random delle foglie nella griglia
    def randomGrid(self) -> None:
        for i in range(self.leaves):
            a = self.rs.randint(0, self.m - 1)
            b = self.rs.randint(0, self.n - 1)
            while self.grid[a][b] == 1: # controllo che la cella sia vuota
                a = self.rs.randint(0, self.m - 1)
                b = self.rs.randint(0, self.n - 1)
            self.grid[a][b] = 1
        self.grid_for_reset = self.grid

    # muove le foglie
    def moveLeaves(self) -> None:   
        newgrid = np.zeros((self.m,self.n))
        for i in range(self.m):
            for j in range(self.n):
                # cerco le foglie nella griglia
                if self.grid[i][j] == 1:
                    # se nella casella c'è una foglia, genero una nuova posizione
                    a = self.rs.randint(0, self.m - 1)
                    b = self.rs.randint(0, self.n - 1)
                    # se nella nuova posizione vi è presente già una foglia, allora riposiziono la foglia dove era prima, se dove era prima c'è già una foglia
                    # allora genero randomicamente una poszione fino a poterla inserire se no si va a perdere nel vuoto una foglia
                    # non posso controllare usando la vecchia matrice dato che se no scorrendo la matrice 
                    # per ogni foglia potrei spostarne una foglia più volte se capita di posizionarsi più avanti nel ciclo
                    if newgrid[a][b] == 1:
                        if newgrid[i][j] == 1:
                            while newgrid[a][b] == 1:
                                a = self.rs.randint(0, self.m - 1)
                                b = self.rs.randint(0, self.n - 1)
                            newgrid[a][b] = 1
                        else:
                            newgrid[i][j] = 1
                    else:
                        newgrid[a][b] = 1
        self.grid = newgrid

    # resetta l'ambiente
    def reset(self) -> int:
        self.agentPosition = 0
        self.leaves_sucked = 0
        self.grid = self.grid_for_reset
        return self.neighboursToInt(self.getNeighbours())

    # stampa la matrice a schermo
    def printMatrix(self) -> None:
        elem = "-------"
        print(elem*(self.n-1)+"-"*self.n)
        x, y = self.getAgentRowAndColumn()
        for row in range(self.m):
            for col in range(self.n):
                if row == x and col == y:
                    print ('X', end='\t')
                elif  self.grid[row][col]==1:
                    print ('F', end='\t')
                else:
                    print('-', end='\t')
            print('\n')
        print(elem*(self.n-1)+"-"*self.n)

    # renderizza l'ambiente sulla GUI
    def render(self, print_matrix=False) -> None:   
        if print_matrix:
            self.printMatrix()
        x, y = self.getAgentRowAndColumn()
        self.gui.draw((x,y), self.grid)
        tmp = self.grid.ravel()
        total_sum = tmp.tolist().count(1)
        print(f"Leaves in the grid: {total_sum}")
        print(f"Leaves sucked: {self.leaves_sucked}")

    # genera una azione random
    def actionSpaceSample(self) -> str:
        return self.rs.choice(self.possibleActions)

#####