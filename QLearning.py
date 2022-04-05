from os import path
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import LeavesWorld
from os import system, name
from Randomizer import Randomizer

# pulisce la console
def clear() -> None:
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

class QLearning(object):
    def __init__(self, env: LeavesWorld.LeavesWorld, rs: np.random.RandomState):
        self.rs = rs
        self.env = env
        self.Q = self.newMatrix()

    # calcola l'azione migliore per uno stato basandosi sulla Q matrix
    def maxAction(self,Q: dict, state: int, actions: list, in_execution=False) -> int:
        values = np.array([Q[state,a] for a in actions])
        action = np.argmax(values)

        if in_execution:
            tmp = [i for i, x in enumerate(values) if x == values[action]]
            if len(tmp) > 1:
                action = self.rs.choice(tmp)

        return actions[action]

    #salva la Q matrix in un file
    def saveQ(self,Q: dict,fileName: str) -> None:
        numStates = len(self.env.stateSpace)
        Qmatrix = np.zeros((numStates, 5))
        x=0
        for state in self.env.stateSpace:
                y=0
                for action in self.env.possibleActions:
                    Qmatrix[x][y]=Q[state, action]
                    y+=1
                x +=1
        np.savetxt(fileName, Qmatrix, delimiter=" ")

    # carica la Q matrix da un file
    def loadQ(self,fileName: str) -> dict:
        Q = {}
        for state in self.env.stateSpace:
            for action in self.env.possibleActions:
                Q[state, action] = 0
        if path.exists(fileName):
            with open(fileName) as file_name:
                Qmatrix = np.loadtxt(file_name, delimiter=" ")
            x=0
            for state in self.env.stateSpace:
                    y=0
                    for action in self.env.possibleActions:
                            Q[state,action]=Qmatrix[x][y]
                            y+=1
                    x+=1   
        return Q

    # genera un dataset di esempio casualmente
    def createDataSet(self, min_lenght = 2, max_lenght = 10, dimension = 100, percentage = 50, verbose=False) -> list:
        dataset = []
        for _ in range(dimension):
            m = self.rs.randint(min_lenght, max_lenght+1)
            n = self.rs.randint(min_lenght, max_lenght+1)
            elem = [m,n,percentage]
            dataset.append(elem)
        if verbose:
            print(dataset)
        return dataset

    # training della Q matrix usando un dataset di esempio
    def trainingWithDataset(self, dataset: list, max_steps: int) -> None:
        self.blankQMatrix()
        self.Q = self.loadQ(fileName="Qmatrix")
        num_training = 1
        Q = {}
        for elem in dataset:
            self.env = LeavesWorld.LeavesWorld(elem[0],elem[1],elem[2])
            print(f"Training number: {num_training}")
            Q = self.training(epochs = 10000, steps = max_steps, dataset=True)
            num_training += 1
        self.saveQ(Q,"Qmatrix")

    # genera una Q matrix vuota
    def blankQMatrix(self) -> None:
        self.saveQ(self.newMatrix(), 'Qmatrix')

    # genera una Q matrix con tutti i valori a 0
    def newMatrix(self) -> dict:
        Q = {}
        for state in self.env.stateSpace:
            for action in self.env.possibleActions:
                Q[state, action] = 0
        return Q

    # esegue l'agente usando la Q matrix caricata su un dataset di esempio
    def executeOnDataset(self, dataset: list, max_steps=1500) -> None:
        Q=self.loadQ("Qmatrix")
        evaluation = []
        iterator = 0
        times_stuck = 0
        totLeaves = 0
        reward_list = []
        step_list = []
        for elem in dataset:
            print(f"Execution on the {iterator+1} element of the dataset")
            iterator += 1
            self.env = LeavesWorld.LeavesWorld(elem[0],elem[1],elem[2])
            totLeaves += self.env.leaves
            totReward=0
            steps = 0
            while steps < max_steps:
                action=self.maxAction(Q, self.env.neighboursToInt(self.env.getNeighbours()), self.env.possibleActions, in_execution=True)
                observationNext, reward, done, info = self.env.step(action)
                steps += 1
                totReward += reward
                if done:
                    step_list.append(steps)
                    evaluation.append(round(self.env.percentageLeavesSucked, 2))
                    reward_list.append(totReward)
                    break
            if steps == max_steps:
                step_list.append(max_steps)
                evaluation.append(0.0)
                reward_list.append(totReward)
                times_stuck += 1
                print("Agent got stuck, skip iteration")
        mean_steps = np.mean(step_list)
        mean_reward = np.mean(reward_list)
        mean_percentage = np.mean(evaluation)
        print(f"Times stuck: {times_stuck}")
        print(f"Mean reward: {round(mean_reward, 2)}")
        print(f"Mean steps: {round(mean_steps, 2)}")
        print(f"Total number of leaves in the dataset: {totLeaves}")
        print(f"Mean percentage of leaves sucked in the experiment: {round(mean_percentage, 2)}%")

    # esegue l'agente usando la Q matrix su un ambiente passato dall'utente
    def execute(self, auto=True, fast=False, max_steps=1500) -> None:
        # se l'esecuzione è in modalità auto l'agente esegue le azioni basandosi 
        # sulla Q matrix caricata senza poter eseguire azioni passatogli dall'utente
        if auto:
            # se l'esecuzione è in modalità fast, non viene mostrato l'ambiente
            if not fast:
                self.env.render()
            Q=self.loadQ("Qmatrix")
            totReward=0
            steps = 0
            #clear()
            print("Executing...")
            while steps < max_steps:
                action=self.maxAction(Q, self.env.neighboursToInt(self.env.getNeighbours()) , self.env.possibleActions, in_execution=True)
                observationNext, reward, done, info = self.env.step(action)
                steps += 1
                totReward += reward
                if not fast:
                    print("Action: "+str(action)+" Reward: "+str(reward)+"\n")
                    self.env.render()
                    print(f"Total reward: {totReward}")
                if done:
                    print(f"Total steps: {steps}")
                    print(f"Total reward: {totReward}")
                    print(f"Total leaves in the environment: {self.env.leaves}")
                    print(f"Leaves sucked: {self.env.leaves_sucked}")
                    print(f"Percentage of leaves sucked: {round(self.env.percentageLeavesSucked, 2)}%")
                    #exit()
                    return
                if not fast:
                    sleep(0.01)
                    clear()
            print("Max step reached")
            #exit()
            return
        else:
            # se l'esecuzione è in modalità non auto, l'agente esegue le azioni passate 
            # dall'utente oppure basandosi sulla Q matrix caricata se non è stata passata una azione
            clear()
            self.env.render(print_matrix=True)
            Q=self.loadQ("Qmatrix")
            totReward=0
            command=input("Total reward: "+str(totReward)+"\nExecuteNext?(y/n/uP/dOWN/lEFT/riGTH/sUCK):")
            while  command != 'n':
                if max_steps >= 0:
                    if command =='y':
                        action=self.maxAction(Q, self.env.neighboursToInt(self.env.getNeighbours()), self.env.possibleActions, in_execution=True)
                    elif command == 'u':
                        action='U'
                    elif command == 'd':
                        action='D'
                    elif command == 'l':
                        action='L'
                    elif command == 'r':
                        action='R'
                    elif command == 's':
                        action='S'
                    observationNext, reward, done, info = self.env.step(action)
                    max_steps -= 1
                    totReward += reward
                    clear()
                    print("Action:"+str(action)+" Reward:"+str(reward)+"\n")
                    self.env.render(print_matrix=True)
                    if not done:
                        command=input("Total reward: "+str(totReward)+"\nExecuteNext?(y/n/uP/dOWN/lEFT/rIGTH/sUCK):")
                    if done:
                        print(f"Percentage of leaves sucked: {round(self.env.percentageLeavesSucked, 2)}%")
                        exit()
                else:
                    print(f"Percentage of leaves sucked: {round(self.env.percentageLeavesSucked, 2)}%")
                    print("Max step reached")
                    exit()

    #Relazione con valori diversi
    # esegue il training dell'agente su un ambiente passato dall'utente
    def training(self, epochs = 80000, steps = 1500, ALPHA= 0.1, GAMMA = 1.0, EPS = 1.0, plot=True, dataset=False) -> None:
        '''
        hyperparametri di default:
        ALPHA = 0.1 è il fattore di apprendimento
        GAMMA = 1.0 è il fattore di discount
        EPS = 1.0   la variabile epsilon è relativo al valore greedy dell'algoritmo
        Epochs = 80000 è il numero massimo di iterazioni
        Steps = 1500 è il numero massimo di azioni ( step ) per epoca
        '''
        # inizializzo la Q matrix o la carico se esiste già
        starting_eps = EPS
        if not dataset:
            Q = self.newMatrix()
        else:
            Q = self.Q.copy()
        
        self.env.reset()
        if not dataset:
            self.env.printMatrix()
        totalRewards = np.zeros(epochs)
        for i in range(epochs):
            if i % int(epochs/10) == 0:
                print('starting game ', i)
            done = False
            epRewards = 0
            numActions = 0 
            observation = self.env.reset()
            while not done and numActions <= steps :
                rand = self.rs.random()
                action = self.maxAction(Q,observation, self.env.possibleActions) if rand < (1-EPS) \
                                                        else self.env.actionSpaceSample()
                observationNext, reward, done, info = self.env.step(action)
                numActions+= 1 
                epRewards += reward
                actionNext = self.maxAction(Q, observationNext, self.env.possibleActions)
                Q[observation,action] = Q[observation,action] + ALPHA*(reward + \
                            GAMMA*Q[observationNext,actionNext] - Q[observation,action])
                observation = observationNext
            if EPS - 2 / epochs > 0:
                EPS -= 2 / epochs
            else:
                EPS = 0
            totalRewards[i] = epRewards

        if not dataset:
            if plot:
                plt.plot(totalRewards)
                plt.savefig(f"imgs/n_{self.env.n}_m_{self.env.m}_percentuale_foglie_{self.env.p_leaves}_epoche_{epochs}_steps_{steps}_alpha_{ALPHA}_gamma_{GAMMA}_eps_{starting_eps}.png", bbox_inches='tight')
                plt.cla()
                plt.clf()
        if dataset:
            self.Q = Q.copy()
            return Q
        else:
            self.saveQ(Q,"Qmatrix")
