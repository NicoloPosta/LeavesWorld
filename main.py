import LeavesWorld
import QLearning
from QLearning import clear
import numpy as np
import random

def getDimensions(dataset = False) -> tuple[int, int, int, int, QLearning.QLearning]:
    clear()
    n=input("***********************\n    LeafWrorld v1.0\n***********************\nSelect n dimension: ")
    clear()
    m=input(f"***********************\n    LeafWrorld v1.0\n***********************\nn = {n}\nSelect m dimension: ")
    clear()
    percentage=input(f"***********************\n    LeafWrorld v1.0\n***********************\nn = {n}\nm = {m}\nSelect % of leaves in the maze (if not a valid percentage it makes it random): ")
    if int(percentage) < 0 or int(percentage) > 100:
        percentage = np.random.randint(0, 101)
    clear()
    max_steps=input(f"***********************\n    LeafWrorld v1.0\n***********************\nn = {n}\nm = {m}\nleaves = {percentage}%\nSelect max steps: ")
    if not dataset:
        env = LeavesWorld.LeavesWorld(int(m),int(n),int(percentage))
        QL = QLearning.QLearning(env)
        return n, m, percentage, int(max_steps), QL
    else:
        clear()
        dateset_size=input(f"***********************\n    LeafWrorld v1.0\n***********************\nn = {n}\nm = {m}\nleaves = {percentage}%\nMax steps = {max_steps}\nSelect dataset size: ")
        env = LeavesWorld.LeavesWorld(int(2),int(2),int(0))
        QL = QLearning.QLearning(env)
        return int(n), int(m), int(percentage), int(dateset_size), int(max_steps),QL

def main():
    np.random.seed(1)
    random.seed(1)
    clear()
    command=input(f"***********************\n    LeafWrorld v1.0\n***********************\nTraining (1)\nExecute step-by step (2)\nTraining with dataset (3)\nExecute with dataset (4)\nExit (0) ")
    if command == '1':
        clear()
        n, m, percentage, max_steps, QL = getDimensions()
        clear()
        print(f"Training\nn = {n}\nm = {m}\nLeaves = {percentage}%\nMax steps = {max_steps}")
        QL.training(epochs = 80000, steps = max_steps, ALPHA= 0.1, GAMMA = 1.0, EPS = 1.0, plot=True)
    elif command == '2':
        clear()
        n, m, percentage, max_steps, QL = getDimensions()
        clear()
        auto=input("***********************\n    LeafWrorld v1.0\n***********************\nSelect autoplay: y/n   (Default yes)\n")
        clear()
        if auto == "n":
            print(f"Execute\nn = {n}\nm = {m}\nleaves = {percentage}%")
            QL.execute(auto=False, max_steps=max_steps)
        else:
            print(f"Execute\nn = {n}\nm = {m}\nleaves = {percentage}%")
            QL.execute(max_steps=max_steps)
    elif command == '3':
        n, m, percentage, dateset_size, max_steps, QL = getDimensions(dataset=True)
        clear()
        print("Training")
        QL.trainingWithDataset(QL.createDataSet(min_lenght=n, max_lenght=m, percentage=percentage, dimension=dateset_size), max_steps=max_steps)
    elif command == '4':
        n, m, percentage, dateset_size, max_steps, QL = getDimensions(dataset=True)
        clear()
        print("Executing")
        QL.executeOnDataset(QL.createDataSet(min_lenght=n, max_lenght=m, percentage=percentage, dimension=dateset_size), max_steps=max_steps)
    else:
        print('End\n')

if __name__ == '__main__':
    main()
