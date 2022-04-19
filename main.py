import LeavesWorld
import QLearning
from QLearning import clear
import numpy as np
import random
import Tester
from Randomizer import Randomizer

def getDimensions(mode = "train", dataset = False) -> tuple[int, int, int, int, QLearning.QLearning]:
    my_randomizer = Randomizer()
    rs_train, _, rs_execute, _ = my_randomizer.randomStates()
    clear()
    n=input("***********************\n    LeafWrorld v1.0\n***********************\nSelect n dimension: ")
    clear()
    m=input(f"***********************\n    LeafWrorld v1.0\n***********************\nn = {n}\nSelect m dimension: ")
    clear()
    percentage=input(f"***********************\n    LeafWrorld v1.0\n***********************\nn = {n}\nm = {m}\nSelect % of leaves in the maze (if not a valid percentage it makes it random): ")
    if int(percentage) < 0 or int(percentage) > 100:
        percentage = random.randint(0, 101)
    clear()
    max_steps=input(f"***********************\n    LeafWrorld v1.0\n***********************\nn = {n}\nm = {m}\nleaves = {percentage}%\nSelect max steps: ")
    if not dataset:
        if mode == "train":
            env = LeavesWorld.LeavesWorld(rs=rs_train, m=int(m), n=int(n), p_leaves=int(percentage))
            QL = QLearning.QLearning(env, rs=rs_train)
            return n, m, percentage, int(max_steps), QL
        elif mode == "execute":
            env = LeavesWorld.LeavesWorld(rs=rs_train, m=int(m), n=int(n), p_leaves=int(percentage))
            QL = QLearning.QLearning(env, rs=rs_execute)
            return n, m, percentage, int(max_steps), QL
    else:
        clear()
        dataset_size=input(f"***********************\n    LeafWrorld v1.0\n***********************\nn = {n}\nm = {m}\nleaves = {percentage}%\nMax steps = {max_steps}\nSelect dataset size: ")
        if mode == "train":
            env = LeavesWorld.LeavesWorld(rs=rs_train, m=int(2), n=int(2), p_leaves=int(0))
            QL = QLearning.QLearning(env, rs=rs_train)
            return int(n), int(m), int(percentage), int(dataset_size), int(max_steps), QL
        elif mode == "execute":
            env = LeavesWorld.LeavesWorld(rs=rs_train, m=int(2), n=int(2), p_leaves=int(0))
            QL = QLearning.QLearning(env, rs=rs_execute)
            return int(n), int(m), int(percentage), int(dataset_size), int(max_steps), QL

def main():
    random.seed(1)
    clear()
    command=input(f"***********************\n    LeafWrorld v1.0\n***********************\nTraining (1)\nExecute step-by step (2)\nTraining with dataset (3)\nExecute with dataset (4)\nExit (0) ")
    if command == '1':
        clear()
        n, m, percentage, max_steps, QL = getDimensions(mode="train")
        clear()
        print(f"Training\nn = {n}\nm = {m}\nLeaves = {percentage}%\nMax steps = {max_steps}")
        QL.training(epochs = 70000, steps = max_steps, ALPHA= 0.1, GAMMA = 1.0, EPS = 1.0, plot=True)
    elif command == '2':
        clear()
        n, m, percentage, max_steps, QL = getDimensions(mode="execute")
        clear()
        auto=input("***********************\n    LeafWrorld v1.0\n***********************\nSelect autoplay: y/n   (Default yes)\n")
        clear()
        if auto == "n":
            print(f"Execute\nn = {n}\nm = {m}\nleaves = {percentage}%")
            QL.execute(auto=False, max_steps=max_steps)
        else:
            print(f"Execute\nn = {n}\nm = {m}\nleaves = {percentage}%")
            QL.execute(max_steps=max_steps, fast=False)
    elif command == '3':
        n, m, percentage, dataset_size, max_steps, QL = getDimensions(mode="train",dataset=True)
        clear()
        print("Training")
        QL.trainingWithDataset(QL.createDataSet(min_lenght=n, max_lenght=m, percentage=percentage, dimension=dataset_size), max_steps=max_steps)
    elif command == '4':
        n, m, percentage, dataset_size, max_steps, QL = getDimensions(mode="execute",dataset=True)
        clear()
        print("Executing")
        QL.executeOnDataset(QL.createDataSet(min_lenght=n, max_lenght=m, percentage=percentage, dimension=dataset_size), max_steps=max_steps)
    # funzionalità utilizzate per il tesing e il debug
    # non sono implementate graficamente ma comunque utilizzabili se inserito il giusto numero nel prompt
    elif command == '5':
        my_randomizer = Randomizer()
        rs_train, _, rs_execute, _ = my_randomizer.randomStates()
        env = LeavesWorld.LeavesWorld(rs=rs_execute,m=7,n=7,p_leaves=40)
        QL = QLearning.QLearning(env, rs=rs_execute)
        QL.execute(max_steps=200, fast=True)
    elif command == '6':
        my_randomizer = Randomizer()
        rs_train, _, rs_execute, _ = my_randomizer.randomStates()
        env = LeavesWorld.LeavesWorld(rs=rs_train,m=7,n=7,p_leaves=40)
        QL = QLearning.QLearning(env, rs=rs_train)
        QL.training(epochs = 10000, steps = 200, ALPHA= 0.1, GAMMA = 1.0, EPS = 1.0, plot=True)
    elif command == '7':
        tester = Tester.Tester(10000, 200, 20000, 200, 1)
        tester.test()
    elif command == '8':
        tester = Tester.Tester(10000, 200, 20000, 200, 5)
        tester.test_values()
    elif command == '9':
        tester = Tester.Tester(10000, 200, 20000, 200, 15)
        tester.multi_test()
    else:
        print('End\n')

if __name__ == '__main__':
    main()
