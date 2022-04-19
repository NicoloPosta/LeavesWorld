import LeavesWorld
import QLearning
from QLearning import clear
import numpy as np
from Randomizer import Randomizer

class Tester(object):

    def __init__(self, epochs_start, steps_start, epoch_increment, steps_increment, repetitions):
        self.epochs_start = epochs_start
        self.steps_start = steps_start
        self.epoch_increment = epoch_increment
        self.steps_increment = steps_increment
        self.repetitions = repetitions
        my_randomizer = Randomizer()
        self.rs_train, self.savedState_train, self.rs_execute, self.savedState_execute = my_randomizer.randomStates()
    
    # effettua il training e test con valori di epoche e passi diversi
    def test(self):
        clear()
        for i in range(self.repetitions):
            major_epoch_size = self.epochs_start + (self.epoch_increment * i)
            for j in range(self.repetitions):
                self.rs_train.set_state(self.savedState_train)
                self.rs_execute.set_state(self.savedState_execute)
                major_steps_size = self.steps_start + (self.steps_increment * j)
                print(f"------------------------------------------------------\nExecution with {major_epoch_size} epochs and {major_steps_size} steps.\n------------------------------------------------------")
                env = LeavesWorld.LeavesWorld(self.rs_train, 7,7,40)
                QL = QLearning.QLearning(env, self.rs_train)
                QL.training(epochs = major_epoch_size, steps = major_steps_size, ALPHA= 0.1, GAMMA = 1.0, EPS = 1.0, plot=True)
                env = LeavesWorld.LeavesWorld(self.rs_execute, 7,7,40)
                QL = QLearning.QLearning(env, self.rs_execute)
                QL.execute(max_steps=major_steps_size, fast=True)

    # effettua il training e test con valori gamma diversi
    def test_values(self):
        clear()
        max_steps = 1000
        epochs = 70000
        alpha = 0.1
        gamma = 0.2
        eps = 1.0
        for i in range(self.repetitions):
            self.rs_train.set_state(self.savedState_train)
            self.rs_execute.set_state(self.savedState_execute)
            print(f"------------------------------------------------------\nExecution with {alpha} ALPHA, {gamma*(i)} GAMMA and {eps} EPS\n------------------------------------------------------")
            env = LeavesWorld.LeavesWorld(self.rs_train, 7,7,40)
            QL = QLearning.QLearning(env, self.rs_train)
            QL.training(epochs = epochs, steps = max_steps, ALPHA= alpha, GAMMA = round(gamma*(i),1), EPS = eps, plot=True)
            env = LeavesWorld.LeavesWorld(self.rs_execute, 7,7,40)
            QL = QLearning.QLearning(env, self.rs_execute)
            QL.execute(max_steps=max_steps, fast=True)

    # effettua il training e test con matrici di grandezza diversa
    def multi_test(self):
        clear()
        print("Start Multitest")
        max_steps = 1000
        epochs = 70000
        alpha = 0.1
        gamma = 1.0
        eps = 1.0
        for i in range(self.repetitions):
            if i == 0 or i == 1:
                continue
            self.rs_train.set_state(self.savedState_train)
            self.rs_execute.set_state(self.savedState_execute)
            print(f"------------------------------------------------------\nExecution with matrix of size {i}x{i} \n------------------------------------------------------")
            env = LeavesWorld.LeavesWorld(self.rs_train, i,i,40)
            QL = QLearning.QLearning(env, self.rs_train)
            QL.training(epochs = epochs, steps = max_steps, ALPHA= alpha, GAMMA = gamma, EPS = eps, plot=True)
            env = LeavesWorld.LeavesWorld(self.rs_execute, 7,7,40)
            QL = QLearning.QLearning(env, self.rs_execute)
            QL.execute(max_steps=max_steps, fast=True)