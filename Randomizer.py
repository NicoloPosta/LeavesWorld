import numpy as np

class Randomizer(object):

    def __init__(self) -> None:
        self.a = 1
    
    # Genera random states per training and execution
    def randomStates(self):
        np.random.seed(1)
        rs_train = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123456789)))
        savedState_train = rs_train.get_state()
        rs_execute = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(987654321)))
        savedState_execute = rs_execute.get_state()
        return rs_train, savedState_train, rs_execute, savedState_execute