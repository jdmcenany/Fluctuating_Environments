import numpy as np

class PopulationMatrix:
    """ Contains a numpy array representing the population.
    """

    def __init__(self):
        # Axes 0-2: number of generalist muts, specialist muts favoring state +1,
        # specialist muts favoring state -1. Axis 3: lineage (0 = resident, 1 = tagged test mutant).
        self.matrix = np.zeros((1,1,1,2), dtype=float)
        self.offset = [0,0,0]  # Min number of muts in each category
        

    def purge(self):
        """ Clears empty columns at beginning of self.matrix
        """
        nonzero_indices = np.where(self.matrix > 0)
        first0 = np.min(nonzero_indices[0])
        first1 = np.min(nonzero_indices[1])
        first2 = np.min(nonzero_indices[2])
        self.matrix = self.matrix[first0:,first1:,first2:,:]
        self.offset[0] += first0
        self.offset[1] += first1
        self.offset[2] += first2

class EcoEvoParameters:
    """Contains mutation rate and size parameters
    """
    def __init__(self, sg, ss, ug, us):
        self.sg = sg
        self.ss = ss
        self.ug = ug
        self.us = us

class Population(PopulationMatrix, EcoEvoParameters):
    """ Population contains all the objects related to the organisms in an ecosystem.
    """
    def __init__(self, sg, ss, ug, us):
        PopulationMatrix.__init__(self)
        EcoEvoParameters.__init__(self, sg, ss, ug, us)

    def purge_and_update(self):
        self.purge()
