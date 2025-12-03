import numpy as np

class PopulationMatrix:
    """
    Contains a numpy array whose rows represent different strategies or strategy-lineages,
    and whose columns represent different (discrete) fitnesses (i.e. number of mutations).
    Also contains methods to calculate sums and averages along different axes.
    """

    def __init__(self, test_lineage = False):
        # Cols: Pure  beneficial, more specialist, less specialist, favored env
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
    def __init__(self, ua, da, ux, dx):
        self.ua = ua
        self.da = da
        self.ux = ux
        self.dx = dx

class Population(PopulationMatrix, EcoEvoParameters):
    """ Population contains all the objects related to the organisms in an ecosystem.
    """
    def __init__(self, ua, da, ux, dx):
        PopulationMatrix.__init__(self)
        EcoEvoParameters.__init__(self, ua, da, ux, dx)

    def purge_and_update(self):
        self.purge()
