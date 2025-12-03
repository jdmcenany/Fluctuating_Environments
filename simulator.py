import numpy as np
import time
import copy
import ecosystem as ec
#import matplotlib.pyplot as plt

class Simulation(ec.Ecosystem):
    """
    Class contains an ecosystem (population + environment),
    as well as objects that contain a record of mutation, strategy frequencies.
    """

    def __init__(self, pop_size, init_state, ua, da, ux, dx, switch_time, switch_drift, seed, use_test_strategy, minimal_output, ds0, apply_ds0_all):
        ec.Ecosystem.__init__(self, pop_size, init_state, ua, da, ux, dx, switch_time, switch_drift, seed, use_test_strategy, ds0, apply_ds0_all)
        self.times = []
        self.offsets = []
        self.states = []
        self.avg_fitness = []
        self.avg_pure_fitness = []

        self.pop_mats = []
        self.pfixes = []
        self.fitness_distribution1 = []
        self.fitness_distribution2 = []
        self.prop_offsets = []

        self.final_pop_mats = []
        self.final_states = []
        self.final_offsets = []
        
        self.mut_freqs1 = []
        self.mut_freqs2 = []
        self.mut_freqs3 = []

        self.rate_gen = [0]
        self.rate_spec = [0]

        self.sim_results = {
            'times': self.times,
            'fixations': self.fixations,
            'fixation_times': self.fixation_times,
            'fixation_starts': self.fixation_starts,
            'states': self.states,
            'pfix_adjustments': self.pfix_adjustments,
            'mut_freqs1': self.mut_freqs1,
            'mut_freqs2': self.mut_freqs2,
            'mut_freqs3': self.mut_freqs3,
            'final_popmats': self.final_pop_mats,
            'final_states': self.final_states,
            'final_offsets': self.final_offsets,
            'rate_gen': self.rate_gen,
            'rate_spec': self.rate_spec
        }

        self.minimal_sim_results = {
            'fixations': self.fixations,
            'pfix_adjustments': self.pfix_adjustments,
            'rate_gen': self.rate_gen,
            'rate_spec': self.rate_spec
        }

    def record(self, final_flag = False):
        self.times.append(self.t)
        if len(self.offsets) > 1:
            if self.offsets[-2][0] < self.offsets[-1][0]:
                pass
                #print("Generalist fixed")
            if self.offsets[-2][1] + self.offsets[-2][2] < self.offsets[-1][1]+self.offsets[-1][2]:
                pass
                #print("Specialist fixed at time ")# + str(self.t/(np.log(self.da/self.ua)/self.da)))
        self.offsets.append(self.offset.copy())
        #self.pop_mats.append(self.matrix)
        self.states.append(self.state)
        #if self.t > 0:
            #self.calc_pfix()
        self.calc_fitness_distribution()
        if len(self.times) >= 110:
            self.rate_gen[-1] = np.polyfit(np.array(self.times)[100:], np.array(self.mut_freqs1)[100:],1)[0]
            self.rate_spec[-1] = np.polyfit(np.array(self.times)[100:], np.array(self.mut_freqs2)[100:] + np.array(self.mut_freqs3)[100:],1)[0]

        if final_flag:
            self.final_pop_mats.append(self.matrix)
            self.final_states.append(self.state)
            self.final_offsets.append(self.offset.copy())

    def evolve(self, steps, steps_per_record, print_time=True, purge_and_update=True):
        t0 = time.time()
        for i in range(1, steps + 1):
            self.full_step(purge_and_update=purge_and_update)
            if i % steps_per_record == 0:
                if self.times[-1] > steps - 10*400:
                    self.record(final_flag = True)
                else:
                    self.record()
                #print(100*i/steps)
                #print(np.sum(np.multiply(np.arange(self.matrix.shape[0]) + self.offset[0], np.sum(self.matrix,axis=(1,2))))/np.sum(self.matrix))
        if print_time:
            return time.time()-t0
        
    def calc_fitness_distribution(self):
        #exp_fits = self.dx * (np.arange(self.matrix.shape[0]).reshape(-1,1,1) + np.arange(self.matrix.shape[1]).reshape(1,-1,1) + np.arange(self.matrix.shape[2]).reshape(1,1,-1) + self.offset[0]+self.offset[1]+self.offset[2])
        #env_bias = np.arange(self.matrix.shape[1]).reshape(1,-1,1) - np.arange(self.matrix.shape[2]).reshape(1,1,-1) + self.offset[1] - self.offset[2]

        self.mut_freqs1.append(((np.arange(self.matrix.shape[0]).reshape(-1,1,1,1)+self.offset[0])*self.matrix).sum()/self.matrix.sum())
        self.mut_freqs2.append(((np.arange(self.matrix.shape[1]).reshape(1,-1,1,1)+self.offset[1])*self.matrix).sum()/self.matrix.sum())
        self.mut_freqs3.append(((np.arange(self.matrix.shape[2]).reshape(1,1,-1,1)+self.offset[2])*self.matrix).sum()/self.matrix.sum())
        
        #if self.mut_freqs2[-1] + self.mut_freqs3[-1] >= 1:
        #    print(self.mut_freqs2[-1] + self.mut_freqs3[-1])
        
        #fitness_mat_1 = self.da * env_bias + exp_fits
        #fitness_mat_2 = -self.da * env_bias + exp_fits
        
        #if self.state == 1:
        #    self.avg_fitness.append((fitness_mat_1*self.matrix).sum()/self.matrix.sum())
        #else:
        #    self.avg_fitness.append((fitness_mat_2*self.matrix).sum()/self.matrix.sum())
        #self.avg_pure_fitness.append((exp_fits*self.matrix).sum()/self.matrix.sum())
        
        #dists1 = []
        #dists2 = []
        #for i in np.unique([fitness_mat_1,fitness_mat_2]):
        #    dists1.append((i,self.matrix[fitness_mat_1 == i].sum()))
        #    dists2.append((i,self.matrix[fitness_mat_2 == i].sum()))
        #self.fitness_distribution1.append(dists1)
        #self.fitness_distribution2.append(dists2)
        #tempmat = self.matrix.sum(axis=0)

        #print(self.t/(np.log(self.da/self.ua)/self.da))
        """
        if len(self.times) % 10000 == 0:
            plt.cla()
            if self.state == 1:
                plt.hist(fitness_mat_1.squeeze().flatten(),bins=30,density=False,fc=(1, 0, 0, 0.75),weights=self.matrix.flatten()/self.matrix.sum(),log=True)
                plt.hist(fitness_mat_2.squeeze().flatten(),bins=30,density=False,fc=(0, 0, 1, 0.1),weights=self.matrix.flatten()/self.matrix.sum(),log=True)
            else:
                plt.hist(fitness_mat_2.squeeze().flatten(),bins=30,density=False,fc=(0, 0, 1, 0.75),weights=self.matrix.flatten()/self.matrix.sum(),log=True)
                plt.hist(fitness_mat_1.squeeze().flatten(),bins=30,density=False,fc=(1, 0, 0, 0.1),weights=self.matrix.flatten()/self.matrix.sum(),log=True)
            #plt.plot(np.array([-0.002,0.002]),np.array([1/(2*self.da),1/(2*self.da)]))
            plt.show()
            
            #tclick = np.log(self.da/self.ua)**2/(2*self.da*np.log(self.N*self.da))
            #print("Time " + str(self.t/tclick))
            #print(self.avg_fitness[-1]/self.da)
        """
            
        self.prop_offsets.append((self.matrix[:,1:,:].sum() + self.matrix[:,0,1:].sum())/self.matrix.sum())
        if len(self.times) % 1000 == 0:
            pass
            #print(self.prop_offsets[-1])
            #print(self.matrix[:,1:,:].sum())
            
            #print(tempmat[env_bias.squeeze() == 1].sum()/tempmat[env_bias.squeeze() == 0].sum())
        
        
    def calc_pfix(self):        
        pure = self.offset[0]/(self.t*self.N)
        spec = (self.offset[1]+self.offset[2])/(self.t*self.N)
        #self.pfixes.append((pure/self.ux, spec/self.ua))
        #print(self.offset)

    def initialize(self):
        self.populate()
        self.record()
