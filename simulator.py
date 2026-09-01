import numpy as np
import time
import ecosystem as ec

class Simulation(ec.Ecosystem):
    """
    Class contains an ecosystem (population + environment),
    as well as objects that contain a record of mutation, strategy frequencies.
    """

    def __init__(self, pop_size, init_state, sg, ss, ug, us, tau, sigma_tau, seed, use_test_strategy, minimal_output, ds0, apply_ds0_all, two_tau, measure_dist):
        ec.Ecosystem.__init__(self, pop_size, init_state, sg, ss, ug, us, tau, sigma_tau, seed, use_test_strategy, ds0, apply_ds0_all, two_tau, measure_dist)
        self.times = []
        self.states = []

        self.final_pop_mats = []
        self.final_states = []
        self.final_offsets = []
        
        self.mut_freqs1 = []
        self.mut_freqs2 = []
        self.mut_freqs3 = []
        self.test_freqs = []

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
            'test_freqs': self.test_freqs,
            'final_popmats': self.final_pop_mats,
            'final_states': self.final_states,
            'final_offsets': self.final_offsets,
            'rate_gen': self.rate_gen,
            'rate_spec': self.rate_spec,
            'backgrounds': self.backgrounds,
            'mean_backgrounds': self.mean_backgrounds,
            'mut_types': self.mut_types,
            'mean_bg_diff': self.mean_bg_diff,
            'bias_num': self.bias_num,
            'bias_spread': self.bias_spread,
            'bias_den': self.bias_den,
            'fitness_dist': self.fitness_dist,
            'epoch_lengths': self.epoch_lengths,
            'bias_epoch_starts': self.bias_epoch_starts 
        }

        self.minimal_sim_results = {
            'fixations': self.fixations,
            'pfix_adjustments': self.pfix_adjustments,
            'rate_gen': self.rate_gen,
            'rate_spec': self.rate_spec,
            'bias_num': self.bias_num,
            'bias_den': self.bias_den,
            'bias_spread': self.bias_spread
        }

    def record(self, final_flag = False):
        self.times.append(self.t)
        if np.mod(self.t, 10000) == 0: print(self.t, flush=True)
        self.states.append(self.state)
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
                if self.times[-1] > steps - 11:
                    self.record(final_flag = True)
                else:
                    self.record()
        if print_time:
            return time.time()-t0
        
    def calc_fitness_distribution(self):
        self.mut_freqs1.append(((np.arange(self.matrix.shape[0]).reshape(-1,1,1,1)+self.offset[0])*self.matrix).sum()/self.matrix.sum())
        self.mut_freqs2.append(((np.arange(self.matrix.shape[1]).reshape(1,-1,1,1)+self.offset[1])*self.matrix).sum()/self.matrix.sum())
        self.mut_freqs3.append(((np.arange(self.matrix.shape[2]).reshape(1,1,-1,1)+self.offset[2])*self.matrix).sum()/self.matrix.sum())
        self.test_freqs.append((self.matrix[:,:,:,1]).sum()/self.matrix.sum())

    def initialize(self):
        self.populate()
        self.record()
