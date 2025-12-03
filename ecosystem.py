import numpy as np
import population as pop
import environment as env

class Ecosystem(pop.Population, env.FlucEnvironment):
    def __init__(self, pop_size, init_state, ua, da, ux, dx, switch_time, switch_drift, seed, use_test_strategy, ds0, apply_ds0_all):
        pop.Population.__init__(self, ua, da, ux, dx)
        env.FlucEnvironment.__init__(self, pop_size, init_state, switch_time, switch_drift)
        self.t = 0
        self.rnd = np.random.default_rng(seed)
        
        if self.d_tau > 0:
            self.next_switch = self.rnd.gamma((self.tau/self.d_tau)**2, self.d_tau**2/self.tau)
            self.next_switch2 = self.next_switch + self.rnd.gamma((self.tau/self.d_tau)**2, self.d_tau**2/self.tau)
            self.next_switch3 = self.next_switch2 + self.rnd.gamma((self.tau/self.d_tau)**2, self.d_tau**2/self.tau)
        else:
            self.next_switch = self.tau
            self.next_switch2 = 2*self.tau
            self.next_switch3 = 3*self.tau
            
        # Parameters used when sampling one strategy mutant at a time
        self.next_mut = 0
        self.pfix_adjustments = []
        self.fixations = []
        self.fixation_times = []
        self.fixation_starts = []
        self.fixation_k = []
        self.k_mode = []
        self.k_max = []
        self.extra_waiting_time = 0
        self.death_flag = False
        self.backgrounds = []
        self.mut_types = []
        self.mean_backgrounds = []
        self.mean_bg_diff = []

        self.ds0 = ds0
        self.apply_ds0_all = apply_ds0_all
        self.use_test_strategy = use_test_strategy

    def populate(self):
        self.matrix[0][0][0][0] = self.N
        
    def lambdas(self):
        '''Calculate Poisson parameter lambda (mean) for each nonzero (sub)clade,
        to be used when drawing new population sizes in selection step.'''
        current_pop = self.matrix.sum()

        base_fitness = self.dx * (np.arange(self.matrix.shape[0]).reshape(-1,1,1,1) + np.arange(self.matrix.shape[1]).reshape(1,-1,1,1) + np.arange(self.matrix.shape[2]).reshape(1,1,-1,1))

        if self.ds0 != 0:
            if self.apply_ds0_all:
                base_fitness += self.ds0 * (np.arange(self.matrix.shape[1]).reshape(1,-1,1,1) + np.arange(self.matrix.shape[2]).reshape(1,1,-1,1))
            else:
                base_fitness += self.ds0 * np.arange(self.matrix.shape[3]).reshape(1,1,1,-1)
        exp_fits = self.matrix/current_pop * np.exp(base_fitness)

        env_bias = np.arange(self.matrix.shape[1]).reshape(1,-1,1,1) - np.arange(self.matrix.shape[2]).reshape(1,1,-1,1)

        lambdas = np.exp(self.state * self.da * env_bias) * exp_fits
        return lambdas[self.matrix > 0] * self.N / lambdas.sum()

    def selection_step(self):
        '''Given clades class, perform a selection step.'''
        lambdas = self.lambdas()
        self.matrix[self.matrix > 0] = self.safe_poisson(lambdas)
        
    def safe_poisson(self, lambdas):
        out = np.zeros(lambdas.shape)

        small_mask = lambdas < 1e5
        large_mask = ~small_mask

        if np.any(small_mask): out[small_mask] = self.rnd.poisson(lambdas[small_mask])
        if np.any(large_mask): out[large_mask] = self.rnd.normal(lambdas[large_mask],np.sqrt(lambdas[large_mask]))
        return out
    
    def safe_binomial(self, n, p):
        out = np.zeros(n.shape)

        binom_mask = n < 1e5
        poiss_mask = (n >= 1e5) & (n*p*(1-p) < 10)
        gauss_mask = (n >= 1e5) & (n*p*(1-p) >= 10)

        if np.any(binom_mask): out[binom_mask] = self.rnd.binomial(n[binom_mask].astype(int),p)
        if np.any(poiss_mask): out[poiss_mask] = self.rnd.poisson(n[poiss_mask] * p)
        if np.any(gauss_mask): out[gauss_mask] = self.rnd.normal(n[gauss_mask] * p, np.sqrt(n[gauss_mask]*p*(1-p)))

        out[out < 0] = 0
        return out

    def mutate_x_step(self, extend=1):
        '''Given clades class, perform a mutation step.'''
        if np.any(self.matrix[-1,:,:]):  # only extend axis when chance of mutant at nose
            self.matrix = np.concatenate((self.matrix, np.zeros((extend, self.matrix.shape[1], self.matrix.shape[2], 2), dtype=int)),axis=0)
        nonzero_indices = self.matrix > 0
        rolled_nonzero_indices = np.roll(nonzero_indices, 1, axis=0)
        x_mutants = self.safe_binomial(self.matrix[nonzero_indices], self.ux)
        # x_mutants = self.rnd.poisson(self.matrix[nonzero_indices] * self.ux)
        if np.any(x_mutants):
            self.matrix[nonzero_indices] -= x_mutants
            self.matrix[rolled_nonzero_indices] += x_mutants
            
    def add_rare_strategy_mut(self):
        if np.any(self.matrix[:,-1,:,:]):
            self.matrix = np.concatenate((self.matrix, np.zeros((self.matrix.shape[0],1,self.matrix.shape[2],2), dtype=float)),axis=1)
        if np.any(self.matrix[:,:,-1,:]):
            self.matrix = np.concatenate((self.matrix, np.zeros((self.matrix.shape[0],self.matrix.shape[1],1,2), dtype=float)),axis=2)
        nonzero_indices = np.where(self.matrix > 0)
        
        # If there are no specialists, initialize new one at a random time
        if np.max(nonzero_indices[1]) == np.min(nonzero_indices[1]) and np.max(nonzero_indices[2]) == np.min(nonzero_indices[2]):
            # Determine waiting time
            if self.next_mut == 0:
                if self.death_flag == True:
                    self.death_flag = False
                    self.fixation_times[-1] = self.t - self.fixation_times[-1]
                if self.t >= self.extra_waiting_time:
                    in_next_epoch = self.rnd.exponential(2 * self.tau) < self.next_switch3 - self.next_switch
                    if in_next_epoch:
                        self.extra_waiting_time = 0
                        self.next_mut = self.rnd.uniform(self.next_switch,self.next_switch3)
                    else:
                        self.extra_waiting_time = self.next_switch3

            elif self.t >= self.next_mut and self.extra_waiting_time == 0:
                self.death_flag = True
                
                q_min = np.argmax(self.matrix[:,0,0,0])
                q_max = np.max(nonzero_indices[0])
                q_mut = self.rnd.integers(q_min,q_max+1)
                
                starting_N = np.max((1/(2*(self.da + (1+q_mut)*self.dx)),1))

                self.pfix_adjustments.append((q_max+1-q_min)*self.matrix[q_mut,0,0,0]/(starting_N*self.matrix.sum()))
                self.matrix[q_mut][1][0][0] += starting_N
                self.fixations.append(0)
                self.next_mut = 0
                self.fixation_times.append(self.t)
                self.fixation_k.append(q_mut)
                if self.state == 1:
                    self.fixation_starts.append(self.t)
                else:
                    self.fixation_starts.append(self.t)
                self.k_mode.append(q_min)
                self.k_max.append(q_max)

    def add_test_strategy_mut(self):
        nonzero_indices = np.where(self.matrix > 0)

        # If test charge lineage has fixed, record
        if np.sum(self.matrix[:,:,:,0]) <= 0:
            self.fixations[-1] = 1
            self.matrix[:,:,:,0] = self.matrix[:,:,:,1]
            self.matrix[:,:,:,1] = 0*self.matrix[:,:,:,1]
        
        # If there are no "test charge" specialists, initialize new one at a random time
        if np.sum(self.matrix[:,:,:,1]) <= 0:
            # Determine waiting time
            if self.next_mut == 0:
                if self.death_flag == True:
                    self.death_flag = False
                    self.fixation_times[-1] = self.t - self.fixation_times[-1]
                if self.t >= self.extra_waiting_time:
                    in_next_epoch = self.rnd.exponential(2 * self.tau) < self.next_switch3 - self.next_switch
                    if in_next_epoch:
                        self.extra_waiting_time = 0
                        self.next_mut = self.rnd.uniform(self.next_switch,self.next_switch3)
                    else:
                        self.extra_waiting_time = self.next_switch3

            elif self.t >= self.next_mut and self.extra_waiting_time == 0:
                self.death_flag = True

                q1_min, q2_min, q3_min = np.min(nonzero_indices[0]), np.min(nonzero_indices[1]), np.min(nonzero_indices[2])
                q1_max, q2_max, q3_max = np.max(nonzero_indices[0]), np.max(nonzero_indices[1]), np.max(nonzero_indices[2])

                correct_mut_flag = False
                while not correct_mut_flag:
                    q1_mut, q2_mut, q3_mut = self.rnd.integers(q1_min,q1_max+1), self.rnd.integers(q2_min,q2_max+1), self.rnd.integers(q3_min,q3_max+1)
                    
                    if self.matrix[q1_mut,q2_mut,q3_mut,0] > 0: correct_mut_flag = True

                n_categories = 0
                for i in range(q1_min, q1_max+1):
                    for j in range(q2_min, q2_max+1):
                        for k in range(q3_min, q3_max+1):
                            n_categories += self.matrix[i,j,k,0] > 0
                
                mut_type = self.rnd.integers(-1, 2)
                starting_N = np.max((1/(2*(1+q1_mut+q2_mut+q3_mut)*(self.dx+self.da)),1))

                self.pfix_adjustments.append(n_categories*self.matrix[q1_mut,q2_mut,q3_mut,0]/(starting_N*self.matrix.sum()))
                while mut_type == 0 and self.ds0 > 0 and not self.apply_ds0_all: mut_type = self.rnd.integers(-1,2)


                if mut_type == 0:
                    self.matrix[q1_mut+1][q2_mut][q3_mut][1] += starting_N
                elif mut_type == -1:
                    self.matrix[q1_mut][q2_mut+1][q3_mut][1] += starting_N
                else:
                    self.matrix[q1_mut][q2_mut][q3_mut+1][1] += starting_N

                self.backgrounds.append((q1_mut,q2_mut,q3_mut))
                self.mut_types.append(mut_type)

                freq1 = ((np.arange(self.matrix.shape[0]).reshape(-1,1,1)+self.offset[0])*self.matrix[:,:,:,0]).sum()/self.matrix[:,:,:,0].sum()
                freq2 = ((np.arange(self.matrix.shape[1]).reshape(1,-1,1)+self.offset[1])*self.matrix[:,:,:,0]).sum()/self.matrix[:,:,:,0].sum()
                freq3 = ((np.arange(self.matrix.shape[2]).reshape(1,1,-1)+self.offset[2])*self.matrix[:,:,:,0]).sum()/self.matrix[:,:,:,0].sum()
                self.mean_backgrounds.append((freq1,freq2,freq3))

                mut_diff = ((np.arange(self.matrix.shape[1]).reshape(1,-1,1)+self.offset[1])*self.matrix[:,:,:,0])
                mut_diff -= ((np.arange(self.matrix.shape[2]).reshape(1,1,-1)+self.offset[2])*self.matrix[:,:,:,0])
                self.mean_bg_diff.append(np.abs(mut_diff).sum()/self.matrix[:,:,:,0].sum())

                self.fixations.append(0)
                self.next_mut = 0
                self.fixation_times.append(self.t)

    def mutate_strategy_step(self, extend=1):
        if np.any(self.matrix[:,-1,:,:]):
            self.matrix = np.concatenate((self.matrix, np.zeros((self.matrix.shape[0],extend,self.matrix.shape[2],2), dtype=int)),axis=1)
        if np.any(self.matrix[:,:,-1,:]):
            self.matrix = np.concatenate((self.matrix, np.zeros((self.matrix.shape[0],self.matrix.shape[1],extend,2), dtype=int)),axis=2)
        
        nonzero_indices = self.matrix > 0
        rolled_up_indices = np.roll(nonzero_indices, 1, axis=1)
        rolled_down_indices = np.roll(nonzero_indices, 1, axis=2)
        up_muts = self.safe_binomial(self.matrix[nonzero_indices], self.ua)
        down_muts = self.safe_binomial(up_muts, 0.5)
        up_muts = up_muts - down_muts
                
        if np.any(up_muts):
            self.matrix[nonzero_indices] -= up_muts
            self.matrix[rolled_up_indices] += up_muts
        if np.any(down_muts):
            self.matrix[nonzero_indices] -= down_muts
            self.matrix[rolled_down_indices] += down_muts
        
    def env_shift_step(self):
        if self.t >= self.next_switch:
            self.state *= -1
            self.next_switch = self.next_switch2
            self.next_switch2 = self.next_switch3
            if self.d_tau > 0:
                self.next_switch3 += self.rnd.gamma((self.tau/self.d_tau)**2, self.d_tau**2/self.tau)
            else:
                self.next_switch3 += self.tau
            
    def full_step(self, purge_and_update=True):
        self.selection_step()
        self.mutate_x_step()

        if self.ua >= 0:
            self.mutate_strategy_step()
            if self.use_test_strategy: self.add_test_strategy_mut()
        else:
            self.add_rare_strategy_mut()
        self.t += 1
        self.env_shift_step()
        if purge_and_update:
            offset_old = self.offset[1] + self.offset[2]
            self.purge_and_update()
            if self.ua < 0:
                if self.offset[1] + self.offset[2] > offset_old:
                    self.fixations[-1] = 1
        
       
