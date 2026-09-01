import simulator as sm
import numpy as np
import pickle

def wf_sim(sg, ss, ug, us, N, period, period_CV, rand_seed, run_time, use_test_strategy = False, save_freq = 10, minimal_output = False, save_id = -1, ds0=0, apply_ds0_all=False, two_tau=None, measure_dist = True):
    r""" Runs a Wright-Fisher simulation of a population in a fluctuating environment and pickles the results.

    sg: Mean fitness effect of every mutation
    ss: Tradeoff strength of specialist mutations (\Delta s)
    ug: generalist mutation rate
    us: total specialist mutation rate. Set -1 for rare specialists
    N: population size
    period: environmental period = 2*tau
    period_CV: strength of seasonal drift = sigma_tau / (tau * sqrt(2))
    ds0: shift in specialist mean fitness, so specialist sbar = sg + ds0
    apply_ds0_all: if True, applies ds0 to all specialists; if False, applies ds0 only to test lineages
    use_test_strategy: Flag to add test lineage to calculate more detailed fixation statistics, when us > 0
    two_tau: Alternate stochasticity mode
    minimal_output: pickle the reduced results dict instead of the full one.
    measure_dist: accumulate the 2D mutation-count/bias histogram. Only active when period_CV == 0.
    """
    sim = sm.Simulation(float(N), 1, sg, ss, ug, us, period/2, period*period_CV/np.sqrt(2), rand_seed, use_test_strategy,minimal_output,ds0,apply_ds0_all, two_tau, measure_dist)
    sim.initialize()
    sim.evolve(int(run_time), save_freq)

    if us < 0:
        fname = f'WF_Results/Summary_Sims/N_{N}_us_rare_ss_{ss}_ug_{ug}_sg_{sg}_tau_{period}_dtau_{period_CV}'
    else:
        fname = f'WF_Results/Summary_Sims/N_{N}_us_{us}_ss_{ss}_ug_{ug}_sg_{sg}_tau_{period}_dtau_{period_CV}'
    if minimal_output: fname += '_small_output'
    if two_tau is not None: fname += f'_long_tau_{two_tau[0]}_{two_tau[1]}'
    if ds0 != 0: fname += f'_ds0_{ds0}'
    if ds0 != 0 and apply_ds0_all: fname += '_all'
    fname_end = '.pickle'
    if save_id >= 0: fname_end = '_' + str(rand_seed - save_id) + '.pickle'
    with open(fname + fname_end, 'wb') as file:
        if minimal_output: pickle.dump(sim.minimal_sim_results, file)
        else: pickle.dump(sim.sim_results, file)
