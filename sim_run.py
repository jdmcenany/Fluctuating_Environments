import simulator as sm
import numpy as np
import pickle

def wf_sim(sg, ss, ug, us, N, tau, dtau, rand_seed, run_time, use_test_strategy = False, save_freq = 10, minimal_output = False, save_id = -1, ds0=0, apply_ds0_all=False, two_tau=None):
    sim = sm.Simulation(float(N), 1, us, ss, ug, sg, tau/2, tau*dtau/np.sqrt(2), rand_seed, use_test_strategy,minimal_output,ds0,apply_ds0_all, two_tau)
    sim.initialize()
    sim.evolve(int(run_time), save_freq)

    if us < 0:
        fname = f'WF_Results/Summary_Sims/N_{N}_us_rare_ss_{ss}_ug_{ug}_sg_{sg}_tau_{tau}_dtau_{dtau}'
    else:
        fname = f'WF_Results/Summary_Sims/N_{N}_us_{us}_ss_{ss}_ug_{ug}_sg_{sg}_tau_{tau}_dtau_{dtau}'
    if minimal_output: fname += '_small_output'
    if two_tau is not None: fname += f'_long_tau_{two_tau[0]}_{two_tau[1]}'
    if ds0 != 0: fname += f'_ds0_{ds0}'
    if ds0 != 0 and apply_ds0_all: fname += '_all'
    fname_end = '.pickle'
    if save_id >= 0: fname_end = '_' + str(rand_seed - save_id) + '.pickle'
    with open(fname + fname_end, 'wb') as file:
        if minimal_output: pickle.dump(sim.minimal_sim_results, file)
        else: pickle.dump(sim.sim_results, file)
    #with open(fname + '_full.pickle', 'wb') as file:
    #    pickle.dump(sim, file)
