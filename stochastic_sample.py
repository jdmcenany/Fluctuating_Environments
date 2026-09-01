import numpy as np
import os
import pickle

def sample_epochs(xperp, sg, xc, vpar, vperp, tau, sigma_tau, ss, K, rng, N_samples=100000):
    M = max(1,int(sg / (2 * vpar * tau)))

    gamma1 = rng.gamma((tau / sigma_tau)**2, sigma_tau**2 / tau, size=(N_samples, K, M))
    gamma2 = rng.gamma((tau / sigma_tau)**2, sigma_tau**2 / tau, size=(N_samples, K, M))

    time_sums = ((gamma1 - gamma2) * 2 * tau / (gamma1 + gamma2)).sum(axis=2)

    n_idx = np.arange(K)
    w = np.exp(-max(sg, 2*vpar*tau) * n_idx / xc)

    weighted_sums = time_sums * w[np.newaxis, :]
    weighted_sums2 = time_sums**2 * w[np.newaxis, :]

    T = np.cumsum(weighted_sums[:, ::-1], axis=1)[:, ::-1]
    T2 = np.cumsum(weighted_sums2[:, ::-1], axis=1)[:, ::-1]

    J_idx = np.arange(K)
    A_factor = np.exp(max(sg, 2*vpar*tau) * (J_idx - 1) / xc)
    A = T * A_factor[np.newaxis, :]

    A2_0 = (T2 * A_factor[np.newaxis, :])[:, 0]

    term1 = (vpar / xc) * xperp * A[:, 0]

    J_range = np.arange(1, K)
    e_J = np.exp(-max(sg, 2*vpar*tau) * J_range / xc)
    A_J = A[:, 1:]
    N_Jm1 = time_sums[:, :K-1]

    term2 = (vpar / xc) * ss * max(1, 2*vpar*tau/sg) * np.sum(e_J[np.newaxis, :] * np.abs(A_J), axis=1)
    term3 = -(vpar / xc) * vperp * np.sum(e_J[np.newaxis, :] * N_Jm1 * A_J, axis=1)
    term4 = -(vpar / xc) * (vperp / 2) * A2_0

    return term1 + term2 + term3 + term4


def run_sample_epochs(seed, fname, *inds, trials=100, n_xperp=30, n_x=100,
                      N_samples=1_000_000, output_dir="."):
    """ Samples the fitness distribution for one parameter combination from the pickle at fname
    and saves the survival curve at each starting xperp.

    The survival curve at each xperp pools all trials*N_samples draws: the first batch fixes the
    x grid, then every batch contributes its exceedance counts.

    inds: index into ss_vals, followed by an index into the period_CV grid when the pickle holds more than one.
    """
    rng = np.random.default_rng(seed)

    with open(fname, "rb") as file:
        data = pickle.load(file)

    ss_vals = data["ss_vals"]
    period_CV_vals = data["dtau_vals"]
    tau = data["tau"]
    sg = data["s"]

    # A single period_CV means vpar, vperp and xc are indexed by ss alone, otherwise by (ss, period_CV)
    if len(period_CV_vals) == 1:
        ind1, = inds
        ind, period_CV = ind1, period_CV_vals[0]
    else:
        ind1, ind2 = inds
        ind, period_CV = (ind1, ind2), period_CV_vals[ind2]

    ss = ss_vals[ind1]
    sigma_tau = period_CV * tau * np.sqrt(2)
    vpar = data["v_vals"][ind]
    vperp = data["vt_vals"][ind]
    xc = data["xc_vals"][ind]

    xperp_vals = np.linspace(0, xc*1.5, n_xperp)

    x_vals_all = np.zeros((len(xperp_vals), n_x))
    surv_vals_all = np.zeros((len(xperp_vals), n_x))

    # Avoid K = 0 if xc is small compared to sg
    K = max(1, int(5 * xc / max(sg, 2*vpar*tau)))

    for xperp_ind, xperp in enumerate(xperp_vals):
        print(xperp_ind/len(xperp_vals), flush=True)

        counts = np.zeros(n_x)
        for i in range(trials):
            fitnesses = np.sort(sample_epochs(xperp, sg, xc, vpar, vperp, tau, sigma_tau, ss, K, rng, N_samples=N_samples))

            # The first batch fixes the x grid, up to its second largest sample
            if i == 0:
                if fitnesses.size < 2:
                    raise ValueError("Need at least two samples to define the x range.")
                x_vals = np.linspace(0, fitnesses[-2], n_x)

            counts += N_samples - np.searchsorted(fitnesses, x_vals, side="right")

        x_vals_all[xperp_ind, :] = x_vals
        surv_vals_all[xperp_ind, :] = counts / (trials * N_samples)

    sim_results = {
        "y0_vals": xperp_vals,
        "x_vals_all": x_vals_all,
        "surv_vals_all": surv_vals_all,
    }

    stem = os.path.splitext(os.path.basename(fname))[0]
    ind_str = "_".join(str(i) for i in inds)
    regime = "short" if 2*vpar*tau < sg else "long"
    fname_out = f"{output_dir}/WF_Results/Sampling_Results/sample_{regime}_epochs_{stem}_{ind_str}.pickle"
    with open(fname_out, "wb") as file:
        pickle.dump(sim_results, file)

    return sim_results, fname_out
