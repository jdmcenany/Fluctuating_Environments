class FlucEnvironment:
    def __init__(self, pop_size, init_state, tau, sigma_tau):
        self.N = pop_size
        self.state = init_state
        self.tau = tau
        self.sigma_tau = sigma_tau
        self.next_switch = 0
