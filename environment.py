class FlucEnvironment:
    def __init__(self, pop_size, init_state, switch_time, switch_drift):
        self.N = pop_size
        self.state = init_state
        self.tau = switch_time
        self.d_tau = switch_drift
        self.next_switch = 0