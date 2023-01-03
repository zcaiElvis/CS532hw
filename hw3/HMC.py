import torch as tc

def HMC_sampling(lnf, start, n_points=int(1e3), M=1., dt=0.1, T=1., verbose=False):
    '''
    Hamiltonian Monte Carlo to create a chain of length n
    lnf: ln(f(x)) natural logarithm of the target function
    start: starting location in parameter space
    n_points: Number of points per chain
    M: Mass for the 'particles' TODO: Make matrix
    dt: Time-step for the particles
    T: Integration time per step for the particles
    '''
    # Functions for leap-frog integration
    def get_gradient(x, lnf):
        x = x.detach()
        x.requires_grad_()
        lnf(x).backward()
        dlnfx = x.grad
        x = x.detach() # TODO: Not sure if this is necessary
        return dlnfx
    def leap_frog_step(x, p, lnf, M, dt):
        dlnfx = get_gradient(x, lnf)
        p_half = p+0.5*dlnfx*dt
        x_full = x+p_half*dt/M
        dlnfx = get_gradient(x_full, lnf)
        p_full = p_half+0.5*dlnfx*dt
        return x_full, p_full
    def leap_frog_integration(x_init, p_init, lnf, M, dt, T):
        N_steps = int(T/dt)
        x, p = tc.clone(x_init), tc.clone(p_init)
        for _ in range(N_steps):
            x, p = leap_frog_step(x, p, lnf, M, dt)
        return x, p
    def Hamiltonian(x, p, lnf, M):
        T = 0.5*tc.dot(p, p)/M
        V = -lnf(x)
        return T+V
    # MCMC step
    n = len(start)
    x_old = tc.clone(start); xs = []; n_accepted = 0
    for i in range(n_points):
        p_old = tc.normal(0., 1., size=(n,))
        if i == 0: H_old = 0.
        x_new, p_new = leap_frog_integration(x_old, p_old, lnf, M, dt, T)
        H_new = Hamiltonian(x_new, p_new, lnf, M)
        acceptance = 1. if (i == 0) else min(tc.exp(H_old-H_new), 1.) # Acceptance probability
        accept = (tc.rand((1,)) < acceptance)
        if accept: x_old, H_old = x_new, H_new; n_accepted += 1
        xs.append(x_old)
    chain = tc.stack(xs)
    if verbose: print('Acceptance fraction: %1.2f'%(n_accepted/n_points))
    return chain