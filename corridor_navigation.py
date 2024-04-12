import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.integrate import odeint

import cvxpy as cp


'''Similation parameters:
    maximal time step, time interval,
    start and end positions'''
sim_mdl = {'max_t': 500,
           'dt': .05,
           'x_lim': 2,
           'y_start': 3,
           'y_end': -2,
           'epsilon': .05,
           'err_rate': .05,
           'learn_rate': .05}

'''Agent parameters: 
    radius, sensing range,
    maximal velocity per time step'''
agent_mdl = {'r': .15,
            'sense': 2,
            'max_v': .5}

'''Obstacle parameters: 
    radius'''
obs_mdl = {'r': .5}

#@staticmethod
def agents_model(z, t, u):
    '''
    z[4*i+0], z[4*i+1]: Position of agent i in stationary coordinates
    z[4*i+2], z[4*i+3]: Velocity of agent i in stationary coordinates
    u[i][0], u[i][1]: input velocity of agent i
    '''
    N = len(z) // 4
    max_v = agent_mdl['max_v']

    dzdt = []
    for i in range(N):
        vx_i = z[4*i+2]
        if abs(vx_i) > max_v:
            vx_i = np.sign(vx_i) * max_v
        vy_i = z[4*i+3]
        if abs(vy_i) > max_v:
            vy_i = np.sign(vy_i) * max_v
        dzdt.append(vx_i)
        dzdt.append(vy_i)
        dzdt.append(u[2*i+0])
        dzdt.append(u[2*i+1])

    return dzdt

def dist2(a1, a2):
    return (a1[0] - a2[0]) ** 2 + (a1[1] - a2[1]) ** 2

def dist(a1, a2):
    return np.sqrt(dist2(a1, a2))

def lyapunov_f(z, i, d):
    return dist2(z[4*i:4*i+2], d[i])

def clf_c(z, i, d, ui, epsilon, delta_i):
    vec_xidi = [z[4*i+0] - d[i][0], z[4*i+1] - d[i][1]]
    return 2 * np.matmul(vec_xidi, ui) + epsilon * lyapunov_f(z, i, d) + delta_i <= 0

def h_agents(z, i, j, agent_mdl):
    return dist2(z[4*i:4*i+2], z[4*j:4*j+2]) - (2 * agent_mdl['r']) ** 2

def h_obstacle(z, obs, i, l, agent_mdl, obs_mdl):
    return dist2(z[4*i:4*i+2], obs[l]) - (agent_mdl['r'] + obs_mdl['r']) ** 2

def cbf_agents_ML(z, i, j, ui, zeta_a, eta_a, agent_mdl):
    vec_ij = [z[4*i+0] - z[4*j+0], z[4*i+1] - z[4*j+1]]
    vec_uivj = [ui[0] - z[4*j+2], ui[1] - z[4*j+3]]
    h = h_agents(z, i, j, agent_mdl)
    return 2 * np.matmul(vec_ij, vec_uivj) + zeta_a * (h ** eta_a) >= 0

def cbf_obstacle_ML(z, obs, i, l, ui, zeta_o, eta_o, agent_mdl, obs_mdl):
    vec_il = [z[4*i+0] - obs[l][0], z[4*i+1] - obs[l][1]]
    h = h_obstacle(z, obs, i, l, agent_mdl, obs_mdl)
    return 2 * np.matmul(vec_il, ui) + zeta_o * (h ** eta_o) >= 0

def edge_sensing_range(z, obs):
    N = len(z) // 4
    r_agent = agent_mdl['r']
    r_sense = agent_mdl['sense']
    M = len(obs)
    r_obs = obs_mdl['r']

    # Compute what is in the sensing range for each agent
    sensed_a = [[] for _ in range(N)]
    sensed_o = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if dist(z[4*i:4*i+2], z[4*j:4*j+2]) < r_sense + r_agent:
                sensed_a[i].append(j)
                sensed_a[j].append(i)
        for l in range(M):
            if dist(z[4*i:4*i+2], obs[l]) < r_sense + r_obs:
                sensed_o[i].append(l)
    
    return sensed_a, sensed_o


def qp_controller(z, obs, agent_mdl, obs_mdl, d, lmbd, zeta_a, eta_a, zeta_o, eta_o):
    '''Inputs:
	   z: Current agents' state
	   Output to be computed: 	
	   u: control velocity
    '''
    N = len(z) // 4
    sensed_a, sensed_o = edge_sensing_range(z, obs)

    r_agent_2 = agent_mdl['r'] ** 2

    u = []
    slack = []
    udim = 2
    epsilon = sim_mdl['epsilon']
    for i in range(N):

        if dist2(z[4*i:4*i+2], d[i]) < r_agent_2:
            u.append(None)
            u.append(None)
            slack.append(None)
        else:
            ui = cp.Variable(udim)
            di = cp.Variable(1)
            objective = cp.Minimize(cp.norm2(ui) + di ** 2)
            constraints = [clf_c(z, i, d, ui, epsilon, di)]
            for j in sensed_a[i]:
                constraints.append(cbf_agents_ML(z, i, j, ui, zeta_a(lmbd[i]), eta_a(lmbd[i]), agent_mdl))
            for l in sensed_o[i]:
                constraints.append(cbf_obstacle_ML(z, obs, i, l, ui, zeta_o(1), eta_o(1), agent_mdl, obs_mdl))
            problem = cp.Problem(objective, constraints)

            problem.solve()
            if problem.status != 'infeasible':
                slack.append(di.value)
                u.append(ui.value[0])
                u.append(ui.value[1])
            else:
                u.append(None)
                u.append(None)
                slack.append(None)

    return u, sensed_a


def soft_positive_linear(x):
    return np.log(1 + np.exp(x))

def next_lmbd(lmbd, err):
    eta = sim_mdl['learn_rate']
    eps = sim_mdl['err_rate']
    return [lmbd[i] + eta * (eps - err[i]) for i in range(len(lmbd))]

def path_control(z0, u0, lmbd, t, d, obs, agent_mdl, obs_mdl, zeta_a, eta_a, zeta_o, eta_o):
    '''
        z0,u0: initial state, velocity
        t: time sequence for simulation
        cntr: controller to be used
    '''
    nb_steps = len(t)
    N = len(z0) // 4 
    r_agent = agent_mdl['r']

    '''These vectors will store the state variables of the vehicle'''
    x = np.array([])
    y = np.array([])
    vx = np.array([])
    vy = np.array([])
    ''' Initialization of x, y, theta, xf, yf'''
    x = np.append(x, [z0[4*i+0] for i in range(N)])
    y = np.append(y, [z0[4*i+1] for i in range(N)])
    vx = np.append(x, [z0[4*i+2] for i in range(N)])
    vy = np.append(y, [z0[4*i+3] for i in range(N)])

    """ setting up the figure """
    plt.figure()
    ax = plt.gca()

    """This loop solves the ODE for each pair of time points
    with fixed controller input"""
    last_step = nb_steps - 2
    for s in range(1, nb_steps):
        # The next time interval to compute the solution over
        tspan = [t[s - 1], t[s]]

        # next controller input
        next_u, sensed_a = qp_controller(z0, obs, agent_mdl, obs_mdl, d, lmbd, zeta_a, eta_a, zeta_o, eta_o)
        for ui in next_u:
            if ui == None:
                print("Can't continue loop")
                print("Step: ", s)
                u0 = None
                break
        if u0 == None:
            last_step = s - 1
            break
        else:
            u0 = next_u

        # solve ODE for next time interval with input u0 from new initial set z0
        z = odeint(agents_model, z0, tspan, args=(u0,))
        # store solution (x,y) for plotting
        x = np.append(x, [z[1][4*i+0] for i in range(N)])
        y = np.append(y, [z[1][4*i+1] for i in range(N)])
        vx = np.append(x, [z[1][4*i+2] for i in range(N)])
        vy = np.append(y, [z[1][4*i+3] for i in range(N)])
    
        # next initial conditions
        z0 = z[1]
        # next conformal parameters
        err = [0] * N
        for i in range(N):
            for j in sensed_a[i]:
                if not cbf_agents_ML(z0, i, j, u0[2*i:2*i+2], zeta_a(lmbd[i]), eta_a(lmbd[i]), agent_mdl):
                    err[i] = 1
                    break  
        lmbd = next_lmbd(lmbd, err)

    '''Plot initial positions and obstacles'''
    for i in range(N):
        start_c = plt.Circle((x[i], y[i]), r_agent, fill=False) #to change
        ax.add_patch(start_c)
    for l in range(M):
        obs_c = plt.Circle((obs[l][0], obs[l][1]), obs_mdl['r'], fill=True)
        ax.add_patch(obs_c)

    '''Plot the trajectories of the agents'''
    for i in range(N):
        x_i = [x[N*s+i] for s in range(last_step)]
        y_i = [y[N*s+i] for s in range(last_step)]
        plt.plot(x_i, y_i, '--', color='g', linewidth=2)

    '''Plot goal positions'''
    for i in range(N):
        corner = [d[i][0] - r_agent, d[i][1] - r_agent]
        side = 2 * r_agent
        ax.add_patch(Rectangle(corner, side, side, linewidth=1,
                               edgecolor='r', facecolor='none'))

    '''plot parameters'''
    plt.legend(loc='lower right', fontsize='x-large')
    plt.tick_params(axis='both', labelsize=20)
    plt.title('test')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-4, 4])

    ax.set_aspect('equal', adjustable='box')

    print("Length of path:", len(x))
    plt.show()


''' Setup for runnint the simulation
    Thorizon: Time horizon for simulation
    Nsample: number of sample points that will be use to split [0, Thorizon]
    t will be the array of time points
'''
Nsample = int(1 / sim_mdl['dt'])
Thorizon = sim_mdl['max_t'] / Nsample
t = np.linspace(0, Thorizon, Nsample)

'''Initial state and input'''
N = 2
x_lim = sim_mdl['x_lim']
y_start = sim_mdl['y_start']
y_end = sim_mdl['y_end']
x_start = [2 * x_lim * i / (N+1) - x_lim for i in range(1, N+1)]

z0 = [x for i in range(N) for x in [x_start[i], y_start, 0, 0]]

u0 = [0 for _ in range(2 * N)]

'''Random obstacle generation'''
M = 4
r_obs = obs_mdl['r']
y_obs_min = y_end + agent_mdl['r'] + r_obs
y_obs_max = y_start - agent_mdl['r'] - r_obs
x_obs_min = - x_lim + r_obs
x_obs_max = x_lim - r_obs
obs = []
while len(obs) < M:
    candidate = (x_obs_max - x_obs_min) * np.random.random(2) + x_obs_min
    for o in obs:
        if dist(o, candidate) < 2 * r_obs:
            candidate = [None, None]
            break
    if candidate[0]:
        obs.append(candidate)

'''Goal generation'''
# directly underneath
d = [[x_start[i], y_end] for i in range(N)]
# two groups
pass
# all together
pass
# cone
pass

zeta_a = lambda x: x
eta_a = lambda _: 1
zeta_o = lambda _: 1
eta_o = lambda _: 1

lmbd0 = [1] * N

path_control(z0, u0, lmbd0, t, d, obs, agent_mdl, obs_mdl, zeta_a, eta_a, zeta_o, eta_o)
