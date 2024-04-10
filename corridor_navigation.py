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
           'y_end': -2}

'''Agent parameters: 
    radius, sensing range,
    maximal velocity per time step'''
agent_mdl = {'r': .15,
            'sense': 2,
            'max_v': .5}

'''Obstacle parameters: 
    radius'''
obs_mdl = {'r': .5}


def agents_model(z, t, u):
    '''
    z[i][0], z[i][1]: Position of agent i in stationary coordinates
    z[i][2], z[i][3]: Velocity of agent i in stationary coordinates
    u[i][0], u[i][1]: input velocity of agent i
    '''
    N = len(z)
    
    dzdt = [[z[i][2], 
             z[i][3], 
             u[i][0], 
             u[i][1]] 
             for i in range(N)]
    return dzdt

def dist2(a1, a2):
    return (a1[0] - a2[0]) ** 2 + (a1[1] - a2[1]) ** 2

def dist(a1, a2):
    return np.sqrt(dist2(a1, a2))

def lyapunov_f(z, i, d):
    return dist2(z[i], d[i])

def clf_c(z, i, d, ui, epsilon, delta_i):
    for i in range(len(z)):
        vec_pd = [z[i][0] - d[i][0], z[i][1] - d[i][1]]
        if 2 * np.matmul(vec_pd, ui) + epsilon * lyapunov_f(z, i, d) + delta_i > 0:
            return False
    return True

def h_agents(z, i, j, agent_mdl):
    return dist2(z[i], z[j]) - (2 * agent_mdl['r']) ** 2

def h_obstacle(z, obs, i, l, agent_mdl, obs_mdl):
    return dist2(z[i], obs[l]) - (agent_mdl['r'] + obs_mdl['r']) ** 2

def cbf_agents_ML(z, i, j, ui, zeta_a, eta_a, agent_mdl):
    vec_ij = [z[i][0] - z[j][0], z[i][1] - z[j][1]]
    vec_uivj = [ui[0] - z[j][2], ui[1] - z[j][3]]
    h = h_agents(z, i, j, agent_mdl)
    return 2 * np.matmul(vec_ij, vec_uivj) + zeta_a[i] * (h ** eta_a[i]) >= 0

def cbf_obstacle_ML(z, obs, i, l, ui, zeta_o, eta_o, agent_mdl, obs_mdl):
    vec_il = [z[i][0] - obs[l][0], z[i][1] - obs[l][1]]
    h = h_obstacle(z, obs, i, l, agent_mdl, obs_mdl)
    return 2 * np.matmul(vec_il, ui) + zeta_o[i] * (h ** eta_o[i]) >= 0

def sensing_range(z, obs):
    N = len(z)
    r_agent = agent_mdl['r']
    r_sense = agent_mdl['sense']
    M = len(obs)
    r_obs = obs_mdl['r']

    # Compute what is in the sensing range for each agent
    sensed_a = [[] for _ in range(N)]
    sensed_o = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if dist(z[i], z[j]) < r_sense + r_agent:
                sensed_a[i].append(j)
                sensed_a[j].append(i)
        for l in range(M):
            if dist(z[i], obs[l]) < r_sense + r_obs:
                sensed_o[i].append(l)
    
    return sensed_a, sensed_o

def qp_controller(z, obs, agent_mdl, obs_mdl, d, zeta_a, eta_a, zeta_o, eta_o):
    '''Inputs:
	   z: Current agents' state
	   Output to be computed: 	
	   u: control velocity
    '''
    N = len(z)
    sensed_a, sensed_o = sensing_range(z, obs)

    # return None if agent at the goal y_end?

    u = []
    slack = []
    udim = 2
    for i in range(N):
        ui = cp.Variable(udim)
        di = cp.Variable(1)
        objective = cp.Minimize(cp.norm2(ui) + di ** 2)
        constraints = [clf_c(z, i, d, ui, epsilon, di)]
        for j in sensed_a[i]:
            constraints.append(cbf_agents_ML(z, i, j, ui, zeta_a, eta_a, agent_mdl))
        for l in sensed_o[i]:
            constraints.append(cbf_obstacle_ML(z, obs, i, l, ui, zeta_o, eta_o, agent_mdl, obs_mdl))
        problem = cp.Problem(objective, constraints)

        problem.solve()
        if problem.status != 'infeasible':
            slack.append(di.value)
            u.append(ui.value)
        else:
            u.append(None)
            slack.append(None)

    return u

def u_ref(p, d, obs):
    sensed_a, sensed_o = sensing_range(z, obs)


def soft_positive_linear(x):
    return np.log(1 + np.exp(x))

def cbf_agent_cfm(z, i, j, ui, lmbd, predictor, agent_mdl):
    return predictor(z, i, j, ui) >= - soft_positive_linear(lmbd[i]) * h_agents(z, i, j, agent_mdl)

def cbf_obstacle_cfm(z, obs, i, l, ui, agent_mdl, obs_mdl):
    return

def conformal_controller(z, obs, agent_mdl, obs_mdl, d, u_ref, lmbd, predictor):
    '''Inputs:
	   z: Current agents' state
	   Output to be computed: 	
	   u: control velocity
    '''
    N = len(z)
    sensed_a, sensed_o = sensing_range(z, obs)

    u = []
    udim = 2
    for i in range(N):
        ui = cp.Variable(udim)
        objective = cp.Minimize(cp.norm2(ui - u_ref[i]))
        constraints = [clf_c(z, i, d, ui, epsilon, 0)]
        for j in sensed_a[i]:
            constraints.append(cbf_agent_cfm(z, i, j, ui, lmbd, predictor, agent_mdl))
        for l in sensed_o[i]:
            constraints.append(cbf_obstacle_cfm(z, obs, i, l, ui, agent_mdl, obs_mdl))
        problem = cp.Problem(objective, constraints)

        problem.solve()
        if problem.status != 'infeasible':
            u.append(ui.value)
        else:
            u.append(None)

    return u



def path_control(z0, u0, t, d, obs, agent_mdl, obs_mdl):
    '''
        z0,u0: initial state, velocity
        t: time sequence for simulation
        cntr: controller to be used
    '''
    nb_steps = len(t)
    N = len(z0)
    r_agent = agent_mdl['r']

    '''These vectors will store the state variables of the vehicle'''
    x = np.array([])
    y = np.array([])
    vx = np.array([])
    vy = np.array([])
    ''' Initialization of x, y, theta, xf, yf'''
    x = np.append(x, [z0[i][0] for i in range(N)])
    y = np.append(y, [z0[i][1] for i in range(N)])
    vx = np.append(x, [z0[i][2] for i in range(N)])
    vy = np.append(y, [z0[i][3] for i in range(N)])

    controller = qp_controller

    """ setting up the figure """
    plt.figure()
    ax = plt.gca()

    """This loop solves the ODE for each pair of time points
    with fixed controller input"""
    for s in range(1, nb_steps):
        # The next time interval to compute the solution over
        tspan = [t[s - 1], t[s]]

        # next controller input
        delta = controller(z0, obs, agent_mdl, obs_mdl)
        if None in delta:
            print("Can't continue loop")
            break
        else:
            u0 = delta

        # solve ODE for next time interval with input u0 from new initial set z0
        z = odeint(agents_model, z0, tspan, args=(u0, 1))
        # store solution (x,y) for plotting
        x = np.append(x, [z[1][i][0] for i in range(N)])
        y = np.append(y, [z[1][i][1] for i in range(N)])
        vx = np.append(x, [z[1][i][2] for i in range(N)])
        vy = np.append(y, [z[1][i][3] for i in range(N)])
    
        # next initial conditions
        z0 = z[1]

    '''Plot initial positions and obstacles'''
    for i in range(N):
        start_c = plt.Circle((x[i], y[i]), r_agent, fill=False) #to change
        ax.add_patch(start_c)
    for l in range(M):
        obs_c = plt.Circle((obs[l][0], obs[l][1]), obs_mdl['r'], fill=True)
        ax.add_patch(obs_c)

    '''Plot the trajectories of the agents'''
    '''for i in range(N):
        x_i = [x[s][i] for s in range(nb_steps)]
        y_i = [y[s][i] for s in range(nb_steps)]
        plt.plot(x_i, y_i, '--', color='0.5', linewidth=2)'''

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
N = 4
x_lim = sim_mdl['x_lim']
y_start = sim_mdl['y_start']
y_end = sim_mdl['y_end']
x_start = [2 * x_lim * i / (N+1) - x_lim for i in range(1, N+1)]

z0 = [[x_start[i], y_start, 0, 0] for i in range(N)]
u0 = [[0, 0] for _ in range(N)]

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

path_control(z0, u0, t, d, obs, agent_mdl, obs_mdl)
