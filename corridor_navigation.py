import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.collections as mcoll

from scipy.integrate import odeint

import cvxpy as cp


'''Similation parameters:
    - maximal time step,
    - time interval between two,
    - x position of the walls,
    - start y position,
    - end y position,
    - factor of Lyapunov function in CLF,
    - error rate for agent CBF adptation,
    - learning rate for conformal parameters'''
sim_mdl = {'max_t': 5000,
           'dt': .005,
           'x_lim': 2,
           'y_start': 3,
           'y_end': -2,
           'epsilon': .2, # very sensitive
           'err_rate': .05,
           'learn_rate': .05,
           }

'''Agent parameters: 
    - radius,
    - sensing range,
    - maximal velocity (per sec)'''
agent_mdl = {'r': .15,
            'sense': 10, # default: 2
            'max_v': .5}

'''Obstacle parameters: 
    - radius'''
obs_mdl = {'r': .5}


def agents_model(z, t, u):
    '''
    Dynamics of the concatenated system (z, u)
    Inputs:
        - z[4*i+0], z[4*i+1]: Position of agent i in stationary coordinates
        - z[4*i+2], z[4*i+3]: Velocity of agent i in stationary coordinates
        - t: time (unused)
        - u[i][0], u[i][1]: input velocity of agent i
    '''
    N = len(z) // 4

    dzdt = []
    for i in range(N):
        # change in position
        vx_i = z[4*i+2]
        vy_i = z[4*i+3]
        '''if abs(vx_i) > max_v:
            vx_i = np.sign(vx_i) * max_v
        if abs(vy_i) > max_v:
            vy_i = np.sign(vy_i) * max_v'''
        dzdt.append(vx_i)
        dzdt.append(vy_i)

        # change in velocity
        ux_i = u[2*i+0]
        uy_i = u[2*i+1]
        dzdt.append(ux_i)
        dzdt.append(uy_i)

    return dzdt

def dist2(a1, a2):
    '''
    Square of the euclidian distance for 2D vectors
    Inputs:
        - a1: first vector
        - a2: second vector
    '''
    return (a1[0] - a2[0]) ** 2 + (a1[1] - a2[1]) ** 2

def dist(a1, a2):
    '''
    Euclidian distance for 2D vectors
    Inputs:
        - a1: first vector
        - a2: second vector
    '''
    return np.sqrt(dist2(a1, a2))

def lyapunov_f(z, i, d):
    '''
    Lyapunov function: squared distance between an agent and its goal
    Inputs:
        - z: concatenated state of all agents
        - i: index of agent
        - d: list of goal positions of all agents
    '''
    return dist2(z[4*i:4*i+2], d[i])

def clf_function(z, i, d, ui, epsilon, delta_i):
    '''
    CLF function for an agent (eq. (15))
    Inputs:
        - z: concatenated state of all agents
        - i: index of agent
        - d: list of goal positions of all agents
        - ui: cvxpy variable for control of agent i
        - epsilon: factor of Lyapunov function in CLF
        - delta_i: cvxpy variable for slack of agent i
    '''
    vec_xidi = [z[4*i+0] - d[i][0], z[4*i+1] - d[i][1]]
    return 2 * np.matmul(vec_xidi, ui) + epsilon * lyapunov_f(z, i, d) + delta_i

def h_agents(z, i, j, agent_mdl):
    '''
    Control Barrier Function for a pair of agents
    Inputs:
        - z: concatenated state of all agents
        - i: index of agent
        - j: index of agent
        - agent_mdl: agent parameters
    '''
    return dist2(z[4*i:4*i+2], z[4*j:4*j+2]) - (2 * agent_mdl['r']) ** 2

def h_obstacle(z, obs, i, l, agent_mdl, obs_mdl):
    '''
    Control Barrier Function for a pair (agent, obstacle)
    Inputs:
        - z: concatenated state of all agents
        - obs: list of obstacles
        - i: index of agent
        - l: index of obstacle
        - agent_mdl: agent parameters
        - obs_mdl: obstacle parameters
    '''
    return dist2(z[4*i:4*i+2], obs[l]) - (agent_mdl['r'] + obs_mdl['r'] + .5) ** 2

def h_wall(z, walls, i, l, agent_mdl):
    '''
    Control Barrier Function for a pair (agent, wall)
    Inputs:
        - z: concatenated state of all agents
        - walls: list of walls
        - i: index of agent
        - l: index of wall
    '''
    dir = walls[l][0]
    if dir == 'W' or dir == 'E':
        cy = z[4*i+1]
        if dir == 'W':
            cx = walls[l][1] + agent_mdl['r']
            return np.sign(z[4*i] - cx) * dist2(z[4*i:4*i+2], (cx, cy))
        else:
            cx = walls[l][1] - agent_mdl['r']
            return np.sign(cx - z[4*i]) * dist2(z[4*i:4*i+2], (cx, cy))
    else:
        cx = z[4*i]
        if dir == 'S':
            cy = walls[l][1] + agent_mdl['r']
            return np.sign(z[4*i+1] - cy) * dist2(z[4*i:4*i+2], (cx, cy))
        else:
            cy = walls[l][1] - agent_mdl['r']
            return np.sign(cy - z[4*i+1]) * dist2(z[4*i:4*i+2], (cx, cy))
    

def cbf_agents(z, i, j, ui, zeta_a, eta_a, agent_mdl):
    '''
    CBF constraint for a pair of agents (eq. (17-1))
    Inputs:
        - z: concatenated state of all agents
        - i: index of agent
        - j: index of agent
        - ui: cvxpy variable for control of agent i
        - zeta_a: conformal factor for agents
        - eta_a: conformal exponential for agents
        - agent_mdl: agent parameters
    '''
    vec_ij = [z[4*i+0] - z[4*j+0], z[4*i+1] - z[4*j+1]]
    vec_uivj = [ui[0] - z[4*j+2], ui[1] - z[4*j+3]]
    h = h_agents(z, i, j, agent_mdl)
    return 2 * np.matmul(vec_ij, vec_uivj) + zeta_a * (h ** eta_a) >= 0

def cbf_obstacle(z, obs, i, l, ui, zeta_o, eta_o, agent_mdl, obs_mdl):
    '''
    CBF constraint for a pair of (agent, obstacle) (eq. (17-1))
    Inputs:
        - z: concatenated state of all agents
        - obs: list of obstacles
        - i: index of agent
        - l: index of obstacle
        - ui: cvxpy variable for control of agent i
        - zeta_o: conformal factor for obstacles
        - eta_o: conformal exponential for obstacles
        - agent_mdl: agent parameters
        - obs_mdl: obstacle parameters
    '''
    vec_il = [z[4*i+0] - obs[l][0], z[4*i+1] - obs[l][1]]
    h = h_obstacle(z, obs, i, l, agent_mdl, obs_mdl)
    return 2 * np.matmul(vec_il, ui) + zeta_o * (h ** eta_o) >= 0

def cbf_wall(z, walls, i, l, ui, zeta_o, eta_o, agent_mdl):
    '''
    CBF constraint for a pair of (agent, wall)
    Inputs:
        - z: concatenated state of all agents
        - walls: list of walls
        - i: index of agent
        - l: index of wall
        - ui: cvxpy variable for control of agent i
        - zeta_o: conformal factor for obstacles
        - eta_o: conformal exponential for obstacles
        - agent_mdl: agent parameters
    '''
    dir = walls[l][0]
    if dir == 'W' or dir == 'E':
        cy = z[4*i+1]
        if dir == 'W':
            cx = walls[l][1] + agent_mdl['r']
        else:
            cx = walls[l][1] - agent_mdl['r']
    else:
        cx = z[4*i]
        if dir == 'S':
            cy = walls[l][1] + agent_mdl['r']
        else:
            cy = walls[l][1] - agent_mdl['r']

    vec_il = [z[4*i+0] - cx, z[4*i+1] - cy]
    h = h_wall(z, walls, i, l, agent_mdl)
    return 2 * np.matmul(vec_il, ui) + zeta_o * (h ** eta_o) >= 0

def edge_sensing_range(z, obs):
    '''
    List of agents and obstacles in the sensing range of each agent
    Inputs:
        - z: concatenated state of all agents
        - obs: list of obstacles
    '''
    N = len(z) // 4
    r_agent = agent_mdl['r']
    r_sense = agent_mdl['sense']
    M = len(obs)
    r_obs = obs_mdl['r']

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

def speed_constraints(z, i, ui, max_velocity, sim_mdl):
    '''
    Constraints on the control variable based on the current velocity of an agent
    Inputs:
        - z: concatenated state of all agents
        - i: index of agent
        - ui: cvxpy variable for control of agent i
        - max_velocity: upper bound on the absolute velocity along each axis
        - sim_mdl: simulation parameters
    '''
    inv_dt = 1 / sim_mdl['dt']
    v_i = z[4*i+2:4*i+4]
    return [- inv_dt * (v_i[0] + max_velocity) <= ui[0], ui[0] <= inv_dt * (max_velocity - v_i[0]),
            - inv_dt * (v_i[1] + max_velocity) <= ui[1], ui[1] <= inv_dt * (max_velocity - v_i[1])]


def qp_controller(z, obs, walls, agent_mdl, obs_mdl, d, lmbd, zeta_a, eta_a, zeta_o, eta_o, xi):
    '''
    Use cvxpy to solve for the control and slack respecting the CLF-CBF-speed constraints
    Inputs:
        - z: concatenated state of all agents
        - obs: list of obstacles
        - walls: list of walls
        - agent_mdl: agent parameters
        - obs_mdl: obsctacle parameters
        - d: list of agent goal positions
        - lmbd: conformal parameter
        - zeta_a: conformal factor function for agents
        - eta_a: conformal exponential function for agents
        - zeta_o: conformal factor function for obstacles
        - eta_o: conformal exponential function for obstacles
        - xi: weight function applied to slack
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
            objective = cp.Minimize(cp.norm2(ui) + xi(di) ** 2)
            constraints = [clf_function(z, i, d, ui, epsilon, di) <= 0]
            for j in sensed_a[i]:
                constraints.append(cbf_agents(z, i, j, ui, zeta_a(lmbd[i]), eta_a(lmbd[i]), agent_mdl))
            for l in sensed_o[i]:
                constraints.append(cbf_obstacle(z, obs, i, l, ui, zeta_o(5), eta_o(1), agent_mdl, obs_mdl))
            for l in range(len(walls)):
                constraints.append(cbf_wall(z, walls, i, l, ui, zeta_o(1), eta_o(1), agent_mdl))
            for s in speed_constraints(z, i, ui, agent_mdl['max_v'], sim_mdl):
                constraints.append(s)
            problem = cp.Problem(objective, constraints)

            problem.solve()
            
            if problem.status != 'infeasible' and problem.status != 'infeasible_inaccurate':
                slack.append(di.value)
                u.append(ui.value[0])
                u.append(ui.value[1])
            else:
                u.append(None)
                u.append(None)
                slack.append(None)

    return u, slack, sensed_a


def soft_positive_linear(x):
    '''
    Function used to smoothly project real to positive values
    '''
    return np.log(1 + np.exp(x))

def next_lmbd(lmbd, err):
    '''
    Update of conformal parameter for all agents
    Inputs:
        - list of conformal parameters
        - list of errors detected
    '''
    eta = sim_mdl['learn_rate']
    eps = sim_mdl['err_rate']
    return [lmbd[i] + eta * (eps - err[i]) for i in range(len(lmbd))]

def path_control(z0, u0, lmbd, t, d, obs, walls, agent_mdl, obs_mdl, zeta_a, eta_a, zeta_o, eta_o, xi):
    '''
    Run the control simulation, plot the trajectories and relevant data
    Inputs:
        - z0: concatenated state of all agents
        - u0: concatenated control of all agents
        - lmbd: list of conformal parameters
        - t: array of time points
        - d: list of agent goal positions
        - obs: list of obstacles
        - agent_mdl: agent parameters
        - obs_mdl: obsctacle parameters
        - zeta_a: conformal factor function for agents
        - eta_a: conformal exponential function for agents
        - zeta_o: conformal factor function for obstacles
        - eta_o: conformal exponential function for obstacles
        - xi: weight function applied to slack
    '''
    nb_steps = len(t)
    N = len(z0) // 4 
    r_agent = agent_mdl['r']
    M = len(obs)

    # These vectors will store the state variables of the vehicle
    x = np.array([])
    y = np.array([])
    vx = np.array([])
    vy = np.array([])

    ux = np.array([])
    uy = np.array([])
    h_obs = np.array([])
    clf = np.array([])
    delta = np.array([0])

    # Initialization of x, y, vx, vy
    x = np.append(x, [z0[4*i+0] for i in range(N)])
    y = np.append(y, [z0[4*i+1] for i in range(N)])
    vx = np.append(vx, [z0[4*i+2] for i in range(N)])
    vy = np.append(vy, [z0[4*i+3] for i in range(N)])

    h_obs = np.append(h_obs, [h_obstacle(z0, obs, 0, l, agent_mdl, obs_mdl) for l in range(M)])
    clf = np.append(clf, [clf_function(z0, i, d, u0[2*i:2*i+2], sim_mdl['epsilon'], 0) for i in range(N)])
    

    # setting up the figure
    main_ax = plt.gca()
    _, axs = plt.subplots(2, 2)
    #plt.figure()
    

    # This loop solves the ODE for each pair of time points with fixed controller input
    last_step = nb_steps - 2
    for s in range(1, nb_steps):
        # The next time interval to compute the solution over
        tspan = [t[s - 1], t[s]]

        # next controller input
        next_u, next_delta, sensed_a = qp_controller(z0, obs, walls, agent_mdl, obs_mdl, d, lmbd, zeta_a, eta_a, zeta_o, eta_o, xi)
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
            ux = np.append(ux, [next_u[2*i+0] for i in range(N)])
            uy = np.append(uy, [next_u[2*i+1] for i in range(N)])
            u0 = next_u

        # solve ODE for next time interval with input u0 from new initial set z0
        z = odeint(agents_model, z0, tspan, args=(u0,))
        # store solution (x,y) for plotting
        x = np.append(x, [z[1][4*i+0] for i in range(N)])
        y = np.append(y, [z[1][4*i+1] for i in range(N)])
        vx = np.append(vx, [z[1][4*i+2] for i in range(N)])
        vy = np.append(vy, [z[1][4*i+3] for i in range(N)])

        h_obs = np.append(h_obs, [h_obstacle(z[1], obs, 0, l, agent_mdl, obs_mdl) for l in range(M)])

        clf = np.append(clf, [clf_function(z[1], i, d, u0[2*i:2*i+2], sim_mdl['epsilon'], next_delta[i]) for i in range(N)])

        delta = np.append(delta, next_delta)
    
        # next initial conditions
        z0 = z[1]
        # next conformal parameters
        err = [0] * N
        for i in range(N):
            for j in sensed_a[i]:
                if not cbf_agents(z0, i, j, u0[2*i:2*i+2], zeta_a(lmbd[i]), eta_a(lmbd[i]), agent_mdl):
                    err[i] = 1
                    break  
        lmbd = next_lmbd(lmbd, err)
    

    # Plot initial positions and obstacles
    for i in range(N):
        start_c = plt.Circle((x[i], y[i]), r_agent, fill=False) #to change
        main_ax.add_patch(start_c)
    for l in range(M):
        obs_c = plt.Circle((obs[l][0], obs[l][1]), obs_mdl['r'], fill=True)
        main_ax.add_patch(obs_c)

    # Plot the trajectories of the agents
    t = np.linspace(0, last_step - 1, last_step)

    for i in range(N):
        x_i = [x[N*s+i] for s in range(last_step)]
        y_i = [y[N*s+i] for s in range(last_step)]
        lc = colorline(x_i, y_i, last_step, main_ax, cmap='winter')
        
        u_i = [dist([ux[N*s+i],uy[N*s+i]],[0,0]) for s in range(last_step)]
        axs[0,0].plot(t, u_i)
        axs[0,0].title.set_text('Norm(U) of agents')
    plt.colorbar(lc, label='Time steps')

    # Plot the CBF constraint for agent 1
    for l in range(M):
        h_l = [h_obs[M*s+l] for s in range(last_step)]
        axs[1,0].plot(t, h_l)
        axs[1,0].title.set_text('Obs CBFs of A_1')
    for i in range(N):
        clf_i = [clf[N*s+i] for s in range(last_step)]
        axs[0,1].plot(t, clf_i)
        axs[0,1].title.set_text('CLF of A_1')
    for i in range(N):
        delta_i = [delta[N*s+i] for s in range(last_step)]
        axs[1,1].plot(t, delta_i)
        axs[1,1].title.set_text('Delta of A_1')
    



    # Plot goal positions
    for i in range(N):
        corner = [d[i][0] - r_agent, d[i][1] - r_agent]
        side = 2 * r_agent
        main_ax.add_patch(Rectangle(corner, side, side, linewidth=1,
                               edgecolor='r', facecolor='none'))

    # plot parameters
    main_ax.legend(loc='lower right', fontsize='x-large')
    name = 'max_t:' + str(sim_mdl['max_t']) + ' | dt:' + str(sim_mdl['dt']) + ' | eps:' + str(sim_mdl['epsilon']) + ' | err_rate:' + str(sim_mdl['err_rate']) + ' | learn_rate:' + str(sim_mdl['learn_rate'])
    #plt.title(name)
    main_ax.set_xlim([-2, 2])
    plt.xticks(fontsize=9)
    main_ax.set_ylim([-4, 4])
    plt.yticks(fontsize=9)

    main_ax.set_aspect('equal', adjustable='box')

    print("Length of path:", len(x))
    print("Last setp:", s)
    
    plt.show()

    
def colorline(x, y, max_t, ax, z=None, cmap='copper', linewidth=3, alpha=1.0):
    
    norm = plt.Normalize(0.0, max_t)

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, max_t, len(x))
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    #ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


''' 
Setup for running the simulation
    - Thorizon: time horizon for simulation
    - t: array of time points
'''
Thorizon = sim_mdl['max_t'] * sim_mdl['dt']
t = np.linspace(0, Thorizon, sim_mdl['max_t'])

'''Initial state and input'''
N = 1
x_lim = sim_mdl['x_lim']
y_start = sim_mdl['y_start']
y_end = sim_mdl['y_end']
x_start = [2 * x_lim * i / (N+1) - x_lim for i in range(1, N+1)]

z0 = [x for i in range(N) for x in [x_start[i], y_start, 0, 0]]

u0 = [0 for _ in range(2 * N)]

'''Obstacle generation'''
r_obs = obs_mdl['r']
r_agent = agent_mdl['r']
y_obs_min = y_end + r_agent + r_obs
y_obs_max = y_start - r_agent - r_obs
x_obs_min = - x_lim + r_obs
x_obs_max = x_lim - r_obs
obs = []

mode = 'manual' # manual || random

if mode == 'random':
    M = 1
    while len(obs) < M:
        x_c = (x_obs_max - x_obs_min) * np.random.random(1) + x_obs_min
        y_c = (y_obs_max - y_obs_min) * np.random.random(1) + y_obs_min
        candidate = [x_c, y_c]
        for o in obs:
            if dist(o, candidate) < 2 * r_obs + 2.5 * r_agent:
                candidate = [None, None]
                break
        if not candidate[0] is None:
            obs.append(candidate)

if mode == 'manual':
    empty = []
    center = [[0, 0]]
    pair = [[1, 0], [-1, 0]]
    trapezoid = [[.67, 1], [-.67, 1], [1.33, -1], [-1.33, -1]]
    diamond = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    obs = center
    M = len(obs)

'''Walls'''
walls = [('W', -x_lim), ('E' ,x_lim), ('S', y_end - 1), ('N', y_start + 1)]

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
zeta_o = lambda x: x
eta_o = lambda x: x
xi = lambda delta: delta

lmbd0 = [1] * N

path_control(z0, u0, lmbd0, t, d, obs, walls, agent_mdl, obs_mdl, zeta_a, eta_a, zeta_o, eta_o, xi)
