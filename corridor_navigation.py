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
sim_mdl = {'max_t': 500, # ~1400 to reach goal
           'dt': .05,
           'x_lim': 2,
           'y_start': 3,
           'y_end': -2,
           'epsilon': .3, # very sensitive
           'err_rate': .05,
           'learn_rate': .05,
           }

'''Agent parameters: 
    - radius,
    - sensing range,
    - maximal velocity (per sec)'''
agent_mdl = {'r': .15,
            'sense': 2, # default: 2
            'max_v': .5}

max_dist_per_ts = np.sqrt(2) * agent_mdl['max_v'] * sim_mdl['dt']

bloated_agent_mdl = {'r': agent_mdl['r'] + max_dist_per_ts,
                     'sense': 2, # default: 2
                    'max_v': .5}

'''Obstacle parameters: 
    - radius'''
obs_mdl = {'r': .5}


def agents_model(z, t, u):
    '''
    Dynamics of the system
    Inputs:
        - z: concatenated state of all agents (unused)
        - t: time (unused)
        - u: concatenated control of all agents
    '''
    return u

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
    return dist2(z[2*i:2*i+2], d[i])

def clf_lhs(z, i, d, ui, epsilon, delta_i):
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
    vec_xidi = [z[2*i+0] - d[i][0], z[2*i+1] - d[i][1]]
    return 2 * np.matmul(vec_xidi, ui) + epsilon * lyapunov_f(z, i, d) + delta_i

def h_agents(z, i, j, agent_mdl_i, agent_mdl_j):
    '''
    Control Barrier Function for a pair of agents
    Inputs:
        - z: concatenated state of all agents
        - i: index of agent
        - j: index of agent
        - agent_mdl: agent parameters
    '''
    return dist2(z[2*i:2*i+2], z[2*j:2*j+2]) - (agent_mdl_i['r'] + agent_mdl_j['r']) ** 2

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
    return dist2(z[2*i:2*i+2], obs[l]) - (agent_mdl['r'] + obs_mdl['r']) ** 2

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
        cy = z[2*i+1]
        if dir == 'W':
            cx = walls[l][1] + agent_mdl['r']
            return np.sign(z[2*i] - cx) * dist2(z[2*i:2*i+2], (cx, cy))
        else:
            cx = walls[l][1] - agent_mdl['r']
            return np.sign(cx - z[2*i]) * dist2(z[2*i:2*i+2], (cx, cy))
    else:
        cx = z[2*i]
        if dir == 'S':
            cy = walls[l][1] + agent_mdl['r']
            return np.sign(z[2*i+1] - cy) * dist2(z[2*i:2*i+2], (cx, cy))
        else:
            cy = walls[l][1] - agent_mdl['r']
            return np.sign(cy - z[2*i+1]) * dist2(z[2*i:2*i+2], (cx, cy))
    

def cbf_agents(z, i, j, ui, vj, zeta_a, eta_a, agent_mdl_i, agent_mdl_j):
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
    vec_ij = [z[2*i] - z[2*j], z[2*i+1] - z[2*j+1]]
    vec_uivj = [ui[0] - vj[0], ui[1] - vj[1]]
    h = h_agents(z, i, j, agent_mdl_i, agent_mdl_j)
    return 2 * np.matmul(vec_ij, vec_uivj) + zeta_a * (np.sign(h) * np.abs(h) ** eta_a) >= 0

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
    vec_il = [z[2*i] - obs[l][0], z[2*i+1] - obs[l][1]]
    h = h_obstacle(z, obs, i, l, agent_mdl, obs_mdl)
    return 2 * (vec_il[0] * ui[0] + vec_il[1] * ui[1]) + zeta_o * (np.sign(h) * np.abs(h) ** eta_o) >= 0

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
        cy = z[2*i+1]
        if dir == 'W':
            cx = walls[l][1] + agent_mdl['r']
        else:
            cx = walls[l][1] - agent_mdl['r']
    else:
        cx = z[2*i]
        if dir == 'S':
            cy = walls[l][1] + agent_mdl['r']
        else:
            cy = walls[l][1] - agent_mdl['r']

    vec_il = [z[2*i] - cx, z[2*i+1] - cy]
    h = h_wall(z, walls, i, l, agent_mdl)
    return 2 * np.matmul(vec_il, ui) + zeta_o * (np.sign(h) * np.abs(h) ** eta_o) >= 0

def edge_sensing_range(z, obs, walls, i, agent_mdl):
    '''
    List of agents and obstacles in the sensing range of each agent
    Inputs:
        - z: concatenated state of all agents
        - obs: list of obstacles
        - i: index of agent
        - agent_mdl: agent parameters
    '''
    N = len(z) // 2
    r_agent = agent_mdl['r']
    r_sense = agent_mdl['sense']
    M = len(obs)
    r_obs = obs_mdl['r']
    L = len(walls)

    sensed_a = []
    sensed_o = []
    sensed_w = []
    
    for j in range(N):
        if j != i and dist(z[2*i:2*i+2], z[2*j:2*j+2]) < r_sense + r_agent:
            sensed_a.append(j)
    for l in range(M):
        if dist(z[2*i:2*i+2], obs[l]) < r_sense + r_obs:
            sensed_o.append(l)
    for k in range(L):
        dir = walls[k][0]
        if dir == 'W' or dir == 'E':
            cy = z[2*i+1]
            if dir == 'W':
                cx = walls[l][1] + agent_mdl['r']
            else:
                cx = walls[l][1] - agent_mdl['r']
        else:
            cx = z[2*i]
            if dir == 'S':
                cy = walls[l][1] + agent_mdl['r']
            else:
                cy = walls[l][1] - agent_mdl['r']
        if dist(z[2*i:2*i+2], [cx, cy]) < r_sense:
            sensed_w.append(k)

    return sensed_a, sensed_o, sensed_w

def speed_constraints(ui, max_velocity):
    '''
    Constraints on the control variable based on the current velocity of an agent
    Inputs:
        - ui: cvxpy variable for control of agent i
        - max_velocity: upper bound on the absolute velocity along each axis
    '''
    return [- max_velocity <= ui[0], ui[0] <= max_velocity, - max_velocity <= ui[1], ui[1] <= max_velocity]


def conservative_control(z, obs, walls, i, agent_mdl, obs_mdl, d, lmbd, zeta_a, eta_a, zeta_o, eta_o, xi, opt=None):
    '''
    Use cvxpy to solve for the control and slack respecting the CLF-CBF-speed constraints (bloating other agents)
    Inputs:
        - z: concatenated state of all agents
        - obs: list of obstacles
        - walls: list of walls
        - i: index of agent
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

    sensed_a, sensed_o, sensed_w = edge_sensing_range(z, obs, walls, i, agent_mdl)

    u = [None, None]
    slack = None
    udim = 2
    epsilon = sim_mdl['epsilon']

    if dist2(z[2*i:2*i+2], d[i]) < (agent_mdl['r'] ** 2):
        u = [0,0]
        slack = 0
    else:

        ui = cp.Variable(udim)
        di = cp.Variable(1)
        
        objective = cp.Minimize(cp.norm2(ui) + xi(di) ** 2)
        constraints = [clf_lhs(z, i, d, ui, epsilon, di) <= 0]
        
        z_a_i, e_a_i, z_o_i, e_o_i = zeta_a(lmbd[i]), eta_a(lmbd[i]), zeta_o(lmbd[i]), eta_o(lmbd[i])
        for j in sensed_a:
            uj = [0, 0]
            constraints.append(cbf_agents(z, i, j, ui, uj, z_a_i, e_a_i, agent_mdl, bloated_agent_mdl))
            
        for l in sensed_o:
            constraints.append(cbf_obstacle(z, obs, i, l, ui, z_o_i, e_o_i, agent_mdl, obs_mdl))
        for k in sensed_w:
            constraints.append(cbf_wall(z, walls, i, k, ui, z_o_i, e_o_i, agent_mdl))
        for s in speed_constraints(ui, agent_mdl['max_v']):
            constraints.append(s)
        problem = cp.Problem(objective, constraints)

        problem.solve()
        
        if problem.status != 'infeasible' and problem.status != 'infeasible_inaccurate':
            slack = di.value
            u = ui.value
        else:
            print(problem.status)

    return u, slack


def conformal_control(z, obs, walls, i, agent_mdl, obs_mdl, d, lmbd, zeta_a, eta_a, zeta_o, eta_o, xi, opt=None):
    '''
    Use cvxpy to solve for the control and slack respecting the CLF-CBF-speed constraints (learning behaviour)
    Inputs:
        - z: concatenated state of all agents
        - obs: list of obstacles
        - walls: list of walls
        - i: index of agent
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
    
    prdct, agent_type, tspan = opt
    sensed_a, sensed_o, sensed_w = edge_sensing_range(z, obs, i, agent_mdl)

    u = [None, None]
    slack = None
    udim = 2
    epsilon = sim_mdl['epsilon']
    dt = sim_mdl['dt']


    if dist2(z[2*i:2*i+2], d[i]) < (agent_mdl['r'] ** 2):
        u = [0,0]
        slack = 0
    else:
        # only trajectories in sensed_a
        trajectories = prdct(z, obs, walls, i, lmbd, agent_type, tspan) # distinguish ego agent i?
        next_state = trajectories[0]

        ui = cp.Variable(udim)
        di = cp.Variable(1)
        
        objective = cp.Minimize(cp.norm2(ui) + xi(di) ** 2)
        constraints = [clf_lhs(z, i, d, ui, epsilon, di) <= 0]
        
        z_a_i, e_a_i, z_o_i, e_o_i = zeta_a(lmbd[i]), eta_a(lmbd[i]), zeta_o(lmbd[i]), eta_o(lmbd[i])
        for j in sensed_a:
            uj = (next_state[2*j:2*j+2] - z[2*j:2*j+2]) / dt
            constraints.append(cbf_agents(z, i, j, ui, uj, z_a_i, e_a_i, agent_mdl, agent_mdl))
            
        for l in sensed_o:
            constraints.append(cbf_obstacle(z, obs, i, l, ui, z_o_i, e_o_i, agent_mdl, obs_mdl))
        # introduce sensed walls
        for k in sensed_w:
            constraints.append(cbf_wall(z, walls, i, k, ui, z_o_i, e_o_i, agent_mdl))
        for s in speed_constraints(ui, agent_mdl['max_v']):
            constraints.append(s)
        problem = cp.Problem(objective, constraints)

        problem.solve()
        
        if problem.status != 'infeasible' and problem.status != 'infeasible_inaccurate':
            slack = di.value
            u = ui.value
        else:
            print(problem.status)

    return u, slack


def agressive_control(z, obs, walls, i, agent_mdl, obs_mdl, d, lmbd, zeta_a, eta_a, zeta_o, eta_o, xi, opt=None):
    '''
    Use cvxpy to solve for the control and slack respecting the CLF-CBF-speed constraints (ignoring others' movement)
    Inputs:
        - z: concatenated state of all agents
        - obs: list of obstacles
        - walls: list of walls
        - i: index of agent
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
    
    sensed_a, sensed_o, sensed_w = edge_sensing_range(z, obs, i, agent_mdl)

    u = [None, None]
    slack = None
    udim = 2
    epsilon = sim_mdl['epsilon']

    if dist2(z[2*i:2*i+2], d[i]) < (agent_mdl['r'] ** 2):
        u = [0,0]
        slack = 0
    else:
        ui = cp.Variable(udim)
        di = cp.Variable(1)
        
        objective = cp.Minimize(cp.norm2(ui) + xi(di) ** 2)
        constraints = [clf_lhs(z, i, d, ui, epsilon, di) <= 0]
        
        z_a_i, e_a_i, z_o_i, e_o_i = zeta_a(lmbd[i]), eta_a(lmbd[i]), zeta_o(lmbd[i]), eta_o(lmbd[i])
        for j in sensed_a:
            uj = [0,0]
            constraints.append(cbf_agents(z, i, j, ui, uj, z_a_i, e_a_i, agent_mdl, agent_mdl))
            
        for l in sensed_o:
            constraints.append(cbf_obstacle(z, obs, i, l, ui, z_o_i, e_o_i, agent_mdl, obs_mdl))
        for k in sensed_w:
            constraints.append(cbf_wall(z, walls, i, k, ui, z_o_i, e_o_i, agent_mdl))
        for s in speed_constraints(ui, agent_mdl['max_v']):
            constraints.append(s)
        problem = cp.Problem(objective, constraints)

        problem.solve()
        
        if problem.status != 'infeasible' and problem.status != 'infeasible_inaccurate':
            slack = di.value
            u = ui.value
        else:
            print(problem.status)

    return u, slack

def very_agressive_control(z, obs, walls, i, agent_mdl, obs_mdl, d, lmbd, zeta_a, eta_a, zeta_o, eta_o, xi, opt=None):
    '''
    Use cvxpy to solve for the control and slack respecting the CLF-CBF-speed constraints (ignoring other agents)
    Inputs:
        - z: concatenated state of all agents
        - obs: list of obstacles
        - walls: list of walls
        - i: index of agent
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
    
    _, sensed_o, sensed_w = edge_sensing_range(z, obs, i, agent_mdl)

    u = [None, None]
    slack = None
    udim = 2
    epsilon = sim_mdl['epsilon']

    if dist2(z[2*i:2*i+2], d[i]) < (agent_mdl['r'] ** 2):
        u = [0,0]
        slack = 0
    else:
        ui = cp.Variable(udim)
        di = cp.Variable(1)
        
        objective = cp.Minimize(cp.norm2(ui) + xi(di) ** 2)
        constraints = [clf_lhs(z, i, d, ui, epsilon, di) <= 0]
        
        z_o_i, e_o_i = zeta_o(lmbd[i]), eta_o(lmbd[i])
        for l in sensed_o:
            constraints.append(cbf_obstacle(z, obs, i, l, ui, z_o_i, e_o_i, agent_mdl, obs_mdl))
        for k in sensed_w:
            constraints.append(cbf_wall(z, walls, i, k, ui, z_o_i, e_o_i, agent_mdl))
        for s in speed_constraints(ui, agent_mdl['max_v']):
            constraints.append(s)
        problem = cp.Problem(objective, constraints)

        problem.solve()
        
        if problem.status != 'infeasible' and problem.status != 'infeasible_inaccurate':
            slack = di.value
            u = ui.value
        else:
            print(problem.status)

    return u, slack

def soft_positive_linear(x):
    '''
    Function used to smoothly project real to positive values
    '''
    return np.log(1 + np.exp(x))

def next_lmbd(lmbd, err, sim_mdl):
    '''
    Update of conformal parameter based on the binary error 
    Inputs:
        - lmbd: conformal parameters
        - err: error bolean
    '''    
    return lmbd + sim_mdl['learn_rate'] * (sim_mdl['err_rate'] - err)

def auto_predictor(z, obs, walls, i, lmbd, agent_type, tspan):
    '''
    Computes the trajectories of the agents from a learning agent's point of view
    Inputs:
        - z: concatenated state of all agents
        - obs: list of obstacles
        - walls: list of walls
        - i: index of the learning agent
        - lmbd: list of conformal parameters
        - agent_type: list of the agents' type
    '''

    next_u = []
    for i in range(N):
        if agent_type[i] != 1:
            next_u_i, _ = controllers[agent_type[i]](z, obs, walls, i, agent_mdl, obs_mdl, d, lmbd, zeta_a, eta_a, zeta_o, eta_o, xi)
            
            next_u.append(next_u_i[0])
            next_u.append(next_u_i[1])
        else:
            next_u.append(0)
            next_u.append(0)

    # solve ODE for next time interval with input u0 from new initial set z0
    z = odeint(agents_model, z, tspan, args=(next_u,))
    return [z[1]]

def path_control(z0, u0, lmbd, controllers, agent_type, t, d, obs, walls, agent_mdl, obs_mdl, zeta_a, eta_a, zeta_o, eta_o, xi, experiment_name):
    '''
    Run the control simulation, plot the trajectories and relevant data
    Inputs:
        - z0: concatenated state of all agents
        - u0: concatenated control of all agents
        - lmbd: list of conformal parameters
        - controllers
        - agent_type: list of the agents' type
        - t: array of time points
        - d: list of agent goal positions
        - obs: list of obstacles
        - walls: list of walls
        - agent_mdl: agent parameters
        - obs_mdl: obsctacle parameters
        - zeta_a: conformal factor function for agents
        - eta_a: conformal exponential function for agents
        - zeta_o: conformal factor function for obstacles
        - eta_o: conformal exponential function for obstacles
        - xi: weight function applied to slack
    '''
    nb_steps = len(t)
    N = len(z0) // 2 
    r_agent = agent_mdl['r']
    M = len(obs)

    # These vectors will store the state variables of the vehicle
    x = np.array([])
    y = np.array([])
    ux = np.array([])
    uy = np.array([])
    h_obs = np.array([])
    clf = np.array([])
    delta = np.array([0])
    lambda_ego = np.array([])

    # Initialization of x, y
    x = np.append(x, [z0[2*i+0] for i in range(N)])
    y = np.append(y, [z0[2*i+1] for i in range(N)])
    h_obs = np.append(h_obs, [h_obstacle(z0, obs, 0, l, agent_mdl, obs_mdl) for l in range(M)])
    clf = np.append(clf, [clf_lhs(z0, i, d, u0[2*i:2*i+2], sim_mdl['epsilon'], 0) for i in range(N)])
    

    # setting up the figure
    main_ax = plt.gca()
    name = experiment_name + '| dt:' + str(sim_mdl['dt']) + ' | err_rate:' + str(sim_mdl['err_rate']) + ' | learn_rate:' + str(sim_mdl['learn_rate'])
    plt.title(name)
    _, axs = plt.subplots(2, 2)
    #plt.figure()
    

    # This loop solves the ODE for each pair of time points with fixed controller input
    last_step = nb_steps - 1
    collision = False
    for s in range(1, nb_steps):
        # The next time interval to compute the solution over
        tspan = [t[s - 1], t[s]]

        # next controller input
        next_u = []
        next_delta = []
        for i in range(N):
            next_u_i, next_delta_i = controllers[agent_type[i]](z0, obs, walls, i, agent_mdl, obs_mdl, d, lmbd, zeta_a, eta_a, zeta_o, eta_o, xi, opt=(auto_predictor, agent_type, tspan))
            
            if next_delta_i == None:
                print(f"Can't continue loop because of agent {i}")
                u0 = None
                last_step = s - 1
                break
            else:
                next_u.append(next_u_i[0])
                next_u.append(next_u_i[1])
                next_delta.append(next_delta_i)

        if u0 is None:
            break

        ux = np.append(ux, [next_u[2*i+0] for i in range(N)])
        uy = np.append(uy, [next_u[2*i+1] for i in range(N)])
        u0 = next_u

        # solve ODE for next time interval with input u0 from new initial set z0
        z = odeint(agents_model, z0, tspan, args=(u0,))
        # store solution (x,y) for plotting
        x = np.append(x, [z[1][2*i+0] for i in range(N)])
        y = np.append(y, [z[1][2*i+1] for i in range(N)])

        # CBFs for conformal agent
        h_obs = np.append(h_obs, [h_obstacle(z[1], obs, 1, l, agent_mdl, obs_mdl) for l in range(M)])

        clf = np.append(clf, [clf_lhs(z[1], i, d, u0[2*i:2*i+2], sim_mdl['epsilon'], next_delta[i]) for i in range(N)])

        delta = np.append(delta, next_delta)

        lambda_ego = np.append(lmbd[N // 2], lambda_ego)
    
        # next conformal parameters
        err = np.zeros(N)
        for i in range(N):
            if agent_type[i] == 1: # conformal agent
                sensed_a, _, _ = edge_sensing_range(z0, obs, i, agent_mdl)
                for j in sensed_a:
                    if dist(z[1][2*j:2*j+2], z[1][2*i:2*i+2]) < 2 * r_agent:
                        collision = True
                        unsafe_pair = (i, j)
                    z_a_i, e_a_i = zeta_a(lmbd[i]), eta_a(lmbd[i])
                    if not cbf_agents(z0, i, j, u0[2*i:2*i+2], u0[2*j:2*j+2], z_a_i, e_a_i, agent_mdl, agent_mdl):
                        err[i] = 1
                        break
                lmbd[i] = next_lmbd(lmbd[i], err[i], sim_mdl)
                        
        # next initial conditions
        z0 = z[1]

        if collision:
            pass
            # last_step = s - 1
            # break
        
    

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
        
        #u_i = [dist([ux[N*s+i],uy[N*s+i]],[0,0]) for s in range(last_step)]
        u_x = [ux[N*s+i] for s in range(last_step)]
        u_y = [uy[N*s+i] for s in range(last_step)]
        axs[0,0].plot(t, u_x)
        axs[0,0].plot(t, u_y)
        axs[0,0].title.set_text('U of agents')
    plt.colorbar(lc, label='Time steps')

    # Plot the CBF constraints for conformal agent (idx: N // 2)
    for l in range(M):
        h_l = [h_obs[M*s+l] for s in range(last_step)]
        axs[1,0].plot(t, h_l)
        axs[1,0].title.set_text('Obs CBFs of A_1')
    
    if False:
        for i in range(N):
            clf_i = [clf[N*s+i] for s in range(last_step)]
            axs[0,1].plot(t, clf_i)
            axs[0,1].title.set_text('CLF of A_1')
        for i in range(N):
            delta_i = [delta[N*s+i] for s in range(last_step)]
            axs[1,1].plot(t, delta_i)
            axs[1,1].title.set_text('Delta of A_1')

    # lambda, eta, zeta
    axs[1,1].plot(t, lambda_ego)
    axs[1,1].title.set_text('Lambda of conformal agent')
    



    # Plot goal positions
    for i in range(N):
        corner = [d[i][0] - r_agent, d[i][1] - r_agent]
        side = 2 * r_agent
        main_ax.add_patch(Rectangle(corner, side, side, linewidth=1,
                               edgecolor='r', facecolor='none'))

    # plot parameters
    main_ax.legend(loc='lower right', fontsize='x-large')
    main_ax.set_xlim([-2, 2])
    plt.xticks(fontsize=9)
    main_ax.set_ylim([-4, 4])
    plt.yticks(fontsize=9)

    main_ax.set_aspect('equal', adjustable='box')

    print("Last setp:", s)
    if collision:
        i, j = unsafe_pair
        print(f"Collision between agents {i} and {j}.")
    
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
N = 5
x_lim = sim_mdl['x_lim']
y_start = sim_mdl['y_start']
y_end = sim_mdl['y_end']
x_start = [2 * x_lim * i / (N+1) - x_lim for i in range(1, N+1)]

z0 = [x for i in range(N) for x in [x_start[i], y_start]]
u0 = [0 for _ in range(2 * N)]

# strategies: conservative (0) || learning (1) || agressive (2) || very_agressive (2)
start_lmbds = [- np.log(np.exp(2) - 1), - np.log(np.exp(1) - 1), 0, - np.log(np.exp(.25) - 1)]
ego_idx = N // 2

# experiments
all_conservative = np.zeros(N, dtype=int)

all_agressive = np.full(N, 2, dtype=int)

adapt_vs_all_C = np.zeros(N, dtype=int)
adapt_vs_all_C[ego_idx] = 1

adapt_vs_all_A = np.full(N, 2, dtype=int)
adapt_vs_all_A[ego_idx] = 1

adapt_vs_mix_CA = np.zeros(N, dtype=int)
for i in range(ego_idx - 1, -1, -2):
    adapt_vs_mix_CA[i] = 2
for i in range(ego_idx + 2, N, 2):
    adapt_vs_mix_CA[i] = 2
adapt_vs_mix_CA[ego_idx] = 1

all_conformal = np.ones(N, dtype=int)

adapt_vs_one_vA = np.ones(N, dtype=int)
adapt_vs_one_vA[ego_idx] = 3

controllers = [conservative_control, conformal_control, agressive_control, very_agressive_control]

# Current Experiment
agent_type = adapt_vs_mix_CA
experiment_name = 'adapt_vs_mix_CA'
lmbd0 = [start_lmbds[agent_type[i]] for i in range(N)]


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
    M = 3
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
    obs = diamond
    M = len(obs)

'''Walls'''
walls = [('W', -x_lim), ('E', x_lim), ('S', y_end - 1), ('N', y_start + 1)]

'''Goal generation'''
# directly underneath
d = [[x_start[i], y_end] for i in range(N)]
# two groups
pass
# all together
pass
# cone
pass

zeta_a = lambda x: soft_positive_linear(-x)
eta_a = lambda x: soft_positive_linear(-x)
zeta_o = lambda x: soft_positive_linear(-x)
eta_o = lambda x: soft_positive_linear(-x)
xi = lambda delta: 2 * cp.abs(delta)


path_control(z0, u0, lmbd0, controllers, agent_type, t, d, obs, walls, agent_mdl, obs_mdl, zeta_a, eta_a, zeta_o, eta_o, xi, experiment_name)
