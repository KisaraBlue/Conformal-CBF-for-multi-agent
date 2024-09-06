import numpy as np

from . import spline_planner_utils
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
from skimage.transform import resize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import time
import numba
import cv2
from scipy.integrate import odeint

#from rrt import run_rrt
from qpsolvers import solve_ls

class SplinePlanner:
    # Construct SplinePlanner object.
    def __init__(self, num_waypts, horizon,
                 max_linear_vel, max_angular_vel,
                 gmin, gmax, gnums, goal_rad, human_rad,
                 lr, alpha, binary_env_img_path=None):

        self.num_waypts_ = num_waypts
        self.horizon_ = horizon
        self.max_linear_vel_ = max_linear_vel
        self.max_angular_vel_ = max_angular_vel
        self.gmin_ = gmin
        self.gmax_ = gmax
        self.gnums_ = gnums
        self.sd_obs_ = None
        self.sd_goal_ = None
        self.goal_rad = goal_rad
        self.human_rad = human_rad
        self.collide_dist = None
        self.lamda = 0
        self.lr = lr
        self.alpha = alpha
        self.alphat = alpha
        self.prev_state = []

        self.X, self.Y = np.meshgrid(np.linspace(gmin[0], gmax[0], gnums[0]), \
                            np.linspace(gmin[1], gmax[1], gnums[1]))

        self.disc_xy_ = np.column_stack((self.X.flatten(), self.Y.flatten()))

        if binary_env_img_path is not None:
            print("[Spline Planner] creating obstacle signed distance function...")
            # load the binary env img
            binary_env_img = cv2.imread(binary_env_img_path, cv2.IMREAD_GRAYSCALE)

            # get raw signed distance
            sd_obs = spline_planner_utils.sd_binary_image(binary_env_img)

            # shape the signed distance
            obstacle_thresh = -100. # distance "deep" into obstacle after which you always get this cost
            freespace_thresh = 10. # distance "away" from obstacles after which you get no cost

            sd_obs[sd_obs > freespace_thresh] = 0
            sd_obs[sd_obs > 0] = -1./sd_obs[sd_obs > 0] # add a small buffer around obstacles where you get small penalty
            sd_obs[sd_obs < obstacle_thresh] = obstacle_thresh

            self.sd_obs_ = resize(sd_obs, (gnums[0], gnums[1]))
            self.sd_obs_interp_ = NearestNDInterpolator(list(zip( self.X.flatten(), self.Y.flatten()  )), self.sd_obs_.flatten())
            self.sd_obs_interp_ = resize(self.sd_obs_interp_(np.column_stack((self.X.flatten(), self.Y.flatten()))), (binary_env_img.shape[0], binary_env_img.shape[1]))
            print("[Spline Planner] created signed distance function interpolator!")

        self.qh = None
        #self.ax = plt.axes()

    def r_goal(self, traj, goal):
        return -1000*np.linalg.norm(traj - goal[:2, None]) # First two coordinates are position, last two are v/angle

    def r_goal_sd(self, goal): # deprecated
        return -1 * spline_planner_utils.sd_circle(self.X, self.Y, goal[0:2], self.goal_rad)

    # Plans a path from start to goal.
    def plan(self, start, goal, pred_splines, method, worst_residuals=None, conformal_CBF_args=None):

        if method == 'conformal CBF':
            opt_spline, loss, curr_state, vels = plan_CBF(start, goal, pred_splines, np.array(self.gmin_), np.array(self.gmax_), self.horizon_, self.num_waypts_, self.max_linear_vel_, self.max_angular_vel_, self.lamda, lambda x: self.alpha * x, self.alphat, self.prev_state, self.collide_dist, conformal_CBF_args)
        else:
            opt_spline, _, loss = plan_numba(np.array(start), goal, pred_splines, np.array(self.gmin_), np.array(self.gmax_), self.disc_xy_, self.horizon_, self.num_waypts_, self.max_linear_vel_, self.max_angular_vel_, self.human_rad, self.sd_obs_interp_, self.lamda, worst_residuals, self.alphat, method)

        if opt_spline is None:
            loss = 0
            opt_spline = np.stack([
                start[0] * np.ones(self.horizon_),
                start[1] * np.ones(self.horizon_),
                np.zeros(self.horizon_),
                np.zeros(self.horizon_),
                start[2] * np.ones(self.horizon_)
            ])
        if method in ['conformal controller', 'proactive conformal controller']:
            self.lamda += self.lr * (loss + self.human_rad)
            self.lamda = max(self.lamda, 0)
        if method == 'aci':
            self.alphat -= self.lr * (loss - self.alpha)
        # lambda update for conformal CBF
        # self.alphat is epsilon since it's unused for this method
        if method == 'conformal CBF':
            self.prev_state = curr_state
            if not loss is None:
                self.lamda += conformal_CBF_args[2] * self.lr * (self.alphat + loss)

        if method == 'conformal CBF':
            return opt_spline, vels
        return opt_spline

@numba.njit
def sample_columns(arr, num_samples, axis):
    if arr.shape[axis] < num_samples:
        raise ValueError("Number of samples is greater than number of elements in axis " + str(axis))
    if len(arr.shape) != 2:
        raise ValueError("Only works for 2D arrayd at the moment.")


    indices = np.linspace(0, arr.shape[axis]-1, num_samples).astype(np.int_)

    # Get the shape of the output array
    if axis == 0:
        new_shape = ( num_samples, arr.shape[1] )
    elif axis == 1:
        new_shape = ( arr.shape[0], num_samples )

    # Create an empty array to store the sampled data
    sampled_arr = np.empty(new_shape, dtype=np.float64)

    if axis == 0:
        for i in range(num_samples):
            for j in range(arr.shape[1]):
                sampled_arr[i,j] = arr[indices[i], j]

    if axis == 1:
        for i in range(arr.shape[0]):
            for j in range(num_samples):
                sampled_arr[i,j] = arr[i, indices[j]]

    return sampled_arr

def U_att(X_i, X_g, K_att):
    '''Attractive potential field'''
    vec_ig = X_g - X_i[:2]
    return K_att * np.dot(vec_ig, vec_ig) / 2

def U_att_deriv(X_i, X_g, K_att):
    '''Gradient of the attractive potential field'''
    return K_att * (X_i[:2] - X_g)

def v_des(X_i, X_g, K_att, max_v):
    '''Reference speed for the potential field based controller'''
    v_att = - U_att_deriv(X_i, X_g, K_att)
    return min(1, max_v / np.linalg.norm(v_att)) * v_att

def rho_obs(X_i, X_obs, D_obs):
    '''Distance between agent and the unsafe area around an obstacle'''
    return np.linalg.norm(X_i[:2] - X_obs) - D_obs

def U_rep(X_i, X_obs, D_obs, rho_0, K_rep):
    '''Repulsive potential field'''
    rho_X_i = rho_obs(X_i, X_obs, D_obs)
    if rho_X_i > rho_0:
        return 0
    return K_rep * (1/rho_X_i - 1/rho_0) ** 2 / 2

def nabla_U_rep(X_i, X_obs, D_obs, rho_0, K_rep):
    '''Gradient of the repulsive potential field'''
    rho_X_i = rho_obs(X_i, X_obs, D_obs)
    if rho_X_i > rho_0:
        return np.zeros(2)
    return K_rep * (1/rho_X_i - 1/rho_0) * (X_i[:2] - X_obs) / rho_X_i ** 3

# delta is a small constant, chosen to be 1/16 here
def h_potential_field(X_i, X_j, collide_dist, rho_0, K_rep):
    '''Control barrier function for potential field based controller'''
    return 1 / (1 + U_rep(X_i, X_j, collide_dist, rho_0, K_rep)) - 1 / 2

def h_potential_field_deriv(X_i, X_j, collide_dist, rho_0, K_rep):
    '''Gradient of the CBF for potential field based controller'''
    #rho_X_i = rho_obs(X_i, X_j, collide_dist)
    #nabla_h = - (K_rep * (1/rho_X_i - 1/rho_0)) / (rho_X_i * (rho_X_i * (1 + U_rep(X_i, X_j, collide_dist, rho_0, K_rep))) ** 2) * (X_i[:2] - X_j)
    '''rho_X_i = rho_obs(X_i, X_j, 0)
    if rho_X_i < rho_0:
        print("Nabla U_rep: ", nabla_U_rep(X_i, X_j, collide_dist, rho_0, K_rep))
        print("U_rep: ", U_rep(X_i, X_j, collide_dist, rho_0, K_rep))'''
    return - nabla_U_rep(X_i, X_j, collide_dist, rho_0, K_rep) / (1 + U_rep(X_i, X_j, collide_dist, rho_0, K_rep)) ** 2

def agent_avoid(X_i, X_j, collide_dist):
    '''Control barrier function for orientation based controller'''
    vec_ij = X_j - X_i[:2]
    dist_ij = np.linalg.norm(vec_ij)
    return .2 * (dist_ij - collide_dist) * \
        (3 + (np.cos(X_i[2]) * vec_ij[0] + np.sin(X_i[2]) * vec_ij[1]) / dist_ij)

def agent_avoid_deriv(X_i, X_j, collide_dist):
    '''Gradient of the CBF for orientation based controller'''
    vec_ij = X_j - X_i[:2]
    dist_ij = np.linalg.norm(vec_ij)
    dist_ratio = 1 - collide_dist / dist_ij
    cos_over_d = (np.cos(X_i[2]) * vec_ij[0] + np.sin(X_i[2]) * vec_ij[1]) / (dist_ij ** 2)

    dh_dx = dist_ratio * (vec_ij[0] * cos_over_d - np.cos(X_i[2])) - (cos_over_d + (3 * vec_ij[0]) / dist_ij)
    dh_dy = dist_ratio * (vec_ij[1] * cos_over_d - np.sin(X_i[2])) - (cos_over_d + (3 * vec_ij[1]) / dist_ij)
    dh_dth = dist_ratio * (np.cos(X_i[2]) * vec_ij[1] - np.sin(X_i[2]) * vec_ij[0])

    return .2 * np.array([[dh_dx, dh_dy, dh_dth]])

def approx_agent_avoid_deriv(X_i, X_j, collide_dist, u, dt):
    # todo: factor this code, avoid redundant computation
    X_i__dx = X_i + [dt * u[0] * np.cos(X_i[2]), 0, 0]
    X_i__dy = X_i + [0, dt * u[0] * np.sin(X_i[2]), 0]
    X_i__dth = X_i + [0, 0, dt * u[1]]
    curr_h = agent_avoid(X_i, X_j, collide_dist)

    dh_dx = (agent_avoid(X_i__dx, X_j, collide_dist) - curr_h) / dt
    dh_dy = (agent_avoid(X_i__dy, X_j, collide_dist) - curr_h) / dt
    dh_dth = (agent_avoid(X_i__dth, X_j, collide_dist) - curr_h) / dt
    return np.array([[dh_dx, dh_dy, dh_dth]])

def conformal_CBF_constraint(X_i, X_j, pred_xy_j_dot, lmbd, alpha, collide_dist):
    '''RHS of the QP constraint for orientation based controller'''
    # the constant part of q_i is zero because f(X_i)=0 in dynamics
    dh_j = -agent_avoid_deriv(X_i, X_j, collide_dist)
    pred_q_j = np.dot(dh_j[0, :2], pred_xy_j_dot)
    alpha_h = alpha(agent_avoid(X_i, X_j, collide_dist))
    return pred_q_j + alpha_h + lmbd - .1

def conformal_CBF_constraint_DI(X_i, X_j, pred_xy_j_dot, lmbd, alpha, collide_dist, rho_0, K_rep, K_acc):
    '''RHS of the QP constraint for potential field based controller'''
    dh_i = h_potential_field_deriv(X_i, X_j, collide_dist, rho_0, K_rep)
    #q_i_const = np.dot(dh_i, f_dynamics_DI(X_i)[:2] - K_acc * np.matmul(g_dynamics_DI(), X_i[2:]))
    pred_q_j = np.dot(-dh_i, pred_xy_j_dot)
    alpha_h = alpha(h_potential_field(X_i, X_j, collide_dist, rho_0, K_rep))
    return alpha_h + pred_q_j# + lmbd

def approx_conformal_CBF_constraint(X_i, X_j, next_X_j, pred_xy_j_dot, lmbd, alpha, collide_dist, dt):
    # todo: factor this code, avoid redundant computation
    X_j__dx = np.array([next_X_j[0], X_j[1]])
    X_j__dy = np.array([X_j[0], next_X_j[1]])
    curr_h = agent_avoid(X_i, X_j, collide_dist)

    dh_dx = (curr_h - agent_avoid(X_i, X_j__dx, collide_dist)) / dt
    dh_dy = (curr_h - agent_avoid(X_i, X_j__dy, collide_dist)) / dt
    # the constant part of q_i is zero because f(X_i)=0 in dynamics
    dh_j = [dh_dx, dh_dy]
    pred_q_j = np.dot(dh_j, pred_xy_j_dot)
    alpha_h = alpha(agent_avoid(X_i, X_j, collide_dist))
    return pred_q_j + alpha_h + lmbd - .1

def g_dynamics(X_i):
    '''Dynamics for orientation based controller'''
    return np.array([[np.cos(X_i[2]), 0], [np.sin(X_i[2]), 0], [0, 1]])

def f_dynamics_DI(X_i):
    '''Dynamics for potential field based controller'''
    return np.array([X_i[2], X_i[3], 0, 0])

def g_dynamics_DI():
    '''Dynamics for potential field based controller'''
    #return np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    return np.eye(2)

def prediction_loss(score_func, X_i, current_pred, dXdt_pred, current_gt, dXdt_gt, alpha, lamda, collide_dist):
    '''Loss computed for every prediction, not used anymore'''
    N = len(current_pred)
    pred_q = [np.dot(-agent_avoid_deriv(X_i, current_pred[j], collide_dist)[0, :2], dXdt_pred[j]) for j in range(N)]
    gt_q = [np.dot(-agent_avoid_deriv(X_i, current_gt[j], collide_dist)[0, :2], dXdt_gt[j]) for j in range(N)]
    D_alpha_h = [alpha(agent_avoid(X_i, current_pred[j], collide_dist)) - 
                 alpha(agent_avoid(X_i, current_gt[j], collide_dist)) 
                 for j in range(N)]
    return np.max([score_func(D_alpha_h[j] + pred_q[j] - gt_q[j] + lamda) 
                   for j in range(N)])

def control_loss(score_func, u_qp, A_gt, b_gt, QP_score):
    '''Positive loss computed when control used returns a worse score for ground truth then prediction'''
    return score_func(max(0, np.max(np.matmul(A_gt, u_qp) - b_gt) - QP_score))

def extended_control_loss(score_func, u_qp, A_gt, b_gt, QP_score):
    s = score_func(np.max(np.matmul(A_gt, u_qp) - b_gt) - QP_score)
    if s < 0:
        return s
    return s / 1 #change if needed

def conservativeness_loss(score_func, u_qp, u_ref, A_gt, b_gt, A, b):
    '''Negative loss computed when reference control is safe but used control is unsafe'''
    u_ref_is_safe = np.max(np.matmul(A_gt, u_ref) - b_gt) <= 0
    over_conservative_pred = np.max(np.matmul(A, u_ref) - b) > 0
    if over_conservative_pred and u_ref_is_safe:
        s = score_func(min(0, np.max(np.matmul(A, u_qp)) - np.max(np.matmul(A, u_ref))))
    else:
        s = 0
    return s, u_ref_is_safe, over_conservative_pred

def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    if n1 == 0:
        return 0
    n2 = np.linalg.norm(v2)
    if n2 == 0:
        return 0
    v1_u = v1 / n1
    v2_u = v2 / n2
    diff = np.linalg.norm(v2_u - v1_u)
    if diff < 0.001:
        return 0
    if diff > 2 - 0.001:
        return np.pi
    return np.sign(np.linalg.det(np.stack((v1_u, v2_u)))) * \
            np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def sqrt_alpha_func(h):
    return np.sign(h) * np.sqrt(abs(h))
    
def log_score_func(r):
    return np.sign(r) * np.arctan(np.log(1 + abs(r))) / np.pi

def exp_score_func(r):
    if abs(r) > 225.93:
        return np.sign(r) / 2
    return np.sign(r) * np.arctan(np.exp(np.pi * abs(r)) - 1) / np.pi

def arctan_score_func(r):
    return np.arctan(r) / np.pi

def leaky_score_func(r):
    if r < 0:
        return np.arctan(r * 10) / np.pi
    return np.arctan(r / 10) / np.pi

def dict_mid_to_idx(arr, N):
    return {arr[i]: i for i in range(N)}

def inter_mids(d1, d2):
    return np.intersect1d(np.fromiter(d1.keys(), int), np.fromiter(d2.keys(), int), assume_unique=True)

def linear_inequalities_QP(N, current_pos, dXdt, start, lamda, alpha_func, collide_dist):
    '''current_pos and dXdt have the same size N, and their indexing aligns identical agents'''
    A = np.empty((N, 2))
    b = np.empty(N)
    for j in range(N):
        A[j] = np.matmul(-agent_avoid_deriv(start, current_pos[j], collide_dist), g_dynamics(start))
        b[j] = conformal_CBF_constraint(start, current_pos[j], dXdt[j], lamda, alpha_func, collide_dist)
    return A, b

def linear_inequalities_QP_DI(N, current_pos, dXdt, start, lamda, alpha_func, collide_dist, rho_0, K_rep, K_acc):
    '''current_pos and dXdt have the same size N, and their indexing aligns identical agents'''
    A = np.empty((N, 2))
    b = np.empty(N)
    for j in range(N):
        A[j] = np.matmul(-h_potential_field_deriv(start, current_pos[j], collide_dist, rho_0, K_rep), g_dynamics_DI())
        b[j] = conformal_CBF_constraint_DI(start, current_pos[j], dXdt[j], lamda, alpha_func, collide_dist, rho_0, K_rep, K_acc)
    return A, b

def X_i_deriv(X_i, t, u):
    return np.array([X_i[2], X_i[3], u[0], u[1]])

def ode_solver(X_i_0, u, dt):
    return odeint(X_i_deriv, X_i_0, np.linspace(0, dt, 4), args=(u,))

def approx_linear_ineq_QP(N, current_pos, next_pos, dXdt, start, lamda, alpha_func, collide_dist, u, dt):
    A = np.empty((N, 2))
    b = np.empty(N)
    for j in range(N):
        A[j] = np.matmul(-approx_agent_avoid_deriv(start, current_pos[j], collide_dist, u, dt), g_dynamics(start))
        b[j] = approx_conformal_CBF_constraint(start, current_pos[j], next_pos[j], dXdt[j], lamda, alpha_func, collide_dist, dt)
    return A, b

def reference_spline(ref_spline, start_pos, start_dir, max_linear_vel_, max_angular_vel_, dt, human_init, collide_dist):
    u_ref = None
    ref_length = len(ref_spline[0])
    min_dists_to_ref = [np.linalg.norm(ref_spline[:2, j] - start_pos) for j in range(ref_length)]
    close_ref_idx = np.argmin(min_dists_to_ref)
    close_align_ang = angle_between(start_dir, ang_to_unit_vec(ref_spline[4, close_ref_idx]))
    close_align_w = np.clip(close_align_ang / dt, -max_angular_vel_, max_angular_vel_)
    towards_close_ang = angle_between(start_dir, ref_spline[:2, close_ref_idx] - start_pos)
    towards_close_v = np.clip(min_dists_to_ref[close_ref_idx] / dt, 0, max_linear_vel_) * max(0, np.cos(towards_close_ang))
    towards_close_w = np.clip(towards_close_ang / dt, -max_angular_vel_, max_angular_vel_)
    close_ref_v = ref_spline[2, close_ref_idx] * max(0, np.cos(close_align_ang))
    close_ref_w = ref_spline[3, close_ref_idx] * (1 - towards_close_v / max_linear_vel_)
    sum_close_v = (towards_close_v + close_ref_v)
    sum_close_w = (close_align_w + towards_close_w + close_ref_w)
    for target_ref_idx in range(ref_length - 1, close_ref_idx, -1):
        target_align_ang = angle_between(start_dir, ang_to_unit_vec(ref_spline[4, target_ref_idx]))
        towards_target_ang = angle_between(start_dir, ref_spline[:2, target_ref_idx] - start_pos)
        towards_target_v = np.clip(min_dists_to_ref[target_ref_idx] / dt, 0, max_linear_vel_) * max(0, np.cos(towards_target_ang))
        target_ref_v = ref_spline[2, target_ref_idx] * max(0, np.cos(target_align_ang))
        candidate_v = (sum_close_v + towards_target_v + target_ref_v) / 4
        candidate_pos = start_pos + candidate_v * dt * start_dir
        min_dist_to_obs = np.min([np.linalg.norm(pos_j - candidate_pos) for pos_j in human_init])
        if min_dist_to_obs > collide_dist:
            target_align_w = np.clip(target_align_ang / dt, -max_angular_vel_, max_angular_vel_)
            towards_target_w = np.clip(towards_target_ang / dt, -max_angular_vel_, max_angular_vel_)
            target_ref_w = ref_spline[3, target_ref_idx] * (1 - towards_target_v / max_linear_vel_)
            candidate_w = (sum_close_w + towards_target_w + target_ref_w + target_align_w) / 6
            u_ref = np.array([candidate_v, candidate_w])
            break
    if u_ref is None:
        u_ref = np.zeros(2)
    return u_ref

def reference_rrt(ref_rrt):
    return 0

def ang_to_unit_vec(theta):
    return np.array([np.cos(theta), np.sin(theta)])

def polar_to_cartesian(X_i):
    v = X_i[3] * ang_to_unit_vec(X_i[2])
    return np.array([X_i[0], X_i[1], v[0], v[1]]) 

def cartesian_to_polar(X_i):
    v = X_i[2:]
    return np.array([X_i[0], X_i[1], angle_between([1, 0], v), np.linalg.norm(v)])

def plan_CBF(start, goal, human_data, gmin, gmax, horizon_, num_waypts_, max_linear_vel_, max_angular_vel_, lamda, alpha, _, prev_state, collide_dist, conformal_CBF_args):
    # Prepare data
    ref_object, human_init, tau, loss_type, _, frame_idx, dynamics_type, solve_rate, sub_frame, dist_to_goal = conformal_CBF_args
    curr_next_preds, time_step, N, c_pred_mid_idx, current_gt = human_data
    preds_at_last_tau = curr_next_preds[time_step]
    pred_diff = (1 / solve_rate) * (curr_next_preds[time_step + 1] - preds_at_last_tau)
    current_pred = preds_at_last_tau + sub_frame * pred_diff
    #next_pred = current_pred + pred_diff
    c_gt_mid_idx = dict_mid_to_idx(current_gt[:, 2], len(current_gt))
    current_gt = current_gt[:, :2]
    start_pos = start[:2]
    start_dir = ang_to_unit_vec(start[2])
    dt = horizon_ / (num_waypts_ - 1)
    alpha_func = lambda x: x * .00001
    score_func = arctan_score_func
    sub_dt = dt / solve_rate
    K_acc, K_rep, K_att, rho_0 = 1, 1, 1, 1 + dist_to_goal
    
    '''# Reference control
    if ref_type == 'spline':
        u_ref = reference_spline(ref_object, start_pos, start_dir, max_linear_vel_, max_angular_vel_, dt, human_init, collide_dist)
    else: # rrt by default
        u_ref = reference_rrt(ref_object), 0'''

    # QP problem
    dXdt_pred = np.array(pred_diff) / dt
    #print(f"Max pedestrian speed: {np.max(np.abs(dXdt_pred))}")
    if dynamics_type == 'lin_ang__vel':
        A, b = linear_inequalities_QP(N, current_pred, dXdt_pred, start, lamda, alpha_func, collide_dist)
        lb = np.array([[0], [-max_angular_vel_]])
        ub = np.array([[max_linear_vel_], [max_angular_vel_]])
        u_ref = reference_spline(ref_object, start_pos, start_dir, max_linear_vel_, max_angular_vel_, dt, human_init, collide_dist)
        qp = solve_ls(np.eye(2), u_ref, G=A, h=b, lb=lb, ub=ub, solver="daqp", verbose=True)
    else:
        start = polar_to_cartesian(start)
        A, b = linear_inequalities_QP_DI(N, current_pred, dXdt_pred, start, lamda, alpha_func, 0, rho_0, K_rep, K_acc)

        #A, b = linear_inequalities_QP_DI(N, current_gt, dXdt_pred, start, lamda, alpha_func, collide_dist, rho_0, K_rep, K_acc)
        u_ref = v_des(start, goal[:2], K_att, max_linear_vel_)
        lb = np.array([[-max_linear_vel_], [-max_linear_vel_]])
        ub = np.array([[max_linear_vel_], [max_linear_vel_]])
        qp = solve_ls(np.eye(2), u_ref, G=A, h=b, lb=lb, ub=ub, solver="daqp", verbose=True)
    unsat_qp = qp is None
    if unsat_qp:
        u_qp = np.array([0, 0])
    else:
        u_qp = qp

        '''u_ref = -U_att_deriv(start, goal[:2], 1)
        qp = solve_ls(np.eye(2), u_ref, G=A, h=b, solver="daqp")'''
    
        '''b_minus_Au = b - np.matmul(A, u_qp)
        if np.any([b_minus_Au[j] < -0.001 for j in range(len(b))]) :
            print(A)
            print(b)
            print(np.matmul(A, u_qp))
            print(b - np.matmul(A, u_qp))
            
            print(f"QP problem at {frame_idx} returns negative values (should not happen), qp is {qp} and h_dot_+_alpha_h:\n{b_minus_Au}")
            exit(0)'''

    # Compute next position and loss
    if dynamics_type == 'lin_ang__vel':
        next_X_i = start + np.matmul(g_dynamics(start), u_qp * dt).reshape(3)
        next_pos = np.clip(next_X_i[:2], gmin, gmax)
        next_X_i = np.append(next_pos, [next_X_i[2]])
    else:
        '''u_acc = - K_acc * (start[2:] - u_qp)
        next_traj = ode_solver(start, u_acc, sub_dt)'''
        next_traj = odeint(lambda X, t, v: v, start[:2], np.linspace(0, sub_dt, 4), args=(u_qp,))
        next_X_i = np.append(next_traj[-1], u_qp)


    '''if not unsat_qp:
        dist_to_preds = [np.linalg.norm(pos_j - next_pos) for pos_j in current_pred + pred_diff]
        min_dist = np.min(dist_to_preds)
        agent = np.argmin(dist_to_preds)
        if min_dist < collide_dist:
            print(f"Valid QP movement at {frame_idx} led to collision with predicted next position (should not happen)")
            print(f"QP control: {u_qp}, h_dot_+_alpha_h: {b - np.matmul(A, u_qp)}, violation: {(collide_dist - min_dist) / collide_dist} with agent {agent}")'''

    
    safe_u_ref_count, conservative_pred_count, both_true_but_0_loss_count = 0, 0, 0
    record_CBF = []

    loss = None
    if not sub_frame:
        if not time_step and prev_state:
            if loss_type == 'Predictor':
                loss = -.5 # minimum loss value
                p_preds, p_pred_mid_idx = prev_state[-1]
                prev_state[-1] = [start, current_gt, c_gt_mid_idx, dXdt_pred]
                for t in range(tau):
                    p_start, p_gt, p_gt_mid_idx, p_dXdt_pred = prev_state[t]
                    p_pred = p_preds[t]
                    _, n_gt, n_gt_mid_idx, _ = prev_state[t+1]
                    p_dXdt_gt_mids = inter_mids(p_gt_mid_idx, n_gt_mid_idx)
                    p_dXdt_gt = []
                    for mid in p_dXdt_gt_mids:
                        p_dXdt_gt.append(n_gt[n_gt_mid_idx[mid]] - p_gt[p_gt_mid_idx[mid]])
                    p_dXdt_gt = np.array(p_dXdt_gt) / dt
                    dXdt_gt_mid_idx = dict_mid_to_idx(p_dXdt_gt_mids, len(p_dXdt_gt_mids))
                    loss_mids = inter_mids(dXdt_gt_mid_idx, p_pred_mid_idx)
                    loss_pred = p_pred[[p_pred_mid_idx[mid] for mid in loss_mids]]
                    loss_dXdt_pred = p_dXdt_pred[[p_pred_mid_idx[mid] for mid in loss_mids]]
                    loss_gt = p_gt[[p_gt_mid_idx[mid] for mid in loss_mids]]
                    loss_dXdt_gt = p_dXdt_gt[[dXdt_gt_mid_idx[mid] for mid in loss_mids]]
                    loss = max(loss, prediction_loss(score_func, p_start, loss_pred, loss_dXdt_pred, loss_gt, loss_dXdt_gt, alpha_func, lamda, collide_dist))
            else: # QP loss by default
                qp_gt_loss, ref_lambda_loss = -.5, .5
                prev_state.append([start, current_gt, c_gt_mid_idx, u_qp, u_ref, A, b])
                for t in range(tau):
                    p_start, p_gt, p_gt_mid_idx, p_u_qp, p_u_ref, p_A, p_b = prev_state[t]
                    _, n_gt, n_gt_mid_idx, _, _, _, _ = prev_state[t+1]
                    if True: #np.any(p_u_qp):
                        p_QP_score = np.max(np.matmul(p_A, p_u_qp) - p_b)
                        p_dXdt_gt_mids = inter_mids(p_gt_mid_idx, n_gt_mid_idx)
                        p_dXdt_gt = []
                        for mid in p_dXdt_gt_mids:
                            p_dXdt_gt.append(n_gt[n_gt_mid_idx[mid]] - p_gt[p_gt_mid_idx[mid]])
                        p_dXdt_gt = np.array(p_dXdt_gt) / dt
                        loss_p_gt = p_gt[[p_gt_mid_idx[mid] for mid in p_dXdt_gt_mids]]
                        #loss_n_gt = n_gt[[n_gt_mid_idx[mid] for mid in p_dXdt_gt_mids]]
                        if dynamics_type == 'lin_ang__vel':
                            p_A_gt, p_b_gt = linear_inequalities_QP(len(p_dXdt_gt), loss_p_gt, p_dXdt_gt, p_start, 0, alpha_func, collide_dist)
                        else:
                            p_A_gt, p_b_gt = linear_inequalities_QP_DI(len(p_dXdt_gt), loss_p_gt, p_dXdt_gt, p_start, 0, alpha_func, collide_dist, rho_0, K_rep, K_acc)
                        p_CBF_gt = p_b_gt - np.matmul(p_A_gt, p_u_qp)
                        p_start_pos = p_start[:2]
                        for idx, mid in enumerate(p_dXdt_gt_mids):
                            record_CBF.append((t, mid, p_CBF_gt[idx], np.linalg.norm(p_start_pos - loss_p_gt[idx]) - collide_dist, h_potential_field(p_start, loss_p_gt[idx], collide_dist, rho_0, K_rep), np.linalg.norm(h_potential_field_deriv(start, loss_p_gt[idx], collide_dist, rho_0, K_rep)), np.linalg.norm(p_u_ref), np.linalg.norm(p_u_qp)))
                        qp_gt_loss = max(qp_gt_loss, control_loss(score_func, p_u_qp, p_A_gt, p_b_gt, p_QP_score))
                        p_ref_lambda_loss, u_ref_is_safe, pred_is_over_conservative = conservativeness_loss(score_func, p_u_qp, p_u_ref, p_A_gt, p_b_gt, p_A, p_b)
                        ref_lambda_loss = min(ref_lambda_loss, p_ref_lambda_loss)
                        safe_u_ref_count += 1 if u_ref_is_safe else 0
                        conservative_pred_count += 1 if pred_is_over_conservative else 0
                        both_true_but_0_loss_count += 1 if u_ref_is_safe and pred_is_over_conservative and p_ref_lambda_loss == 0 else 0
                    else:
                        pass
                        #qp_gt_loss = max(qp_gt_loss, 0)
                        #ref_lambda_loss = min(ref_lambda_loss, 0)
                loss = qp_gt_loss if ref_lambda_loss == 0 else ref_lambda_loss
            prev_state = [prev_state[-1]]
        else:
            if loss_type == 'Predictor':
                prev_state.append([start, current_gt, c_gt_mid_idx, dXdt_pred])
                if time_step == tau - 1:
                    prev_state.append([curr_next_preds, c_pred_mid_idx])
            else: # QP loss by default
                prev_state.append([start, current_gt, c_gt_mid_idx, u_qp, u_ref, A, b])


    if dynamics_type == 'lin_ang__vel':
        spline = [[0, next_X_i[0]], [0, next_X_i[1]], [0, u_qp[0]], [0, u_qp[1]], [0, next_X_i[2]]]
        v_ref = u_ref[0]
    else:
        next_X_i = cartesian_to_polar(next_X_i)
        spline = [[0, next_X_i[0]], [0, next_X_i[1]], [0, next_X_i[3]], [0, 0], [0, next_X_i[2]]]
        v_ref = np.linalg.norm(u_ref)
    safety_violations = np.count_nonzero([min(0, np.linalg.norm(start_pos - pos_gt_j) - collide_dist) for pos_gt_j in current_gt])
    #hurt_agents = [k for k in c_gt_mid_idx.keys() if agent_avoid(start, current_gt[c_gt_mid_idx[k]], collide_dist) < 0]
    #count_invis = len([k for k in hurt_agents if not k in c_pred_mid_idx.keys()])
    constraint_violations = 0 if unsat_qp else np.count_nonzero([max(Au_b, 0) for Au_b in np.matmul(A, u_qp) - b])

    return spline, loss, prev_state, (v_ref, np.average(A, axis=0), np.average(b), safety_violations, constraint_violations, lamda, loss, unsat_qp, safe_u_ref_count, conservative_pred_count, both_true_but_0_loss_count, record_CBF)


# Jitted version of the plan function
@numba.njit
def plan_numba(start, goal, pred_splines, gmin, gmax, disc_xy_, horizon_, num_waypts_, max_linear_vel_, max_angular_vel_, human_rad, sd_obs_interp, lamda, worst_residuals, alphat, method):
    opt_reward = -1e15
    opt_spline = None
    curr_spline = None
    opt_loss = None

    if method == 'aci':
        if alphat <= 1/worst_residuals.shape[0]:
            conformal_quantile = np.infty
            return None, None, 0
        elif alphat >= 1:
            conformal_quantile = 0
        else:
            conformal_quantile = np.quantile(worst_residuals[:-1], 1-alphat)
        last_residual = worst_residuals[-1]

    for ti in range(len(disc_xy_)):

        candidate_goal = disc_xy_[ti, :]
        candidate_goal = np.append(candidate_goal, [goal[2], goal[3]])

        arrs = spline(start, candidate_goal, horizon_, num_waypts_)

        # Determine the common shape and create an empty result array
        ## Numba version
        shape = (len(arrs),) + arrs[0].shape
        curr_spline = np.empty(shape, dtype=arrs[0].dtype)
        # Assign each array to the appropriate slice of the result
        for i, arr in enumerate(arrs):
            curr_spline[i] = arr
        ## Reg version
        #curr_spline = np.stack(arrs, axis=0)

        curr_spline[0] = np.clip(curr_spline[0], gmin[0], gmax[0])
        curr_spline[1] = np.clip(curr_spline[1], gmin[1], gmax[1])

        feasible_horizon = compute_dyn_feasible_horizon(curr_spline,
                                                        max_linear_vel_,
                                                        max_angular_vel_,
                                                        horizon_)

        if feasible_horizon <= horizon_:
            traj = curr_spline[:2, :] if curr_spline.shape[1] == pred_splines.shape[2] else sample_columns(curr_spline[:2, :], pred_splines.shape[2], axis=1)
            vel = curr_spline[2, :] if curr_spline.shape[1] == pred_splines.shape[2] else sample_columns(curr_spline[2:3, :], pred_splines.shape[2], axis=1)[0,:]
            if method == 'conformal controller':
                reward, loss = eval_reward_numba_conformal_controller(goal, traj, vel, pred_splines, human_rad, sd_obs_interp, lamda, 'regular')
            if method == 'proactive conformal controller':
                reward, loss = eval_reward_numba_conformal_controller(goal, traj, vel, pred_splines, human_rad, sd_obs_interp, lamda, 'proactive')
            if method == 'aci':
                reward, loss = eval_reward_numba_aci(goal, traj, vel, pred_splines, human_rad, sd_obs_interp, conformal_quantile, last_residual)
            if method == 'conservative':
                reward, loss = eval_reward_numba_conservative(goal, traj, vel, pred_splines, human_rad, sd_obs_interp)

            if reward > opt_reward:
                opt_reward = reward
                opt_spline = curr_spline
                opt_loss = loss

    return opt_spline, opt_reward, opt_loss

# Computes dynamically feasible horizon (given dynamics of car).
@numba.njit
def compute_dyn_feasible_horizon(spline,
                          max_linear_vel,
                          max_angular_vel,
                          final_t):

  # Compute max linear and angular speed.
  plan_max_lin_vel = np.max(spline[2]);
  plan_max_angular_vel = np.max(spline[3]);

  # Compute required horizon to acheive max speed of planned spline.
  feasible_horizon_speed = final_t * plan_max_lin_vel / max_linear_vel

  # Compute required horizon to acheive max angular vel of planned spline.
  feasible_horizon_angular_vel = final_t * plan_max_angular_vel / max_angular_vel

  feasible_horizon = max(feasible_horizon_speed, feasible_horizon_angular_vel)

  return feasible_horizon

# Evaluates the total reward along the trajectory.
@numba.njit
def eval_reward_numba_conformal_controller(goal, traj, vel, pred_splines, human_rad, sd_obs_interp, lamda, strategy_type):
  goal_r = -1.*np.linalg.norm(traj - goal[:2, None])

  obs_r = 0
  xs = traj[0]
  ys = traj[1]
  for i in range(xs.shape[0]):
    obs_r += 1000*sd_obs_interp[int(ys[i])-1, int(xs[i])-1]


  # NOTE: assumes that prediction length and time aligns!
  differences = pred_splines - traj[None, :]
  norms = np.sqrt((differences ** 2).sum(axis=1))
  # Radius of gaussian with std = human_rad/2
  #rad = spline_planner_utils.confidence_interval_half_width_numba(alphat, human_rad/100, 1)
  human_r = 1000*lamda*norms.min()#-10000*lamda*np.any(norms <= human_rad) # Bad if we are within a CI
  #print(goal_r, human_r, obs_r)

  if strategy_type == 'proactive':
    loss = -norms.min()
  elif strategy_type == 'regular':
    loss = -norms[:,0].min()
  else: # CBF
    loss = 0 # sacha_todo

  reward = goal_r + obs_r + human_r

  return reward, loss

# Evaluates the total reward along the trajectory.
@numba.njit
def eval_reward_numba_aci(goal, traj, vel, pred_splines, human_rad, sd_obs_interp, conformal_quantile, last_residual):
  goal_r = -1.*np.linalg.norm(traj - goal[:2, None])

  obs_r = 0
  xs = traj[0]
  ys = traj[1]
  for i in range(xs.shape[0]):
      obs_r += 10000*sd_obs_interp[int(ys[i])-1, int(xs[i])-1]


  # NOTE: assumes that prediction length and time aligns!
  differences = pred_splines - traj[None, :]
  norms = np.sqrt((differences ** 2).sum(axis=1))
  # Radius of gaussian with std = human_rad/2
  #rad = spline_planner_utils.confidence_interval_half_width_numba(alphat, human_rad/100, 1)
  human_r = -10000*np.any(norms <= conformal_quantile) # Bad if we are within a CI
  #print(goal_r, human_r, obs_r)

  loss = 0 if last_residual <= conformal_quantile else 1

  reward = goal_r + obs_r + human_r

  return reward, loss

@numba.njit
def eval_reward_numba_conservative(goal, traj, vel, pred_splines, human_rad, sd_obs_interp):
  goal_r = -1.*np.linalg.norm(traj - goal[:2, None])

  obs_r = 0
  xs = traj[0]
  ys = traj[1]
  for i in range(xs.shape[0]):
    obs_r += 1000*sd_obs_interp[int(ys[i])-1, int(xs[i])-1]


  # NOTE: assumes that prediction length and time aligns!
  differences = pred_splines - traj[None, :]
  norms = np.sqrt((differences ** 2).sum(axis=1))
  # Radius of gaussian with std = human_rad/2
  #rad = spline_planner_utils.confidence_interval_half_width_numba(alphat, human_rad/100, 1)
  human_r = 1000*norms.min()
  #print(goal_r, human_r, obs_r)

  loss = 0

  reward = goal_r + obs_r + human_r

  return reward, loss



#   Computes a 3rd order spline of the form:
#      p(t) = a3(t/final_t)^3 + b3(t/final_t)^2 + c3(t/final_t) + d3
#      x(p) = a1p^3 + b1p^2 + c1p + d1
#      y(p) = a2p^2 + b2p^2 + c2p + d2
@numba.njit
def spline(start, goal, final_t, num_waypts):

  # Get coefficients for the spline.
  (xc, yc, pc) = gen_coeffs(start, goal, final_t);

  # Unpack x coeffs.
  a1 = xc[0]
  b1 = xc[1]
  c1 = xc[2]
  d1 = xc[3]

  # Unpack y coeffs.
  a2 = yc[0]
  b2 = yc[1]
  c2 = yc[2]
  d2 = yc[3]

  # Unpack p coeffs.
  a3 = pc[0]
  b3 = pc[1]
  c3 = pc[2]
  d3 = pc[3]

  # Compute state trajectory at time steps using coefficients.
  xs = np.zeros(num_waypts)
  ys = np.zeros(num_waypts)
  xsdot = np.zeros(num_waypts)
  ysdot = np.zeros(num_waypts)
  ps = np.zeros(num_waypts)
  psdot = np.zeros(num_waypts)
  ths = np.zeros(num_waypts)

  # Compute the control: u1 = linear vel, u2 = angular vel
  u1_lin_vel = np.zeros(num_waypts)
  u2_ang_vel = np.zeros(num_waypts)

  # Compute timestep between each waypt.
  dt = final_t/(num_waypts-1.)
  idx = 0
  t = 0
  #print("numwaypts: ", num_waypts)
  #print("dt: ", dt)
  #print("final_t: ", final_t)

  while (idx < num_waypts):
    tnorm = t/final_t

    # Compute (normalized) parameterized time var p and x,y and time derivatives of each.
    ps[idx]   = a3 * tnorm**3   + b3 * tnorm**2   + c3 * tnorm   + d3;
    xs[idx]   = a1 * ps[idx]**3 + b1 * ps[idx]**2 + c1 * ps[idx] + d1;
    ys[idx]   = a2 * ps[idx]**3 + b2 * ps[idx]**2 + c2 * ps[idx] + d2;
    xsdot[idx]  = 3. * a1 * ps[idx]**2 + 2 * b1 * ps[idx] + c1;
    ysdot[idx]  = 3. * a2 * ps[idx]**2 + 2 * b2 * ps[idx] + c2;
    psdot[idx]  = 3. * a3 * tnorm**2 + 2 * b3 * tnorm + c3;
    ths[idx] = np.arctan2(ysdot[idx], xsdot[idx]);

    # Compute speed (wrt time variable p).
    speed = np.sqrt(xsdot[idx]**2 + ysdot[idx]**2);

    xsddot = 6. * a1 * ps[idx] + 2. * b1
    ysddot = 6. * a2 * ps[idx] + 2. * b2

    # Linear Velocity (real-time):
    #    u1(t) = u1(p(t)) * dp(tnorm)/dtorm * dtnorm/dt
    u1_lin_vel[idx] = speed * psdot[idx] / final_t;

    # Angular Velocity (real-time):
    #    u2(t) = u2(p(t)) * dp(tnorm)/dtorm * dtnorm/dt
    u2_ang_vel[idx] = (xsdot[idx]*ysddot - ysdot[idx]*xsddot)/speed**2 * psdot[idx] / final_t;

    idx = idx + 1
    t = t + dt

  curr_spline = [xs, ys, u1_lin_vel, u2_ang_vel, ths]

  return curr_spline

@numba.njit
def gen_coeffs(start, goal, final_t):
  # Extract states.
  x0 = start[0]
  y0 = start[1]
  th0 = start[2]
  v0 = start[3]

  xg = goal[0]
  yg = goal[1]
  thg = goal[2]
  vg = goal[3]

  # Set heuristic coefficients.
  f1 = v0 + np.sqrt((xg - x0)*(xg - x0) + (yg - y0)*(yg - y0))
  f2 = f1

  # Compute x(p(t)) traj coeffs.
  d1 = x0
  c1 = f1*np.cos(th0)
  a1 = f2*np.cos(thg) - 2.*xg + c1 + 2.*d1
  b1 = 3.*xg - f2*np.cos(thg) - 2.*c1 - 3.*d1

  # Compute y(p(t))traj coeffs.
  d2 = y0
  c2 = f1*np.sin(th0)
  a2 = f2*np.sin(thg) - 2.*yg + c2 + 2.*d2
  b2 = 3.*yg - f2*np.sin(thg) - 2.*c2 - 3.*d2

  # Compute p(t) coeffs.
  d3 = 0.0;
  c3 = (final_t * v0) / f1
  a3 = (final_t * vg) / f2 + c3 - 2.
  b3 = 1. - a3 - c3

  xc = [a1, b1, c1, d1]
  yc = [a2, b2, c2, d2]
  pc = [a3, b3, c3, d3]

  return (xc, yc, pc)


def add_robot(ax, r, c, r_h, r_w, r_goal_reached):
    # Read the image file
    robot_image = plt.imread('./assets/robot_happy.png') if r_goal_reached else plt.imread('./assets/robot.png')

    # Compute the zoom factor to control the size of the image
    zoom = min(r_h / robot_image.shape[0], r_w / robot_image.shape[1])

    # Create an OffsetImage object for the robot emoji
    imagebox = OffsetImage(robot_image, zoom=zoom)

    # Create an AnnotationBbox object with the emoji and the position where you want to place it
    ab = AnnotationBbox(imagebox, (c, r), frameon=False, boxcoords="data", pad=0)

    # Add the AnnotationBbox object to the axes
    ax.add_artist(ab)

### Not njit versions

def my_gen_coeffs(start, goal, final_t):
  # Extract states.
  x0 = start[0]
  y0 = start[1]
  th0 = start[2]
  v0 = start[3]

  xg = goal[0]
  yg = goal[1]
  thg = goal[2]
  vg = goal[3]

  # Set heuristic coefficients.
  f1 = v0 + np.sqrt((xg - x0)*(xg - x0) + (yg - y0)*(yg - y0))
  f2 = f1

  # Compute x(p(t)) traj coeffs.
  d1 = x0
  c1 = f1*np.cos(th0)
  a1 = f2*np.cos(thg) - 2.*xg + c1 + 2.*d1
  b1 = 3.*xg - f2*np.cos(thg) - 2.*c1 - 3.*d1

  # Compute y(p(t))traj coeffs.
  d2 = y0
  c2 = f1*np.sin(th0)
  a2 = f2*np.sin(thg) - 2.*yg + c2 + 2.*d2
  b2 = 3.*yg - f2*np.sin(thg) - 2.*c2 - 3.*d2

  # Compute p(t) coeffs.
  d3 = 0.0
  c3 = (final_t * v0) / f1
  a3 = (final_t * vg) / f2 + c3 - 2.
  b3 = 1. - a3 - c3

  xc = [a1, b1, c1, d1]
  yc = [a2, b2, c2, d2]
  pc = [a3, b3, c3, d3]

  return (xc, yc, pc)

def my_spline(start, goal, final_t, num_waypts):

  # Get coefficients for the spline.
  (xc, yc, pc) = my_gen_coeffs(start, goal, final_t);

  # Unpack x coeffs.
  a1 = xc[0]
  b1 = xc[1]
  c1 = xc[2]
  d1 = xc[3]

  # Unpack y coeffs.
  a2 = yc[0]
  b2 = yc[1]
  c2 = yc[2]
  d2 = yc[3]

  # Unpack p coeffs.
  a3 = pc[0]
  b3 = pc[1]
  c3 = pc[2]
  d3 = pc[3]

  # Compute state trajectory at time steps using coefficients.
  xs = np.zeros(num_waypts)
  ys = np.zeros(num_waypts)
  xsdot = np.zeros(num_waypts)
  ysdot = np.zeros(num_waypts)
  ps = np.zeros(num_waypts)
  psdot = np.zeros(num_waypts)
  ths = np.zeros(num_waypts)

  # Compute the control: u1 = linear vel, u2 = angular vel
  u1_lin_vel = np.zeros(num_waypts)
  u2_ang_vel = np.zeros(num_waypts)

  # Compute timestep between each waypt.
  dt = final_t/(num_waypts-1.)
  idx = 0
  t = 0
  #print("numwaypts: ", num_waypts)
  #print("dt: ", dt)
  #print("final_t: ", final_t)

  while (idx < num_waypts):
    tnorm = t/final_t

    # Compute (normalized) parameterized time var p and x,y and time derivatives of each.
    ps[idx]   = a3 * tnorm**3   + b3 * tnorm**2   + c3 * tnorm   + d3
    xs[idx]   = a1 * ps[idx]**3 + b1 * ps[idx]**2 + c1 * ps[idx] + d1
    ys[idx]   = a2 * ps[idx]**3 + b2 * ps[idx]**2 + c2 * ps[idx] + d2
    xsdot[idx]  = 3. * a1 * ps[idx]**2 + 2 * b1 * ps[idx] + c1
    ysdot[idx]  = 3. * a2 * ps[idx]**2 + 2 * b2 * ps[idx] + c2
    psdot[idx]  = 3. * a3 * tnorm**2 + 2 * b3 * tnorm + c3
    ths[idx] = np.arctan2(ysdot[idx], xsdot[idx])

    # Compute speed (wrt time variable p).
    speed = np.sqrt(xsdot[idx]**2 + ysdot[idx]**2)

    xsddot = 6. * a1 * ps[idx] + 2. * b1
    ysddot = 6. * a2 * ps[idx] + 2. * b2

    # Linear Velocity (real-time):
    #    u1(t) = u1(p(t)) * dp(tnorm)/dtorm * dtnorm/dt
    u1_lin_vel[idx] = speed * psdot[idx] / final_t

    # Angular Velocity (real-time):
    #    u2(t) = u2(p(t)) * dp(tnorm)/dtorm * dtnorm/dt
    u2_ang_vel[idx] = (xsdot[idx]*ysddot - ysdot[idx]*xsddot)/speed**2 * psdot[idx] / final_t;

    idx = idx + 1
    t = t + dt

  return np.array([xs, ys, u1_lin_vel, u2_ang_vel, ths])