import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.preprocessing import load_SDD
from utils.spline_planner import SplinePlanner, my_spline
from tqdm import tqdm
import yaml
#from utils.rrt import run_rrt, point_to_poly

def collapse_samples(df):
    """
    Collapses the samples in the dataframe df to calculate the mean and standard deviation
    of the 'pred_x', 'pred_y', 'goal_x', and 'goal_y' columns, grouped by all columns other
    than these and 'sample'.

    Parameters:
    df (pd.DataFrame): The input dataframe

    Returns:
    pd.DataFrame: The output dataframe with the collapsed samples
    """
    # Handle potential renaming and dropping of columns
    if 'x_y' in df.columns and 'y_y' in df.columns:
        df.drop(['x_y', 'y_y'], axis=1, inplace=True)
    if 'x_x' in df.columns and 'y_x' in df.columns:
        df.rename({'x_x': 'x', 'y_x' : 'y'}, axis=1, inplace=True)

    # Drop unwanted columns
    df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True, errors='ignore')

    # Identify columns other than 'pred_x', 'pred_y', 'goal_x', 'goal_y' and 'sample' to group by
    groupby_cols = [col for col in df.columns if col not in ('pred_x', 'pred_y', 'goal_x', 'goal_y', 'sample')]

    # Group by the identified columns and calculate mean and std dev of 'pred_x', 'pred_y', 'goal_x', and 'goal_y'
    print("df before: ", len(df))
    df = df.groupby(groupby_cols, as_index=False).agg(
        pred_x=('pred_x', 'mean'),
        pred_x_std=('pred_x', 'std'),
        pred_y=('pred_y', 'mean'),
        pred_y_std=('pred_y', 'std'),
        goal_x=('goal_x', 'mean'),
        goal_y=('goal_y', 'mean'),
    )
    print("df after: ", len(df))

    return df

def plan_trajectory(df, params):
    planner = SplinePlanner(
        params['num_waypts'],
        params['horizon'],
        params['max_linear_vel'],
        params['max_angular_vel'],
        params['gmin'],
        params['gmax'],
        params['gnums'],
        params['goal_rad'],
        params['human_rad'],
        params['lr'],
        params['alpha'],
        params['binary_env_img_path']
    )
    r_start = params['r_start']
    j = 0
    r_goal = params['r_goal'][j]
    # If there's only one goal, and you're in it, no need to plan
    r_goal_reached = (np.linalg.norm(r_start[:2] - params['r_goal'][-1][:2]) <= params['goal_rad']) & (len(params['r_goal']) == 1)

    plan_df = []
    worst_residuals = []
    print("[plan_trajectory] Planning...")
    frames_indices = sorted(df[df.frame >= params['starting_frame']].frame.unique())
    init_frame = frames_indices[0]

    if params['method'] == 'conformal CBF':
        # Safe Decentralized Multi-Agent Control using Black-Box Predictors, Conformal Policies, and Barrier Functions
            # constant learning rate for conformal update: eta (planner.lr)
            # user-defined target average loss: epsilon (planner.alphat)
            # period of the sensor's sampling of trajectories: tau (tau)
            # constant for the linear alpha function: a (planner.alpha)
        solve_rate, planner.alpha, dynamics_args, ground_truth, tau, is_learning, planner.alphat, loss_type = params['conformal_CBF_params']
        dynamics_type = dynamics_args[0] # lin_ang__vels, velocity_dyn or double_integral
        if not dynamics_type in ['lin_ang__vels', 'velocity_dyn', 'double_integral']:
            print("Error: Unknown dynamics type [", dynamics_type, "]")
            print("Expected dynamics: lin_ang__vels, velocity_dyn or double_integral")
            exit(1)
        planner.collide_dist = (params['human_size'] + params['robot_size']) * params['px_per_m']
        df = df[df.ahead <= tau]
        if dynamics_type == 'lin_ang__vels':
            init_df = df[(df.frame == init_frame) & (df.ahead == 1)]
            human_init_pos = np.stack([
                init_df[init_df.metaId == mid].iloc[0].loc[['x', 'y']].to_numpy()
                for mid in init_df.metaId.unique()], axis=0)
            ref_path = my_spline(r_start, r_goal, 6 * planner.horizon_, 6 * planner.num_waypts_)
        else:
            human_init_pos = None
            ref_path = None


        time_step = 0
        #dist_to_goal = np.linalg.norm(r_goal[:2] - r_start[:2])

        avg_ref, avg_A, avg_b, safe_vs, constr_vs, avg_lmbd, losses, unsat_frames, nb_safe_u_ref, nb_conservative_pred, nb_both_true_but_0_loss = 0, np.zeros(2), 0, [], [], 0, [], [], 0, 0, 0
        stats_df = [pd.DataFrame({
                                'frame': 0,
                                'metaID': [-1],
                                'h_dot_alpha_h': 0,
                                'distance': 0,
                                'cbf_func': 0,
                                'grad_cbf': 0,
                                'u_ref': 0,
                                'u_qp': 0
                            })]
        #static_pedestrian = np.array([[]])
        
    for frame_idx in tqdm(frames_indices):
        # Check if goal has been reached
        if np.linalg.norm(r_start[:2] - r_goal[:2]) <= params['goal_rad'] and not r_goal_reached:
            j += 1
        if j == len(params['r_goal']): # If we've exceeded the number of goals
            if not r_goal_reached:
                print(f"Goal {j} reached after {frame_idx - init_frame} frames.")
            r_goal_reached = True
        if not r_goal_reached:
            r_goal = params['r_goal'][j]
            
            if params['method'] == 'conformal CBF':
                # Prepare data
                curr_df = df[df.frame == frame_idx]
                curr_metaIds = curr_df.metaId.unique()
                human_pos_gt = np.stack([
                    curr_df[(curr_df.metaId == mid) & (curr_df.ahead == 1)].iloc[0].loc[['x', 'y', 'metaId']].to_numpy()
                    for mid in curr_metaIds], axis=0)
                if not time_step:
                    N = len(curr_metaIds)
                    human_traj_preds = np.empty((tau + 1, N, 2))
                    human_traj_preds[0] = human_pos_gt[:, :2]
                    if ground_truth:
                        for t in range(1, tau+1):
                            preds_at_t = np.empty((N, 2))
                            frame_ahead = frame_idx + t
                            for idx in range(N):
                                mid = curr_metaIds[idx]
                                if ((df.metaId == mid) & (df.frame == frame_ahead)).any():
                                    preds_at_t[idx] = df[(df.metaId == mid) & (df.frame == frame_ahead)].iloc[0].loc[['x', 'y']].to_numpy()
                                else:
                                    preds_at_t[idx] = human_traj_preds[t-1, idx]
                            human_traj_preds[t] = preds_at_t
                    else: # use predictions
                        for t in range(1, tau+1):
                            human_traj_preds[t] = np.stack([
                                curr_df[(curr_df.metaId == mid) & (curr_df.ahead == t)].iloc[0].loc[['pred_x', 'pred_y']].to_numpy()
                                for mid in curr_metaIds], axis=0)
                    preds_mid_idx = {curr_metaIds[i]: i for i in range(N)}
                
                # Plan
                unsat_frame = False
                for sub_frame in range(solve_rate):
                    robot_plan, (v_ref, A, b, safe_v, constr_v, lmbd, loss, unsat, safe_u_ref, conservative_pred, both_true_but_0_loss, record_CBF) = planner.plan(r_start, r_goal, (human_traj_preds, time_step, N, preds_mid_idx, human_pos_gt), 'conformal CBF', worst_residuals=np.array([0]), conformal_CBF_args=(ref_path, human_init_pos, tau, loss_type, frame_idx, solve_rate, sub_frame, dynamics_args, is_learning))
                    avg_ref += v_ref
                    avg_A += A
                    avg_b += b
                    safe_vs.append(safe_v)
                    constr_vs.append(constr_v)
                    avg_lmbd += lmbd
                    if unsat:
                        unsat_frame = True
                        unsat_frames.append((frame_idx, sub_frame))
                    nb_safe_u_ref += safe_u_ref
                    nb_conservative_pred += conservative_pred
                    nb_both_true_but_0_loss += both_true_but_0_loss
                    if not loss is None:
                        losses.append(loss)
                    r_start = [ robot_plan[0][1], robot_plan[1][1], robot_plan[4][1], robot_plan[2][1] ]
                    '''if is_learning and loss_type == 'QPcontrol' and not sub_frame and not record_CBF and not time_step and planner.prev_state:
                        print("Problem")
                        exit()'''
                    for record in record_CBF:
                        t, mid, value, dist, h, grad_h, u_ref, u_qp = record
                        stats_df += [
                            pd.DataFrame({
                                'frame': frame_idx - tau + t / solve_rate,
                                'metaID': [mid],
                                'h_dot_alpha_h': value,
                                'distance': dist,
                                'cbf_func': h,
                                'grad_cbf': grad_h,
                                'u_ref': u_ref,
                                'u_qp': u_qp
                            })
                        ]
                plan_df += [
                    pd.DataFrame({
                        'frame': frame_idx,
                        'x': [r_start[1]], # x
                        'y': [r_start[0]], # y
                        'aheads': 1,
                        'pred_x': robot_plan[1][1],
                        'pred_y': robot_plan[0][1],
                        'r_goal_reached': False,
                        'r_goal_count': j,
                        'lambda' : planner.lamda,
                        'alphat' : planner.alphat,
                        'metaId': -1,
                        'ref_vel': v_ref,
                        'unsatQP': unsat_frame
                    })
                ]

                # Take next step
                time_step = (time_step + 1) % tau
                r_start = [ robot_plan[0][1], robot_plan[1][1], robot_plan[4][1], robot_plan[2][1] ]
            else:
                # Prepare data
                curr_df = df[df.frame == frame_idx]
                human_spline_preds = np.stack([
                    np.stack([
                        curr_df[curr_df.metaId == mid].pred_x.to_numpy(),
                        curr_df[curr_df.metaId == mid].pred_y.to_numpy()
                        ], axis=0)
                    for mid in curr_df.metaId.unique()], axis=0)
                # Plan
                agent_df = curr_df[curr_df.metaId >= 0]
                all_residuals_timestep = np.sqrt((agent_df.pred_x - agent_df.future_x)**2 + (agent_df.pred_y - agent_df.future_y)**2)
                worst_residuals += [ all_residuals_timestep.max() ]
                robot_plan = planner.plan( r_start, r_goal, human_spline_preds, params['method'], worst_residuals=np.array(worst_residuals))
                plan_df += [
                    pd.DataFrame({
                        'frame': frame_idx,
                        'x': (params['horizon']-1)*[r_start[1]], # x
                        'y': (params['horizon']-1)*[r_start[0]], # y
                        'aheads': np.arange(params['horizon'])[:-1] + 1,
                        'pred_x': robot_plan[1][1:],
                        'pred_y': robot_plan[0][1:],
                        'r_goal_reached': False,
                        'r_goal_count': j,
                        'lambda' : planner.lamda,
                        'alphat' : planner.alphat,
                        'metaId': -1
                    })
                ]
                # Take next step
                r_start = [ robot_plan[0][1], robot_plan[1][1], robot_plan[4][1], robot_plan[2][1] ]

        else:
            if params['method'] == 'conformal CBF':
                plan_df += [
                    pd.DataFrame({
                        'frame': frame_idx,
                        'x': r_start[1], # x
                        'y': r_start[0], # y
                        'aheads': 1,
                        'pred_x': [r_start[1]],
                        'pred_y': [r_start[0]],
                        'r_goal_reached': True,
                        'r_goal_count': j,
                        'lambda' : None,
                        'alphat' : None,
                        'metaId': -1
                    })
                ]
            else:
                plan_df += [
                    pd.DataFrame({
                        'frame': frame_idx,
                        'x': (params['horizon']-1)*[r_start[1]], # x
                        'y': (params['horizon']-1)*[r_start[0]], # y
                        'aheads': np.arange(params['horizon'])[:-1] + 1,
                        'pred_x': (params['horizon']-1)*[r_start[1]],
                        'pred_y': (params['horizon']-1)*[r_start[0]],
                        'r_goal_reached': True,
                        'r_goal_count': j,
                        'lambda' : None,
                        'alphat' : None,
                        'metaId': -1
                    })
                ]
    
    plan_df = pd.concat([df] + plan_df, axis=0, ignore_index=True)
    if params['method'] == 'conformal CBF':
        nb_frames = len(frames_indices)
        print(f"Average ref vel: {avg_ref / nb_frames}, Average A: {avg_A / nb_frames}, Average b: {avg_b / nb_frames}, Average lambda: {avg_lmbd / nb_frames}, Unsatisfiable QP: {len(unsat_frames)}")
        safe_vs, constr_vs = np.array(safe_vs), np.array(constr_vs)
        losses = np.array(losses)
        print(f"Safety violations:\nMin: {np.min(safe_vs)}, Avg: {np.average(safe_vs)}, Max: {np.max(safe_vs)}")
        print(f"Steps of violation: {np.nonzero(safe_vs) + init_frame}")
        #print(f"Number of agents that weren't seen: {count_invis}")
        print(f"Predicted constraint violations:\nMin: {np.min(constr_vs)}, Avg: {np.average(constr_vs)}, Max: {np.max(constr_vs)}")
        if is_learning:
            print(f"Loss stats:\nMin: {np.min(losses)}, Avg: {np.average(losses)}, Max: {np.max(losses)}")
            stats_df = pd.concat(stats_df, axis=0, ignore_index=True)
        print(f"Reference cntrl was safe: {nb_safe_u_ref}, Prediction was too conservative: {nb_conservative_pred}, Both happened but positive loss: {nb_both_true_but_0_loss}")
        
        return plan_df, stats_df
    else:
        return plan_df

def convert_state(state, H, W):
    return np.array([ float(state[0]) * W, float(state[1]) * H, float(state[2]), float(state[3]) ])

def convert_units(params, H, W):
    params['H_m'] = int(params['H_m'])
    params['W_m'] = int(params['W_m'])
    px_per_m = 0.5*H/params['H_m'] + 0.5*W/params['W_m'] # Average for robustness
    aheads=list(range(int(params['horizon'])))

    # Unit conversions
    robot_rad = 2 * params['robot_size'] * px_per_m
    params['gmin'] = [params['gmin'][0]*W+robot_rad, params['gmin'][1]*H+robot_rad]
    params['gmax'] = [params['gmax'][0]*W-robot_rad, params['gmax'][1]*H-robot_rad]
    params['aheads'] = aheads
    #params['sd_obs'] = np.ones(params['gnums']) * 10000
    params['binary_env_img_path'] = 'nexus_4_frame_bin.png'
    params['max_linear_vel'] *= px_per_m / float(params['fps'])
    params['goal_rad'] *= px_per_m
    params['human_rad'] *= px_per_m
    params['r_start'] = convert_state(params['r_start'], H, W)
    params['r_goal'] = [ convert_state(g, H, W) for g in params['r_goal'] ]
    params['px_per_m'] = px_per_m
    return params

def setup_params(params, H, W, lr, method, extra=None):
    params['r_goal'] = [np.array(list(g.values())).squeeze() for g in params['r_goal']]
    params = convert_units(params, H, W)
    params['lr'] = lr
    params['method'] = method
    params['conformal_CBF_params'] = extra if method == 'conformal CBF' else None
    return params

if __name__ == "__main__":
    # Setup
    scene = sys.argv[1]
    forecaster = sys.argv[2]
    method = sys.argv[3]
    json_name = './params/' + scene + '.json'
    params = yaml.safe_load(open(json_name))

    mode = 'train' if scene in os.listdir('./data/train') else 'test'
    cmap = plt.get_cmap('terrain')
    frames, _, reference_image = load_SDD(scene, load_frames=[0], mode=mode)
    H, W, _ = frames[0].shape

    if method == 'conformal CBF':
        solve_rate = int(sys.argv[4])
        a_lin = float(sys.argv[5])
        dynamics_type = sys.argv[6]
        str_append = 's' + str(solve_rate) + '_' + 'a' + str(a_lin).replace('.', '_')
        if dynamics_type == 'velocity_dyn':
            K_rep = float(sys.argv[7])
            K_att = float(sys.argv[8])
            rho0 = float(sys.argv[9])
            dynamics_args = (dynamics_type, K_rep, K_att, rho0)
            str_append += '_' + 'Vcntrl' + '_' + str(K_rep).replace('.', '_') + '_' + str(K_att).replace('.', '_') + '_' + str(rho0).replace('.', '_')
            next_arg = 10
        elif dynamics_type == 'double_integral':
            K_acc = float(sys.argv[7])
            K_rep = float(sys.argv[8])
            K_att = float(sys.argv[9])
            rho0 = float(sys.argv[10])
            dynamics_args = (dynamics_type, K_acc, K_rep, K_att, rho0)
            str_append += '_' + 'DI' + '_' + str(K_acc).replace('.', '_') + '_' + str(K_rep).replace('.', '_') + '_' + str(K_att).replace('.', '_') + '_' + str(rho0).replace('.', '_')
            next_arg = 11
        else:
            print("Unimplemented dynamics")
            exit(1)
        ground_truth = (sys.argv[next_arg] == 'gound_truth')
        if ground_truth:
            str_append += '_' + 'gt'
            tau = 2
        else:
            tau = int(sys.argv[next_arg])
            str_append += '_' + 'pred' + str(tau)
        next_arg += 1
        no_learning = (sys.argv[next_arg] == '-no_learning')
        if no_learning:
            next_arg += 1
            lr = 0 #value unused
            eps = 0 #value unused
            loss_type = '' #value unused
            str_append += '_' + 'no_learning'
        else:
            # learning parameters
            loss_type = sys.argv[next_arg]
            lr = float(sys.argv[next_arg + 1])
            eps = float(sys.argv[next_arg + 2])
            str_append += '_' + loss_type + '_' + 'lr' + str(lr).replace('.', '_') + '_' + 'e' + str(eps).replace('.', '_')

        params = setup_params(params, H, W, lr, method, extra=(solve_rate, a_lin, dynamics_args, ground_truth, tau, not no_learning, eps, loss_type))
        plan_folder = './plans/' + scene + '/conformal_CBF/'
        stats_filename = os.path.join(plan_folder, 'stats_df_' + str_append + '.csv')
    else:
        lr = float(sys.argv[4])
        str_append = forecaster + '_' + method.replace(' ', '_') + "_" + str(lr).replace('.', '_')
        params = setup_params(params, H, W, lr, method=method)
        plan_folder = './plans/' + scene + '/decision_theory/'

    plan_filename = os.path.join(plan_folder, 'plan_df_' + str_append + '.csv')

    # By default, do not overwrite, option as last argument
    overwrite = (sys.argv[-1] == '-overwrite')

    if overwrite or not os.path.exists(plan_filename):
        if overwrite:
            print(f"[plan_trajectory] Planning {str_append} from scratch, overwriting previous save if any.")
        elif not os.path.exists(plan_filename):
            print(f"[plan_trajectory] cannot find {str_append}, planning from scratch.")
        # Setup parameters
        df = pd.read_csv('./data/' + forecaster + '/' + scene + '.csv')
        if forecaster == 'ynet':
            df = collapse_samples(df)

        # Create future trajectories
        fake_df = df.copy()
        fake_df['frame'] = df['frame'] - df['ahead']
        fake_df.rename(columns={'x' : 'future_x', 'y' : 'future_y'}, inplace=True)
        if forecaster == 'darts':
            fake_df.drop(['Unnamed: 0', 'trackId', 'pred_x', 'pred_y'], axis=1, inplace=True)
        elif forecaster == 'ynet':
            fake_df.drop(['trackId', 'sceneId', 'pred_x', 'pred_y', 'pred_x_std', 'pred_y_std', 'goal_x', 'goal_y'], axis=1, inplace=True)
        fake_df = fake_df.loc[fake_df['frame'] >= 0]
        future_df = df.merge(fake_df, how='left', on=['frame', 'metaId', 'ahead'])

        # Plan trajectory
        if params['method'] == 'conformal CBF':
            plan_df, stats_df = plan_trajectory(future_df, params)
            # Save the plans, along with the forecaster and learning rate
            os.makedirs(plan_folder, exist_ok=True)
            if not no_learning:
                stats_df.to_csv(stats_filename)
        else:
            plan_df = plan_trajectory(future_df, params)
        # Save the plans, along with the forecaster and learning rate
        os.makedirs(plan_folder, exist_ok=True)
        plan_df.to_csv(plan_filename)
    else:
        print(f"Already cached {str_append}")
