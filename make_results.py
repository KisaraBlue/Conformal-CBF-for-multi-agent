import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.preprocessing import load_SDD
from utils.plotting_utils import generate_video
from utils.metrics import make_metrics, plot_CBFs
from plan_trajectory import setup_params
import yaml

if __name__ == "__main__":
    # Setup
    scene = sys.argv[1]
    forecaster = sys.argv[2]
    method = sys.argv[3]
    json_name = './params/' + scene + '.json'
    step = 1
    params = yaml.safe_load(open(json_name))
    mode = 'train' if scene in os.listdir('./data/ynet_additional_files/data/SDD/train') else 'test'
    # By default, do not overwrite, or if True, do
    # overwrite = False if len(sys.argv) < 8 else sys.argv[7] == 'True'
    cmap = plt.get_cmap('terrain')
    frames, _, _ = load_SDD(scene, mode=mode, load_frames=[0])
    H, W, _  = frames[0].shape
    df = pd.read_csv('./data/' + forecaster + '/' + scene + '.csv')
    
    if method == 'conformal CBF':
        solve_rate = int(sys.argv[4])
        a_lin = float(sys.argv[5])
        dynamics_type = sys.argv[6]
        str_append = 's' + str(solve_rate) + '_' + 'a' + str(a_lin).replace('.', '_')
        if dynamics_type == 'velocity_dyn':
            K_acc = 'NA'
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
            lr = 'NA' #value unused
            eps = 'NA' #value unused
            loss_type = 'no_learning' #value unused
            str_append += '_' + 'no_learning'
        else:
            # learning parameters
            loss_type = sys.argv[next_arg]
            lr = float(sys.argv[next_arg + 1])
            eps = float(sys.argv[next_arg + 2])
            next_arg += 3
            str_append += '_' + loss_type + '_' + 'lr' + str(lr).replace('.', '_') + '_' + 'e' + str(eps).replace('.', '_')
        metrics_run = 'others/' if len(sys.argv) < next_arg + 1 else (sys.argv[next_arg] + '/')

        params = setup_params(params, H, W, lr, method, extra=(solve_rate, a_lin, dynamics_args, ground_truth, tau, not no_learning, eps, loss_type))
        plan_folder = './plans/' + scene + '/conformal_CBF/'
        metrics_folder = './metrics/' + scene + '/conformal_CBF/' + metrics_run
        videos_folder = './videos/' + scene + '/conformal_CBF/'
        stats_filename = os.path.join(plan_folder, 'stats_df_' + str_append + '.csv')
        extra_params = (solve_rate, a_lin, dynamics_type, K_acc, K_rep, K_att, rho0, 'gt' if ground_truth else tau, loss_type, lr, eps)
    else:
        lr = float(sys.argv[4])
        str_append = forecaster + '_' + method.replace(' ', '_') + "_" + str(lr).replace('.', '_')
        params = setup_params(params, H, W, lr, method)
        plan_folder = './plans/' + scene + '/decision_theory/'
        metrics_folder = './metrics/' + scene + '/decision_theory/'
        videos_folder = './videos/' + scene + '/decision_theory/'
        extra_params = None
    
    plan_filename = os.path.join(plan_folder, 'plan_df_' + str_append + '.csv')
    metrics_filename = os.path.join(metrics_folder, 'metrics_df_' + str_append + '.csv')
    video_filename = 'vid_' + str_append

    try:
        plan_df = pd.read_csv(plan_filename)
    except:
        raise ValueError("No plan found for " + str(scene) + str(lr) + str(method))

    frames_to_read = plan_df[plan_df.r_goal_reached == False].frame.unique()
    frames_to_read = frames_to_read[::step]
    if (frames_to_read.max() < plan_df.frame.max()) & ((plan_df.r_goal_reached == True).sum() > 0):
        frames_to_read = np.append(frames_to_read, int(plan_df[plan_df.r_goal_reached == True].frame.unique().min()))
    if method == 'conformal CBF' and not no_learning:
        plot_name = 'plots_' + (str_append if not str_append is None else 'placeholder') + '.png'
        plot_CBFs(pd.read_csv(stats_filename), metrics_folder, plot_name)

    # Generate video
    gen_video = (sys.argv[-1] != '-no_video')
    if gen_video:
        cmap_lut = cmap(np.random.permutation(np.linspace(0, 1, plan_df.metaId.max()+1)))
        frames, _, _ = load_SDD(scene, mode=mode, load_frames=frames_to_read)
        generate_video(frames, plan_df, cmap_lut, videos_folder, video_filename, aheads=params['aheads'], params=params, truncate_video=True, frame_indexes=frames_to_read)

    # Generate metrics
    make_metrics(plan_df, metrics_folder, metrics_filename, forecaster, method, 0 if lr == 'NA' else float(lr), params, str_append=str_append, extra_params=extra_params)
