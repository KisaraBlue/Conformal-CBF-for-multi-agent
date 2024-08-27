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
    lr = float(sys.argv[4])
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
        eps = float(sys.argv[5])
        tau = int(sys.argv[6])
        a_lin = float(sys.argv[7])
        loss_type = 'QPcontrol' if len(sys.argv) < 9 or sys.argv[8] != '-all_predictions' else 'Predictor'
        params = setup_params(params, H, W, lr, method, extra=(eps, tau, a_lin, loss_type))
        str_append = 'h' + str(lr).replace('.', '_') + '_' + 'e' + str(eps).replace('.', '_') + '_' + 't' + str(tau) + '_' + 'a' + str(a_lin).replace('.', '_') + '_' + loss_type
        plan_folder = './plans/' + scene + '/conformal_CBF/'
        metrics_folder = './metrics/' + scene + '/conformal_CBF/'
        videos_folder = './videos/' + scene + '/conformal_CBF/'
        stats_filename = os.path.join(plan_folder, 'stats_df_' + str_append + '.csv')
    else:
        str_append = forecaster + '_' + method.replace(' ', '_') + "_" + str(lr).replace('.', '_')
        params = setup_params(params, H, W, lr, method)
        plan_folder = './plans/' + scene + '/decision_theory/'
        metrics_folder = './metrics/' + scene + '/decision_theory/'
        videos_folder = './videos/' + scene + '/decision_theory/'
    
    plan_filename = os.path.join(plan_folder, 'plan_df_' + str_append + '.csv')
    metrics_filename = os.path.join(metrics_folder, 'metrics_df_' + str_append + '.csv')
    video_filename = str_append

    try:
        plan_df = pd.read_csv(plan_filename)
    except:
        raise ValueError("No plan found for " + str(scene) + str(lr) + str(method))

    frames_to_read = plan_df[plan_df.r_goal_reached == False].frame.unique()
    frames_to_read = frames_to_read[::step]
    if (frames_to_read.max() < plan_df.frame.max()) & ((plan_df.r_goal_reached == True).sum() > 0):
        frames_to_read = np.append(frames_to_read, int(plan_df[plan_df.r_goal_reached == True].frame.unique().min()))
    if method == 'conformal CBF':
        plot_name = 'plots_' + (str_append if not str_append is None else 'placeholder') + '.png'
        plot_CBFs(pd.read_csv(stats_filename), metrics_folder, plot_name)

    # Generate video
    gen_video = (sys.argv[-1] != '-no_video')
    if gen_video:
        cmap_lut = cmap(np.random.permutation(np.linspace(0, 1, plan_df.metaId.max()+1)))
        frames, _, _ = load_SDD(scene, mode=mode, load_frames=frames_to_read)
        generate_video(frames, plan_df, cmap_lut, videos_folder, video_filename, aheads=params['aheads'], params=params, truncate_video=True, frame_indexes=frames_to_read)

    # Generate metrics
    make_metrics(plan_df, metrics_folder, metrics_filename, forecaster, method, float(lr), params, str_append=str_append)
