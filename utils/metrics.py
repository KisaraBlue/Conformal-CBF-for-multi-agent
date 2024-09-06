import os, sys
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def get_distances(df, px_per_m=None):
    unique_frames = df[df.r_goal_reached == False].frame.unique()
    distances = []
    for frame in sorted(unique_frames):
        curr_df = df[df.frame == frame]
        distances += [ # TODO: Fix this issue elsewhere too.
            np.sqrt((curr_df[curr_df.metaId == -1].y.iloc[0] - curr_df[curr_df.metaId != -1].x.to_numpy())**2 +
            (curr_df[curr_df.metaId == -1].x.iloc[0] - curr_df[curr_df.metaId != -1].y.to_numpy())**2).min()
        ]
    distances = np.array(distances) / px_per_m if px_per_m is not None else np.array(distances)
    return distances

def make_metrics(df, metrics_folder, metrics_filename, forecaster, method, lr, params, fps=30, str_append=None):
    unique_frames = df[df.r_goal_reached == False].frame.unique()
    # Calculate distances of closest human to robot in each frame
    distances = get_distances(df)
    collision_distance = ((params['robot_size'] + params['human_size'])*params['px_per_m'])
    did_collide = distances.min() < collision_distance
    did_succeed = df.r_goal_count.max() > 0
    metrics = {
        'forecaster' : forecaster,
        'method' : method,
        'lr' : lr,
        'success' : str(did_succeed),
        'goal time (s)' : (unique_frames.max() - unique_frames.min())/fps/df.r_goal_count.max() if did_succeed else np.infty,
        'safe' : str(~did_collide),
        'min dist (m)' : distances.min()/params['px_per_m'],
        'avg dist (m)' : distances.mean()/params['px_per_m'],
        '5% dist (m)'  : np.quantile(distances, 0.05)/params['px_per_m'],
        '10% dist (m)' : np.quantile(distances, 0.1)/params['px_per_m'],
        '25% dist (m)' : np.quantile(distances, 0.25)/params['px_per_m'],
        '50% dist (m)' : np.quantile(distances, 0.5)/params['px_per_m']
    }
    metrics = pd.DataFrame(metrics, index=[0])
    # Save out
    os.makedirs(metrics_folder, exist_ok=True)
    metrics.to_csv(metrics_filename)  
    if method == 'conformal CBF':
        plot_name = 'lambda_plot_' + (str_append if not str_append is None else 'placeholder') + '.png'
        plot_name = os.path.join(metrics_folder, plot_name)
        plot_lambda(df, unique_frames, plot_name)

def plot_lambda(df, unique_frames, plot_name):
    lmbd = []
    frames = sorted(unique_frames)
    for frame in frames:
        curr_df = df[(df.frame == frame) & (df.metaId == -1)]
        lmbd.append(curr_df.at[curr_df.index[0],'lambda'])
    fig = plt.figure()
    plt.plot(frames, lmbd, color='r', linewidth=0.5, label='lambda')
    plt.legend(loc="upper left")
    plt.savefig(plot_name)
    plt.close(fig)

def plot_CBFs(df, metrics_folder, plot_name):
    unique_metaID = df[df.metaID > -1].metaID.unique()
    N = len(unique_metaID)
    h_color = np.linspace(0, 1, N)
    min_dist, min_cbf = 0, 0
    dist_fig = plt.figure()
    for j in range(N):
        mid = unique_metaID[j]
        dist = []
        unique_frames = df[df.metaID == mid].frame.unique()
        frames = sorted(unique_frames)
        for frame in frames:
            curr_df = df[(df.frame == frame) & (df.metaID == mid)]
            dist.append(curr_df.at[curr_df.index[0], 'distance'])
        min_dist = min(min_dist, np.min(dist))
        color_j = hsv_to_rgb([h_color[j], 1, 1])
        plt.plot(frames, dist, color=color_j, linewidth=0.5, label=('agent ' + str(mid)))
    #plt.ylim(min_dist - 1,  4 * (1 - min_dist))
    plt.axhline(0, color='black', linewidth=0.25)
    plt.savefig(metrics_folder + 'dist_' + plot_name)
    plt.legend(loc="upper left")
    plt.close(dist_fig)
    cbf_fig = plt.figure()
    for j in range(N):
        mid = unique_metaID[j]
        cbf = []
        unique_frames = df[df.metaID == mid].frame.unique()
        frames = sorted(unique_frames)
        for frame in frames:
            curr_df = df[(df.frame == frame) & (df.metaID == mid)]
            cbf.append(curr_df.at[curr_df.index[0], 'h_dot_alpha_h'])
        min_cbf = min(min_cbf, np.min(cbf))
        color_j = hsv_to_rgb([h_color[j], 1, 1])
        plt.plot(frames, cbf, color=color_j, linewidth=0.5, label=('agent ' + str(mid)))
    #plt.ylim(min_cbf - 1,  4 * (1 - min_cbf))
    plt.axhline(0, color='black', linewidth=0.25)
    plt.savefig(metrics_folder + 'cbf_' + plot_name)
    plt.close(cbf_fig)
    h_fig = plt.figure()
    for j in range(N):
        mid = unique_metaID[j]
        h = []
        unique_frames = df[df.metaID == mid].frame.unique()
        frames = sorted(unique_frames)
        for frame in frames:
            curr_df = df[(df.frame == frame) & (df.metaID == mid)]
            h.append(curr_df.at[curr_df.index[0], 'cbf_func'])
        color_j = hsv_to_rgb([h_color[j], 1, 1])
        plt.plot(frames, h, color=color_j, linewidth=0.5, label=('agent ' + str(mid)))
    #plt.ylim(min_dist - 1,  4 * (1 - min_dist))
    plt.axhline(0, color='black', linewidth=0.25)
    plt.savefig(metrics_folder + 'h_' + plot_name)
    plt.legend(loc="upper left")
    plt.close(h_fig)
    grad_h_fig = plt.figure()
    for j in range(N):
        mid = unique_metaID[j]
        grad_h = []
        unique_frames = df[df.metaID == mid].frame.unique()
        frames = sorted(unique_frames)
        for frame in frames:
            curr_df = df[(df.frame == frame) & (df.metaID == mid)]
            grad_h.append(curr_df.at[curr_df.index[0], 'grad_cbf'])
        color_j = hsv_to_rgb([h_color[j], 1, 1])
        plt.plot(frames, grad_h, color=color_j, linewidth=0.5, label=('agent ' + str(mid)))
    #plt.ylim(min_dist - 1,  4 * (1 - min_dist))
    plt.axhline(0, color='black', linewidth=0.25)
    plt.savefig(metrics_folder + 'grad_h_' + plot_name)
    plt.legend(loc="upper left")
    plt.close(grad_h_fig)
    u_fig = plt.figure()
    u_ref, u_qp = [], []
    unique_frames = df.frame.unique()
    frames = sorted(unique_frames)
    for frame in frames:
        curr_df = df[(df.frame == frame)]
        u_ref.append(curr_df.at[curr_df.index[0], 'u_ref'])
        u_qp.append(curr_df.at[curr_df.index[0], 'u_qp'])
    plt.plot(frames, u_ref, color='r', linewidth=0.5, label='reference')
    plt.plot(frames, u_qp, color='b', linewidth=0.5, label='QP control')
    #plt.ylim(min_dist - 1,  4 * (1 - min_dist))
    plt.axhline(0, color='black', linewidth=0.25)
    plt.savefig(metrics_folder + 'control_' + plot_name)
    plt.legend(loc="upper left")
    plt.close(u_fig)
    '''lin_vel, ang_vel = [], []
    unique_frames = df[df.metaID == -1].frame.unique()
    frames = sorted(unique_frames)
    for frame in frames:
        curr_df = df[(df.frame == frame) & (df.metaID == -1)]
        lin_vel.append(curr_df.at[curr_df.index[0], 'linear_vel'])
        ang_vel.append(curr_df.at[curr_df.index[0], 'angular_vel'])
    plt.figure(2)
    plt.subplot(211)
    plt.plot(frames, lin_vel, color='r', linewidth=0.5, label=('linear velocity'))
    #plt.ylim(-.5, 6.5)
    plt.subplot(212)
    plt.plot(frames, ang_vel, color='b', linewidth=0.5, label=('angular velocity'))
    #plt.ylim(-.3, .3)
    plt.axhline(0, color='black', linewidth=0.25)
    plt.savefig(metrics_folder + 'uQP_' + plot_name)
    plt.close()'''