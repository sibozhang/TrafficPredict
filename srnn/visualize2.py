# -*- coding: utf-8 -*-
'''
Visualization script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 3rd April 2017
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.autograd import Variable
import argparse
import os
# import seaborn
import pdb

def plot_trajectories(true_trajs, pred_trajs, nodesPresent, obs_length, name, plot_directory, withBackground=False):
    '''
    Parameters
    ==========

    true_trajs : Numpy matrix of shape seq_length x numNodes x 2
    Contains the true trajectories of the nodes

    pred_trajs : Numpy matrix of shape seq_length x numNodes x 2
    Contains the predicted trajectories of the nodes

    nodesPresent : A list of lists, of size seq_length
    Each list contains the nodeIDs present at that time-step

    obs_length : Length of observed trajectory

    name : Name of the plot

    withBackground : Include background or not
    '''
    plt.switch_backend('agg')
    traj_length, numNodes, _ = true_trajs.shape
    # Initialize figure
    # Load the background
    # im = plt.imread('plot/background.png')
    # if withBackground:
    #    implot = plt.imshow(im)

    # width_true = im.shape[0]
    # height_true = im.shape[1]

    # if withBackground:
    #    width = width_true
    #    height = height_true
    # else:
    width = 1
    height = 1

    traj_data = {}
    valid_traj_data = {}
    pred_start_point = {}
    for ped in range(numNodes):
         if ped in nodesPresent[obs_length-1]:
                valid_traj_data[ped] = [[],[]]



    for tstep in range(traj_length):
        pred_pos = pred_trajs[tstep, :]
        true_pos = true_trajs[tstep, :]

        #for ped in range(numNodes):
            #if ped not in traj_data and tstep < obs_length:
             #   traj_data[ped] = [[], []]

            #if ped in nodesPresent[tstep]:
             #   traj_data[ped][0].append(true_pos[ped, :])
             #   traj_data[ped][1].append(pred_pos[ped, :])

        for ped in valid_traj_data:
            #if tstep >= obs_length:
             #   valid_traj_data[ped][1].append(pred_pos[ped,:])
            if ped in nodesPresent[tstep]:
                valid_traj_data[ped][0].append(true_pos[ped,:])
                valid_traj_data[ped][1].append(pred_pos[ped,:])
                if tstep == obs_length-1:
                    pred_start_point[ped] = len(valid_traj_data[ped][0])


    #pdb.set_trace()
    #print('--------------------------')
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    count = 0
    for j in valid_traj_data:
        #c = np.random.rand(3)
        agent_type = valid_traj_data[j][0][0][2]
        #pdb.set_trace()
        if agent_type>0.9:
            c = np.array([0.1,0.2,0.9])
        elif agent_type>0.7:
            c = np.array([0.4,0.8,0.3])
        else:
            c = np.array([0.9,0.2, 0.1])

        obs_c = np.array([0.1,0.1,0.1])

        true_traj_ped = valid_traj_data[j][0]  # List of [x,y] elements
        pred_traj_ped = valid_traj_data[j][1]

        true_x = [p[0] for p in true_traj_ped]
        true_y = [p[1] for p in true_traj_ped]
        pred_x = [p[0] for p in pred_traj_ped]
        pred_y = [p[1] for p in pred_traj_ped]
        #pdb.set_trace()

        tempmax_x = max( max(pred_x),max(true_x))
        tempmax_y = max( max(pred_y),max(true_y))

        tempmin_x = min( min(pred_x),min(true_x))
        tempmin_y = min( min(pred_y),min(true_y))

        #print(j,min_x,true_x, pred_x)
        #print(j,min_y,true_y, pred_y)
        #pdb.set_trace()
        if count == 0:
            min_x = tempmin_x
            min_y = tempmin_y
            max_x = tempmax_x
            max_y = tempmax_y
        else:
            if tempmin_x<min_x:
                min_x = tempmin_x
            if tempmax_x>max_x:
                max_x =tempmax_x
            if tempmin_y<min_y:
                min_y = tempmin_y
            if tempmax_y>max_y:
                max_y = tempmax_y

        count += 1

        #pdb.set_trace()
        b_p = pred_start_point[j]

        #print('true, pred, b_p {}:{}:{}'.format(len(true_x), len(pred_x),b_p))


        plt.plot(true_x, true_y, color=c, linestyle='solid', linewidth=1, marker='.')
        plt.plot(pred_x, pred_y, color=c, linestyle='solid', linewidth=1, marker='+')
        plt.plot(true_x[0:b_p], true_y[0:b_p], color=obs_c, linestyle='solid', linewidth=1, marker='.')
        plt.plot(pred_x[0:b_p], pred_y[0:b_p], color=obs_c, linestyle='solid', linewidth=2, marker='+')

    if not withBackground:
        #print(min_x,max_x,min_y,max_y)
        plt.xlim((min_x-0.1, max_x+0.1))
        plt.ylim((min_y-0.1, max_y+0.1))



    plt.show()
    if withBackground:
        plt.savefig('plot_with_background/'+name+'.png')
    else:
        plt.savefig(plot_directory+'/'+name+'.png')

    plt.gcf().clear()
    # plt.close('all')
    plt.clf()


def main():
    parser = argparse.ArgumentParser()

    # Experiments
    parser.add_argument('--noedges', action='store_true')
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--temporal_spatial', action='store_true')
    parser.add_argument('--attention', action='store_true')

    parser.add_argument('--test_dataset', type=int, default=5,
                        help='test dataset index')
    parser.add_argument('--flag', type=str, default="",
                        help='Dropout probability')

    # Parse the parameters
    args = parser.parse_args()

    # Check experiment tags
    #if not (args.noedges or args.temporal or args.temporal_spatial or args.attention):
    #    print 'Use one of the experiment tags to enforce model'
    #    return

    # Save directory
    save_directory = 'save/' + str(args.flag) + '/'
    save_directory += str(args.test_dataset) + '/'
    plot_directory = 'plot/' + str(args.flag) + '/'
    if args.noedges:
        print 'No edge RNNs used'
        save_directory += 'save_noedges'
        plot_directory += 'plot_noedges'
    elif args.temporal:
        print 'Only temporal edge RNNs used'
        save_directory += 'save_temporal'
        plot_directory += 'plot_temporal'
    elif args.temporal_spatial:
        print 'Both temporal and spatial edge RNNs used'
        save_directory += 'save_temporal_spatial'
        plot_directory += 'plot_temporal_spatial'
    else:
        print 'Both temporal and spatial edge RNNs used with attention'
        save_directory += 'save_attention'
        plot_directory += 'plot_attention'

    f = open(save_directory+'/results.pkl', 'rb')
    results = pickle.load(f)

    # print "Enter 0 (or) 1 for without/with background"
    # withBackground = int(input())
    withBackground = 0
    if os.path.exists(plot_directory) == False:
        os.makedirs(plot_directory)

    for i in range(len(results)):
        #if i>50:
        #    break
        print i
        name = 'sequence' + str(i)
        plot_trajectories(results[i][0], results[i][1], results[i][2], results[i][3], name, plot_directory, withBackground)


if __name__ == '__main__':
    main()
