# -*- coding: utf-8 -*-
'''
Criterion for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 30th March 2017
'''

import pdb
import torch
import numpy as np
from helper import getCoef
from torch.autograd import Variable


def Gaussian2DLikelihood(outputs, targets, nodesPresent, pred_length):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution
    params:
    outputs: Torch variable containing tensor of shape seq_length x numNodes x output_size (25, 43, 5)
    targets: Torch variable containing tensor of shape seq_length x numNodes x input_size  (25, 43, 3) [x, y, type]
    nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame
    '''

    # Get the sequence length
    seq_length = outputs.size()[0]
    # Get the observed length
    obs_length = seq_length - pred_length

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = torch.pow((normx/sx), 2) + torch.pow((normy/sy), 2) - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - torch.pow(corr, 2)
    #if normx[0]-mux[0] > 10:
     #   pdb.set_trace()
    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom
    #pdb.set_trace()
    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))

    # Compute the loss across all frames and all nodes
    loss = 0
    counter = 0

    for framenum in range(obs_length, seq_length):
        nodeIDs = nodesPresent[framenum]

        for nodeID in nodeIDs:

            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    #print('loss: {}'.format(loss.data[0]/counter))
    #if (loss <0).all():
    #    pdb.set_trace()
    if counter != 0:
        #if (loss/counter > 40).all():
        #    pdb.set_trace()
        return loss / counter
    else:
        return loss


def Gaussian2DLikelihoodInference(outputs, targets, assumedNodesPresent, nodesPresent):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution at test time
    params:
    outputs : predicted locations
    targets : true locations
    assumedNodesPresent : Nodes assumed to be present in each frame in the sequence
    nodesPresent : True nodes present in each frame in the sequence
    '''
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    # Compute the loss
    loss = Variable(torch.zeros(1).cuda())
    counter = 0

    for framenum in range(outputs.size()[0]):
        nodeIDs = nodesPresent[framenum]

        for nodeID in nodeIDs:
            if nodeID not in assumedNodesPresent:
                # If the node wasn't assumed to be present, don't compute loss for it
                continue
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss
