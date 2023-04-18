# -*- coding: utf-8 -*-
'''
Train script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 29th March 2017
'''

import argparse
import os
import pickle
import time
import pdb
import torch
from torch.autograd import Variable

from utils2 import DataLoader
from st_graph import ST_GRAPH
from model import SRNN
from criterion import Gaussian2DLikelihood
from utils2 import init_weights
# torch.backends.cudnn.deterministic = True
#
# torch.manual_seed(121)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(121)

def main():
    parser = argparse.ArgumentParser()

    # RNN size
    parser.add_argument('--human_node_rnn_size', type=int, default=128,
                        help='Size of Human Node RNN hidden state')
    parser.add_argument('--human_human_edge_rnn_size', type=int, default=256,
                        help='Size of Human Human Edge RNN hidden state')

    # Input and output size
    parser.add_argument('--human_node_input_size', type=int, default=3,
                        help='Dimension of the node features')
    parser.add_argument('--human_human_edge_input_size', type=int, default=3,
                        help='Dimension of the edge features')
    parser.add_argument('--human_node_output_size', type=int, default=5,
                        help='Dimension of the node output')

    # Embedding size
    parser.add_argument('--human_node_embedding_size', type=int, default=64,
                        help='Embedding size of node features')
    parser.add_argument('--human_human_edge_embedding_size', type=int, default=64,
                        help='Embedding size of edge features')

    # Attention vector dimension
    parser.add_argument('--attention_size', type=int, default=64,
                        help='Attention size')

    # Sequence length
    parser.add_argument('--seq_length', type=int, default=12,
                        help='Sequence length')
    parser.add_argument('--pred_length', type=int, default=7,
                        help='Predicted sequence length')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')

    parser.add_argument('--resume', type=int, default=0, help='resume')
    # Number of epochs


    parser.add_argument('--epoch', type=int, default=192, help='epoch to resume')

    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')

    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.00005,
                        help='L2 regularization parameter')

    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='decay rate for the optimizer')

    # Dropout rate
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout probability')

    parser.add_argument('--flag', type=str, default="",
                        help='Dropout probability')

    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=5,
                        help='The dataset index to be left out in training')

    args = parser.parse_args()

    train(args)


def trainFromEpoch(net, epoch_path):
    #pdb.set_trace()
    if os.path.isfile(epoch_path):
        print('loading checkpoint')
        checkpoint = torch.load(epoch_path)
        model_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print('Load checkpoint at {}'.format(model_epoch))
    else:
        print "No checkpoint found !!!!!!!!!"
    return net


def train(args):
    datasets = [0,1,2]
    # Remove the leave out dataset from the datasets
    #datasets.remove(args.leaveDataset)
    # datasets = [0]
    # args.leaveDataset = 0

    # Construct the DataLoader object
    dataloader = DataLoader(args.batch_size, args.seq_length + 1, datasets, forcePreProcess=False)
    test_dataloader = DataLoader(args.batch_size, args.seq_length + 1, datasets, True, False)




    #pdb.set_trace()
    # Construct the ST-graph object
    stgraph = ST_GRAPH(1, args.seq_length + 1)
    stgraph_test = ST_GRAPH(1, args.seq_length + 1)

    # Log directory
    log_directory = 'log/'+str(args.flag)+'/'
    log_directory += str(args.leaveDataset)+'/'
    log_directory += 'log_attention'
    if os.path.exists(log_directory) == False:
        os.makedirs(log_directory)

    # Logging file
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Save directory
    save_directory = 'save/'+str(args.flag)+'/'
    save_directory += str(args.leaveDataset)+'/'
    save_directory += 'save_attention'
    if os.path.exists(save_directory) == False:
        os.makedirs(save_directory)
    #pdb.set_trace()
    # Open the configuration file
    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, 'srnn_model_'+str(x)+'.tar')

    # Initialize net
    net = SRNN(args)
    net.cuda()
    init_weights(net, lstm='kaiming')
    start_epoch = 0
    if args.resume == 1:
        net = trainFromEpoch(net,checkpoint_path(args.epoch))
        start_epoch = args.epoch
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, momentum=0.0001, centered=True)
    # optimizer = torch.optim.Adagrad(net.parameters(), lr=args.learning_rate)


    learning_rate = args.learning_rate
    print('Training begin')
    best_val_loss = 0
    best_epoch = 0

    # Training
    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # For each batch
        for batch in range(dataloader.num_batches):
            start = time.time()
            # Get batch data
            x, _, d = dataloader.next_batch(randomUpdate=True)
            #pdb.set_trace()
            # Loss for this batch
            loss_batch = 0

            # For each sequence in the batch
            for sequence in range(dataloader.batch_size):
                # Construct the graph for the current sequence
                stgraph.readGraph([x[sequence]])

                nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()
                #pdb.set_trace()
                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float()).cuda()
                edges = Variable(torch.from_numpy(edges).float()).cuda()

                # Define hidden states
                numNodes = nodes.size()[1]
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()

                cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()

                # Zero out the gradients
                # net.zero_grad()
                optimizer.zero_grad()

                # Forward prop
                outputs, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1], hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs)

                #pdb.set_trace()
                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
                loss_batch += loss.data[0]

                #pdb.set_trace()
                # Compute gradients
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

                # Reset the stgraph
                stgraph.reset()

            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

            print(
                '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                    args.num_epochs * dataloader.num_batches,
                                                                                    epoch,
                                                                                    loss_batch, end - start))
            #pdb.set_trace()
        # Compute loss for the entire epoch
        loss_epoch /= dataloader.num_batches
        # Log it
        log_file_curve.write(str(epoch)+','+str(loss_epoch)+',')

        # Validation
        test_dataloader.reset_batch_pointer()
        loss_epoch = 0

        for batch in range(test_dataloader.num_batches):
            # Get batch data
            x, _, d = test_dataloader.next_batch(randomUpdate=False)

            # Loss for this batch
            loss_batch = 0

            for sequence in range(test_dataloader.batch_size):
                stgraph_test.readGraph([x[sequence]])

                nodes, edges, nodesPresent, edgesPresent = stgraph_test.getSequence()

                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float()).cuda()
                edges = Variable(torch.from_numpy(edges).float()).cuda()

                # Define hidden states
                numNodes = nodes.size()[1]
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()
                cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()

                outputs, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1],
                                             hidden_states_node_RNNs, hidden_states_edge_RNNs,
                                             cell_states_node_RNNs, cell_states_edge_RNNs)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)

                loss_batch += loss.data[0]

                # Reset the stgraph
                stgraph_test.reset()

            loss_batch = loss_batch / test_dataloader.batch_size
            loss_epoch += loss_batch

        loss_epoch = loss_epoch / test_dataloader.num_batches

        # Update best validation loss until now
        if loss_epoch < best_val_loss:
            #print('Find best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))
            best_val_loss = loss_epoch
            best_epoch = epoch
            print('Find best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))

        # Record best epoch and best validation loss
        print('(epoch {}), valid_loss = {:.3f}'.format(epoch, loss_epoch))
        #print('Best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))
        # Log it
        log_file_curve.write(str(loss_epoch)+'\n')

        # Save the model after each epoch
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    # Record the best epoch and best validation loss overall
    print('Best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))
    # Log it
    log_file.write(str(best_epoch)+','+str(best_val_loss))

    # Close logging files
    log_file.close()
    log_file_curve.close()


if __name__ == '__main__':
    main()
