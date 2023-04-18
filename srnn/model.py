# -*- coding: utf-8 -*-
'''
The structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 16th March 2017
'''

import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import pdb
from utils2 import init_weights

class SuperNodeRNN(nn.Module):
    '''
    Class representing super Node RNNs in the st-graph
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(SuperNodeRNN, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_rnn_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_rnn_size
        self.edge_rnn_size = args.human_human_edge_rnn_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # Linear layer to embed edgeRNN hidden states
        self.edge_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)

        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)

    def forward(self, pos, h_temporal, h, c):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(pos)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

        # Concat both the embeddings
        h_edges = h_temporal
        h_edges_embedded = self.relu(self.edge_attention_embed(h_edges))
        h_edges_embedded = self.dropout(h_edges_embedded)

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), 1)

        # One-step of LSTM
        h_new, c_new = self.cell(concat_encoded, (h, c))

        # Get output
        out = self.output_linear(h_new)

        return out, h_new, c_new


class SuperSuperEdgeRNN(nn.Module):
    '''
    Class representing the Super-Super Edge RNN in the s-t graph
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(SuperSuperEdgeRNN, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.rnn_size = args.human_human_edge_rnn_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_rnn_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # The LSTM cell
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

    def forward(self, inp, h, c):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        h : hidden state of the current edgeRNN
        c : cell state of the current edgeRNN
        '''
        # Encode the input position

        encoded_input = self.encoder_linear(inp)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

        # One-step of LSTM
        h_new, c_new = self.cell(encoded_input, (h, c))

        return h_new, c_new


class SuperEdgeAttention(nn.Module):
    '''
    Class representing the attention module
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(SuperEdgeAttention, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size

        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer = nn.Linear(self.human_human_edge_rnn_size, self.attention_size)

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer = nn.Linear(self.human_human_edge_rnn_size, self.attention_size)

    def forward(self, h_temporal, h_spatials):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        class SRNN(nn.Module):
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        '''
        # Number of spatial edges
        num_edges = h_spatials.size()[0]

        # Embed the temporal edgeRNN hidden state
        temporal_embed = self.temporal_edge_layer(h_temporal)
        temporal_embed = temporal_embed.squeeze(0)

        # Embed the spatial edgeRNN hidden states
        spatial_embed = self.spatial_edge_layer(h_spatials)

        # Dot based attention
        attn = torch.mv(spatial_embed, temporal_embed)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)
        #pdb.set_trace()
        # Softmax
        attn = torch.nn.functional.softmax(attn)

        # Compute weighted value
        weighted_value = torch.mv(torch.t(h_spatials), attn)

        return weighted_value, attn


class HumanNodeRNN(nn.Module):
    '''
    Class representing human Node RNNs in the st-graph
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(HumanNodeRNN, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # Linear layer to embed edgeRNN hidden states
        self.edge_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size*2, self.embedding_size)

        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)
        # self.second_output_linear_ped = nn.Linear(self.rnn_size*2, self.output_size)
        # self.second_output_linear_bike = nn.Linear(self.rnn_size*2, self.output_size)
        self.second_output_linear = nn.Linear(self.rnn_size*2, self.output_size)



    def forward(self, pos, h_temporal, h_spatial_other, h, c):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(pos)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

        # Concat both the embeddings
        h_edges = torch.cat((h_temporal, h_spatial_other), 1)
        h_edges_embedded = self.relu(self.edge_attention_embed(h_edges))
        h_edges_embedded = self.dropout(h_edges_embedded)

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), 1)

        # One-step of LSTM
        h_new, c_new = self.cell(concat_encoded, (h, c))

        # Get output
        out = self.output_linear(h_new)

        return out, h_new, c_new

    def secondForward_old(self, ped, bike, car, first_h, second_h):
        '''
        Second forward pass for the model
        '''

        ped_id = torch.LongTensor([0]).cuda()
        bike_id = torch.LongTensor([1]).cuda()
        car_id = torch.LongTensor([2]).cuda()

        h_new = Variable(torch.zeros(len(first_h), self.rnn_size * 2).cuda())
        for p in range(len(ped)):
            h_new[ped[p]] = torch.cat((first_h[ped[p]], second_h[ped_id]), 1)

        for b in range(len(bike)):
            h_new[bike[b]] = torch.cat((first_h[bike[b]], second_h[bike_id]), 1)

        for c in range(len(car)):
            h_new[car[c]] = torch.cat((first_h[car[c]], second_h[car_id]), 1)

        second_out = self.second_output_linear(h_new)

        return second_out

    def secondForward_old2(self, ped, bike, car, h_node_ped, h_node_bike, h_node_car, second_h):
        '''
        Second forward pass for the model
        '''

        ped_id = torch.LongTensor([0]).cuda()
        bike_id = torch.LongTensor([1]).cuda()
        car_id = torch.LongTensor([2]).cuda()

        second_out_ped = None
        second_out_bike = None
        second_out_car = None

        if h_node_ped:
            h_new_ped = Variable(torch.zeros(len(h_node_ped), self.rnn_size*2).cuda())
            assert len(h_node_ped) == len(ped), ("second Forward: h_node_ped mismatch ped")
            for p in range(len(ped)):
                h_new_ped[ped[p]] = torch.cat((h_node_ped[ped[p]], second_h[ped_id]), 1)

            second_out_ped = self.second_output_linear_ped(h_new_ped)

        if h_node_bike:
            h_new_bike = Variable(torch.zeros(len(h_node_bike), self.rnn_size*2).cuda())
            for b in range(len(bike)):
                h_new_bike[bike[b]] = torch.cat((h_node_bike[bike[b]], second_h[bike_id]), 1)

            second_out_bike = self.second_output_linear_bike(h_new_bike)

        if h_node_car:
            h_new_car = Variable(torch.zeros(len(h_node_car), self.rnn_size*2).cuda())
            for c in range(len(car)):
                h_new_car[car[c]] = torch.cat((h_node_car[car[c]], second_h[car_id]), 1)

            second_out_car = self.second_output_linear_car(h_new_car)

        # for p in range(len(ped)):
        #     h_new[ped[p]] = torch.cat((first_h[ped[p]], second_h[ped_id]), 1)
        #
        # for b in range(len(bike)):
        #     h_new[bike[b]] = torch.cat((first_h[bike[b]], second_h[bike_id]), 1)
        #
        # for c in range(len(car)):
        #     h_new[car[c]] = torch.cat((first_h[car[c]], second_h[car_id]), 1)


        # second_out = self.second_output_linear(h_new)

        return second_out_ped, second_out_bike, second_out_car

    def secondForward(self, ped, h_node_ped, second_h, ped_id1):
        '''
        Second forward pass for the model
        '''

        # ped_id = torch.LongTensor([ped_id1]).cuda()
        # bike_id = torch.LongTensor([1]).cuda()
        # car_id = torch.LongTensor([2]).cuda()

        # second_out_ped = None
        # second_out_bike = None
        # second_out_car = None
        second_out_ped = []
        if len(h_node_ped) > 0:
            h_new_ped = Variable(torch.zeros(len(h_node_ped), self.rnn_size*2).cuda())
            assert len(h_node_ped) == len(ped), ("second Forward: h_node_ped mismatch ped")
            for p in range(len(ped)):
                h_new_ped[p] = torch.cat((h_node_ped[p], second_h[ped_id1]), 0)

            second_out_ped = self.second_output_linear(h_new_ped)

        # if h_node_bike:
        #     h_new_bike = Variable(torch.zeros(len(h_node_bike), self.rnn_size*2).cuda())
        #     for b in range(len(bike)):
        #         h_new_bike[bike[b]] = torch.cat((h_node_bike[bike[b]], second_h[bike_id]), 1)
        #
        #     second_out_bike = self.second_output_linear_bike(h_new_bike)
        #
        # if h_node_car:
        #     h_new_car = Variable(torch.zeros(len(h_node_car), self.rnn_size*2).cuda())
        #     for c in range(len(car)):
        #         h_new_car[car[c]] = torch.cat((h_node_car[car[c]], second_h[car_id]), 1)
        #
        #     second_out_car = self.second_output_linear_car(h_new_car)

        # for p in range(len(ped)):
        #     h_new[ped[p]] = torch.cat((first_h[ped[p]], second_h[ped_id]), 1)
        #
        # for b in range(len(bike)):
        #     h_new[bike[b]] = torch.cat((first_h[bike[b]], second_h[bike_id]), 1)
        #
        # for c in range(len(car)):
        #     h_new[car[c]] = torch.cat((first_h[car[c]], second_h[car_id]), 1)


        # second_out = self.second_output_linear(h_new)

        return second_out_ped



class HumanHumanEdgeRNN(nn.Module):
    '''
    Class representing the Human-Human Edge RNN in the s-t graph
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(HumanHumanEdgeRNN, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.rnn_size = args.human_human_edge_rnn_size
        self.embedding_size = args.human_human_edge_embedding_size
        self.input_size = args.human_human_edge_input_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # The LSTM cell
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

    def forward(self, inp, h, c):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        h : hidden state of the current edgeRNN
        c : cell state of the current edgeRNN
        '''
        # Encode the input position
        #pdb.set_trace()
        encoded_input = self.encoder_linear(inp)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

        # One-step of LSTM
        h_new, c_new = self.cell(encoded_input, (h, c))

        return h_new, c_new





class EdgeAttention(nn.Module):
    '''
    Class representing the attention module
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EdgeAttention, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size

        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer = nn.Linear(self.human_human_edge_rnn_size, self.attention_size)

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer = nn.Linear(self.human_human_edge_rnn_size, self.attention_size)

    def forward(self, h_temporal, h_spatials):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        '''
        # Number of spatial edges
        num_edges = h_spatials.size()[0]

        # Embed the temporal edgeRNN hidden state
        temporal_embed = self.temporal_edge_layer(h_temporal)
        temporal_embed = temporal_embed.squeeze(0)
        #pdb.set_trace()
        # Embed the spatial edgeRNN hidden states
        spatial_embed = self.spatial_edge_layer(h_spatials)

        # Dot based attention
        attn = torch.mv(spatial_embed, temporal_embed)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)
        # Softmax
        attn = torch.nn.functional.softmax(attn)

        # Compute weighted value
        weighted_value = torch.mv(torch.t(h_spatials), attn)

        return weighted_value, attn


class Super_SRNN(nn.Module):
    '''
    Class representing the Super_SRNN model
    0: pre
    1: bike
    2: car

    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(Super_SRNN, self).__init__()

        self.args = args
        self.infer = infer

        if self.infer:
            # Test time
            self.seq_length = 1
            self.obs_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length
            self.obs_length = args.seq_length - args.pred_length

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_rnn_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN_ped = SuperNodeRNN(args, infer)
        self.humanNodeRNN_car = SuperNodeRNN(args, infer)
        self.humanNodeRNN_bike = SuperNodeRNN(args, infer)


        # self.humanhumanEdgeRNN_spatial = SuperSuperEdgeRNN(args, infer)
        self.humanhumanEdgeRNN_temporal_ped = SuperSuperEdgeRNN(args, infer)
        self.humanhumanEdgeRNN_temporal_car = SuperSuperEdgeRNN(args, infer)
        self.humanhumanEdgeRNN_temporal_bike = SuperSuperEdgeRNN(args, infer)



        # Initialize attention module
        # self.attn = SuperEdgeAttention(args, infer)
        self.superNode_num = 3
        self.superEdge_num = 9

        # Softmax
        self.sm = nn.Softmax()

    def get_Super_Feature(self,super_h, super_c):
        """
        :param super_h: average the super_h
        :param super_c: score the cell state by using softmax and multiply the current_h
        :return:
        """
        #make sure len(super_h)>0

        length = len(super_h)
        sum_super_h = Variable(torch.zeros(len(super_h[0])).cuda())
        for i in range(length):
            current_c = super_c[i]
            current_h = super_h[i]
            current_c = current_c.unsqueeze(0)
            current_c = self.sm(current_c)
            current_c = current_c.squeeze(0)
            current_h = torch.mul(current_c, current_h)
            sum_super_h += current_h

        sum_super_h = torch.div(sum_super_h, length)

        return sum_super_h

    def get_Super_Feature2(self, super_h, super_c):
        # average the super_h directly
        """
        :param super_h: (num_of_class, 128)
        :param super_c: (num_of_class, 128)
        :return: super_h_mean: (128)
        """
        return torch.mean(super_h, 0)




    def set_Super_Graph(self,super_ped_h, super_ped_c, super_bike_h, super_bike_c, super_car_h, super_car_c):
        #compute feature for each kind of super node
        superNode_ids = []
        # superEdge_ids = []
        superNodes = Variable(torch.zeros(self.superNode_num,self.human_node_rnn_size).cuda()) # 3 * 128
        superEdges = Variable(torch.zeros(self.superNode_num,self.human_node_rnn_size).cuda()) # 3 * 128

        #print(len(super_ped_h), len(super_bike_h), len(super_car_h))

        if len(super_ped_h)>0:
            super_ped_feature = self.get_Super_Feature(super_ped_h, super_ped_c)
            superNode_ids.append(0)
            superNodes[0,:] = super_ped_feature
        if len(super_bike_h)>0:
            super_bike_feature = self.get_Super_Feature(super_bike_h, super_bike_c)
            superNode_ids.append(1)
            superNodes[1,:] = super_bike_feature
        if len(super_car_h)>0:
            super_car_feature = self.get_Super_Feature(super_car_h, super_car_c)
            superNode_ids.append(2)
            superNodes[2,:] = super_car_feature

        # for i in range(len(superNode_ids)-1):
        #     last_id = superNode_ids[i]
        #     for j in range(i+1, len(superNode_ids)):
        #         next_id = superNode_ids[j]
        #         superEdge_ids.append((i,j))
        #         superEdge_ids.append((j,i))
        #         superEdges[i*3+j,:] = superNodes[i] - superNodes[j]
        #         superEdges[j*3+i,:] = superNodes[j] - superNodes[i]

        return superNodes, superEdges, superNode_ids


    def forward(self, nodes, edges, nodesPresent, edgesPresent, hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs):
        """

        :param nodes: superNodes features: (3, 128);
        :param edges: superEdges features: (3, 128); 3 means the number of SuperEdges, 0 for ped; 1 for bike; 2 for car
        :param nodesPresent: superNode_ids: (0, 1, 2), maybe there exists numbers less than 3
        :param edgesPresent: superEdge_ids: (0,0), (1,1), (2,2); 0 for ped, 1 for bike, 2 for car; maybe edges less than 3
        :param hidden_states_node_RNNs:
        :param hidden_states_edge_RNNs:
        :param cell_states_node_RNNs:
        :param cell_states_edge_RNNs:
        :return:
        """
        #pdb.set_trace()
        # Get number of nodes
        numNodes = len(nodesPresent)

        # Initialize output array
        outputs = Variable(torch.zeros(self.superNode_num, self.output_size)).cuda()

        # Data structure to store attention weights
        # attn_weights = {}

        # For each frame
        for framenum in range(1):
            # Find the edges present in the current frame
            edgeIDs = edgesPresent

            # Separate temporal and spatial edges
            temporal_edges = [x for x in edgeIDs if x[0] == x[1]]
            # spatial_edges = [x for x in edgeIDs if x[0] != x[1]]

            # Find the nodes present in the current frame
            nodeIDs = nodesPresent

            # Get features of the nodes and edges present
            nodes_current = nodes
            edges_current = edges

            # Initialize temporary tensors
            hidden_states_nodes_from_edges_temporal = Variable(torch.zeros(self.superNode_num, self.human_human_edge_rnn_size).cuda())
            # hidden_states_nodes_from_edges_spatial = Variable(torch.zeros(numNodes, self.human_human_edge_rnn_size).cuda())

            # If there are any edges
            if len(edgeIDs) != 0:

                # Temporal Edges
                if len(temporal_edges) != 0:

                    ############################## for the pre ######################################################

                    # Get the temporal edges
                    list_of_temporal_edges = torch.LongTensor([0L])
                    # Get nodes associated with the temporal edges
                    list_of_temporal_nodes = torch.LongTensor([0L])
                    ped_id = 0
                    if ped_id in nodeIDs:

                        # Get the corresponding edge features
                        edges_temporal_start_end = edges_current[0:1] # (1, 128)
                        # Get the corresponding hidden states
                        hidden_temporal_start_end = hidden_states_edge_RNNs[0:1] # (1, 256)
                        # Get the corresponding cell states
                        cell_temporal_start_end = cell_states_edge_RNNs[0:1] # (1, 256)

                        # Do forward pass through temporaledgeRNN
                        h_temporal_ped, c_temporal_ped = self.humanhumanEdgeRNN_temporal_ped.forward(
                            edges_temporal_start_end, hidden_temporal_start_end, cell_temporal_start_end)

                        # Update the hidden state and cell state
                        hidden_states_edge_RNNs[0].data = h_temporal_ped.data
                        cell_states_edge_RNNs[0].data = c_temporal_ped.data

                        # Store the temporal hidden states obtained in the temporary tensor
                        hidden_states_nodes_from_edges_temporal[0] = h_temporal_ped


                    ######################### for bike #######################################
                    bike_id = 1
                    if bike_id in nodeIDs:
                        h_temporal_bike, c_temporal_bike = self.humanhumanEdgeRNN_temporal_bike.forward(
                            edges_current[1:2], hidden_states_edge_RNNs[1:2], cell_states_edge_RNNs[1:2])

                        # Update the hidden state and cell state
                        hidden_states_edge_RNNs[1].data = h_temporal_bike.data
                        cell_states_edge_RNNs[1].data = c_temporal_bike.data

                        # Store the temporal hidden states obtained in the temporary tensor
                        hidden_states_nodes_from_edges_temporal[1] = h_temporal_bike


                    ########################## for car #######################################

                    car_id = 2
                    if car_id in nodeIDs:

                        h_temporal_car, c_temporal_car = self.humanhumanEdgeRNN_temporal_car.forward(
                            edges_current[2:3], hidden_states_edge_RNNs[2:3], cell_states_edge_RNNs[2:3])

                        # Update the hidden state and cell state
                        hidden_states_edge_RNNs[2].data = h_temporal_car.data
                        cell_states_edge_RNNs[2].data = c_temporal_car.data

                        # Store the temporal hidden states obtained in the temporary tensor
                        hidden_states_nodes_from_edges_temporal[2] = h_temporal_car

                # Spatial Edges
                # if len(spatial_edges) != 0:
                #     # Get the spatial edges
                #     list_of_spatial_edges = Variable(torch.LongTensor([x[0]*numNodes + x[1] for x in edgeIDs if x[0] != x[1]]).cuda())
                #     # Get nodes associated with the spatial edges
                #     list_of_spatial_nodes = np.array([x[0] for x in edgeIDs if x[0] != x[1]])
                #
                #     # Get the corresponding edge features
                #     edges_spatial_start_end = torch.index_select(edges_current, 0, list_of_spatial_edges)
                #     # Get the corresponding hidden states
                #     hidden_spatial_start_end = torch.index_select(hidden_states_edge_RNNs, 0, list_of_spatial_edges)
                #     # Get the corresponding cell states
                #     cell_spatial_start_end = torch.index_select(cell_states_edge_RNNs, 0, list_of_spatial_edges)
                #
                #     # Do forward pass through spatialedgeRNN
                #     h_spatial, c_spatial = self.humanhumanEdgeRNN_spatial.forward(edges_spatial_start_end, hidden_spatial_start_end, cell_spatial_start_end)
                #
                #     # Update the hidden state and cell state
                #     hidden_states_edge_RNNs[list_of_spatial_edges.data] = h_spatial
                #     cell_states_edge_RNNs[list_of_spatial_edges.data] = c_spatial
                #
                #
                #     # pass it to attention module
                #     # For each node
                #     for node in range(numNodes):
                #         # Get the indices of spatial edges associated with this node
                #         l = np.where(list_of_spatial_nodes == node)[0]
                #         if len(l) == 0:
                #             # If the node has no spatial edges, nothing to do
                #             continue
                #
                #         l = torch.LongTensor(l).cuda()
                #         # What are the other nodes with these edges?
                #         node_others = [x[1] for x in edgeIDs if x[0] == node and x[0] != x[1]]
                #         # If it has spatial edges
                #         # Get its corresponding temporal edgeRNN hidden state
                #         h_node = hidden_states_nodes_from_edges_temporal[node]
                #
                #         # Do forward pass through attention module
                #         hidden_attn_weighted, attn_w = self.attn.forward(h_node.view(1, -1), h_spatial[l])
                #
                #         # Store the attention weights
                #         attn_weights[node] = (attn_w.data.cpu().numpy(), node_others)
                #
                #         # Store the output of attention module in temporary tensor
                #         hidden_states_nodes_from_edges_spatial[node] = hidden_attn_weighted

            # If there are nodes in this frame
            if len(nodeIDs) != 0:

                # Get list of nodes
                ############################ for ped ###########################################
                # list_of_nodes = Variable(torch.LongTensor(nodeIDs).cuda()) # [0, 1, 2]

                ped_id = 0
                if ped_id in nodeIDs:

                    nodes_current_selected_ped = nodes_current[0:1] # nodes_current: (3, 128)


                    # Get the hidden and cell states of the corresponding nodes
                    hidden_nodes_current_ped = hidden_states_node_RNNs[0:1] # hidden_states_node_RNNs: (3, 128)
                    cell_nodes_current_ped = cell_states_node_RNNs[0:1]     # cell_states_node_RNNs : (3, 128)

                    # Get the temporal edgeRNN hidden states corresponding to these nodes
                    h_temporal_other_ped = hidden_states_nodes_from_edges_temporal[0:1]
                    # h_spatial_other = hidden_states_nodes_from_edges_spatial[list_of_nodes.data]

                    # Do a forward pass through nodeRNN
                    outputs[0], h_nodes_ped, c_nodes_ped = self.humanNodeRNN_ped.forward(nodes_current_selected_ped, h_temporal_other_ped, hidden_nodes_current_ped, cell_nodes_current_ped)

                # Update the hidden and cell states
                    hidden_states_node_RNNs[0].data = h_nodes_ped.data
                    cell_states_node_RNNs[0].data = c_nodes_ped.data


                ############################ for bike ###########################################

                bike_id = 1
                if bike_id in nodeIDs:
                    nodes_current_selected_bike = nodes_current[1:2]  # nodes_current: (3, 128)

                    # Get the hidden and cell states of the corresponding nodes
                    hidden_nodes_current_bike = hidden_states_node_RNNs[1:2]  # hidden_states_node_RNNs: (3, 128)
                    cell_nodes_current_bike= cell_states_node_RNNs[1:2]  # cell_states_node_RNNs : (3, 128)

                    # Get the temporal edgeRNN hidden states corresponding to these nodes
                    h_temporal_other_bike = hidden_states_nodes_from_edges_temporal[1:2]
                    # h_spatial_other = hidden_states_nodes_from_edges_spatial[list_of_nodes.data]

                    # Do a forward pass through nodeRNN
                    outputs[1], h_nodes_bike, c_nodes_bike = self.humanNodeRNN_bike.forward(nodes_current_selected_bike,
                                                                                         h_temporal_other_bike,
                                                                                         hidden_nodes_current_bike,
                                                                                         cell_nodes_current_bike)

                    # Update the hidden and cell states
                    hidden_states_node_RNNs[1].data = h_nodes_bike.data
                    cell_states_node_RNNs[1].data = c_nodes_bike.data

                ############################ for car ###########################################
                car_id = 2
                if car_id in nodeIDs:
                    nodes_current_selected_car = nodes_current[2:3]  # nodes_current: (3, 128)

                    # Get the hidden and cell states of the corresponding nodes
                    hidden_nodes_current_car = hidden_states_node_RNNs[2:3]  # hidden_states_node_RNNs: (3, 128)
                    cell_nodes_current_car = cell_states_node_RNNs[2:3]  # cell_states_node_RNNs : (3, 128)

                    # Get the temporal edgeRNN hidden states corresponding to these nodes
                    h_temporal_other_car = hidden_states_nodes_from_edges_temporal[2:3]
                    # h_spatial_other = hidden_states_nodes_from_edges_spatial[list_of_nodes.data]

                    # Do a forward pass through nodeRNN
                    outputs[2], h_nodes_car, c_nodes_car = self.humanNodeRNN_car.forward(nodes_current_selected_car,
                                                                                            h_temporal_other_car,
                                                                                            hidden_nodes_current_car,
                                                                                            cell_nodes_current_car)

                # Update the hidden and cell states
                    hidden_states_node_RNNs[2].data = h_nodes_car.data
                    cell_states_node_RNNs[2].data = c_nodes_car.data


        # Reshape the outputs carefully
        # outputs_return = Variable(torch.zeros(numNodes, self.output_size).cuda())
        # for node in range(numNodes):
        #     outputs_return[node, :] = outputs[node, :]

        return outputs, hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs





class SRNN(nn.Module):
    '''
    Class representing the SRNN model

    0: ped
    1: bike
    2: car

    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(SRNN, self).__init__()

        self.args = args
        self.infer = infer

        if self.infer:
            # Test time
            self.seq_length = 1
            self.obs_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length
            self.obs_length = args.seq_length - args.pred_length

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN_ped = HumanNodeRNN(args, infer)
        self.humanNodeRNN_bike = HumanNodeRNN(args, infer)
        self.humanNodeRNN_car = HumanNodeRNN(args, infer)

        self.humanhumanEdgeRNN_spatial = HumanHumanEdgeRNN(args, infer)

        self.humanhumanEdgeRNN_temporal_ped = HumanHumanEdgeRNN(args, infer)
        self.humanhumanEdgeRNN_temporal_bike = HumanHumanEdgeRNN(args, infer)
        self.humanhumanEdgeRNN_temporal_car = HumanHumanEdgeRNN(args, infer)


        self.superSRNN = Super_SRNN(args, infer)

        self.superNode_num = 3
        self.superEdge_num = 9

        # Initialize attention module
        self.attn = EdgeAttention(args, infer)

    def get_category(self, current_type):
        if current_type > 0.55 and current_type < 0.65:
            return 0
        elif current_type > 0.75 and current_type < 0.85:
            return 1
        elif current_type > 0.95 and current_type < 1.05:
            return 2
        return None

    def forward(self, nodes, edges, nodesPresent, edgesPresent, hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs):
        '''
        Forward pass for the model
        params:
        nodes : input node features
        edges : input edge features
        nodesPresent : A list of lists, of size seq_length
        Each list contains the nodeIDs that are present in the frame
        edgesPresent : A list of lists, of size seq_length
        Each list contains tuples of nodeIDs that have edges in the frame
        hidden_states_node_RNNs : A tensor of size numNodes x node_rnn_size
        Contains hidden states of the node RNNs
        hidden_states_edge_RNNs : A tensor of size numNodes x numNodes x edge_rnn_size
        Contains hidden states of the edge RNNs

        returns:
        outputs : A tensor of shape seq_length x numNodes x 5
        Contains the predictions for next time-step
        hidden_states_node_RNNs
        hidden_states_edge_RNNs
        '''

        hidden_states_superNode_RNNs = Variable(torch.zeros(self.superNode_num, self.human_node_rnn_size).cuda())
        hidden_states_superEdge_RNNs = Variable(torch.zeros(self.superEdge_num, self.human_human_edge_rnn_size).cuda())
        cell_states_superNode_RNNs = Variable(torch.zeros(self.superNode_num, self.human_node_rnn_size).cuda())
        cell_states_superEdge_RNNs = Variable(torch.zeros(self.superEdge_num, self.human_human_edge_rnn_size).cuda())


        # Get number of nodes
        numNodes = nodes.size()[1]

        # Initialize output array
        outputs = Variable(torch.zeros(self.seq_length*numNodes, self.output_size)).cuda()

        # Data structure to store attention weights
        attn_weights = [{} for _ in range(self.seq_length)]


        # For each frame
        for framenum in range(self.seq_length):
            # Find the edges present in the current frame
            edgeIDs = edgesPresent[framenum] # the edge consisting of  current frames

            # Separate temporal and spatial edges
            temporal_edges = [x for x in edgeIDs if x[0] == x[1]]
            spatial_edges = [x for x in edgeIDs if x[0] != x[1]]

            # temporal_edges_ped =

            # Find the nodes present in the current frame
            nodeIDs = nodesPresent[framenum]

            # Get features of the nodes and edges present
            nodes_current = nodes[framenum]
            edges_current = edges[framenum]


            ins_Node_ped = []
            ins_Node_bike = []
            ins_Node_car = []
            if len(nodeIDs):
                list_of_nodes = Variable(torch.LongTensor(nodeIDs).cuda())# 获取当前帧中的物体个数

                nodes_current_selected = torch.index_select(nodes_current, 0, list_of_nodes) # nodes_current: 包含了所有的instance, 这里有43个

                for instance in range(len(nodes_current_selected)):
                    # pdb.set_trace()
                    current_instance = nodes_current_selected[instance]
                    current_type = current_instance[2].data.cpu()[0]
                    # print('{}'.format(nodes_current_selected[instance]))
                    if current_type > 0.55 and current_type < 0.65:
                        ins_Node_ped.append(instance)
                    elif current_type > 0.75 and current_type < 0.85:
                        ins_Node_bike.append(instance)
                    elif current_type > 0.95 and current_type < 1.05:
                        ins_Node_car.append(instance)

                temporal_edges_ped = []
                temporal_edges_bike = []
                temporal_edges_car = []

                if len(temporal_edges) > 0 and len(nodeIDs) > 0:
                    for temporal_edge_tmp in temporal_edges:
                        nodeid = temporal_edge_tmp[0]
                        list_of_nodes_data = list_of_nodes.data.cpu()
                        if nodeid in list_of_nodes_data:
                            current_node_instance = nodes_current[nodeid]
                            temporal_type = current_node_instance[2].data.cpu()[0]
                            if temporal_type > 0.55 and temporal_type < 0.65:
                                temporal_edges_ped.append(temporal_edge_tmp)
                            elif temporal_type > 0.75 and temporal_type < 0.85:
                                temporal_edges_bike.append(temporal_edge_tmp)
                            elif temporal_type > 0.95 and temporal_type < 1.05:
                                temporal_edges_car.append(temporal_edge_tmp)
                        else:
                            print("$$$$$$$$$$$Exists some edges which do not occur in this frame")




            # Initialize temporary tensors
            hidden_states_nodes_from_edges_temporal = Variable(torch.zeros(numNodes, self.human_human_edge_rnn_size).cuda())
            hidden_states_nodes_from_edges_spatial = Variable(torch.zeros(numNodes, self.human_human_edge_rnn_size).cuda())

            # If there are any edges
            if len(edgeIDs) != 0:

                # Temporal Edges
                if len(temporal_edges) != 0:
                    # Get the temporal edges
                    # list_of_temporal_edges = Variable(torch.LongTensor([x[0]*numNodes + x[0] for x in edgeIDs if x[0] == x[1]]).cuda())
                    # # Get nodes associated with the temporal edges
                    # list_of_temporal_nodes = torch.LongTensor([x[0] for x in edgeIDs if x[0] == x[1]]).cuda()
                    #
                    # # Get the corresponding edge features
                    # edges_temporal_start_end = torch.index_select(edges_current, 0, list_of_temporal_edges)
                    # # Get the corresponding hidden states
                    # hidden_temporal_start_end = torch.index_select(hidden_states_edge_RNNs, 0, list_of_temporal_edges)
                    # # Get the corresponding cell states
                    # cell_temporal_start_end = torch.index_select(cell_states_edge_RNNs, 0, list_of_temporal_edges)
                    #
                    # # Do forward pass through temporaledgeRNN
                    # h_temporal, c_temporal = self.humanhumanEdgeRNN_temporal.forward(edges_temporal_start_end, hidden_temporal_start_end, cell_temporal_start_end)
                    #
                    # # Update the hidden state and cell state
                    # hidden_states_edge_RNNs[list_of_temporal_edges.data] = h_temporal
                    # cell_states_edge_RNNs[list_of_temporal_edges.data] = c_temporal
                    #
                    # # Store the temporal hidden states obtained in the temporary tensor
                    # hidden_states_nodes_from_edges_temporal[list_of_temporal_nodes] = h_temporal

                    ###################### for the ped #########################

                    if len(temporal_edges_ped):
                        list_of_temporal_edges_ped = Variable(torch.LongTensor([x[0]*numNodes + x[0] for x in temporal_edges_ped]).cuda())
                        list_of_temporal_nodes_ped = torch.LongTensor([x[0] for x in temporal_edges_ped]).cuda()
                        edges_temporal_start_end_ped = torch.index_select(edges_current, 0, list_of_temporal_edges_ped)
                        hidden_temporal_start_end_ped = torch.index_select(hidden_states_edge_RNNs, 0, list_of_temporal_edges_ped)
                        cell_temporal_start_end_ped = torch.index_select(cell_states_edge_RNNs, 0, list_of_temporal_edges_ped)
                        h_temporal_ped, c_temporal_ped = self.humanhumanEdgeRNN_temporal_ped.forward(edges_temporal_start_end_ped,
                                                                                         hidden_temporal_start_end_ped,
                                                                                         cell_temporal_start_end_ped)

                        hidden_states_edge_RNNs[list_of_temporal_edges_ped.data] = h_temporal_ped
                        cell_states_edge_RNNs[list_of_temporal_edges_ped.data] = c_temporal_ped
                        hidden_states_nodes_from_edges_temporal[list_of_temporal_nodes_ped] = h_temporal_ped

                    if len(temporal_edges_bike):
                        list_of_temporal_edges_bike = Variable(torch.LongTensor([x[0]*numNodes + x[0] for x in temporal_edges_bike]).cuda())
                        list_of_temporal_nodes_bike = torch.LongTensor([x[0] for x in temporal_edges_bike]).cuda()
                        edges_temporal_start_end_bike = torch.index_select(edges_current, 0, list_of_temporal_edges_bike)
                        hidden_temporal_start_end_bike = torch.index_select(hidden_states_edge_RNNs, 0, list_of_temporal_edges_bike)
                        cell_temporal_start_end_bike = torch.index_select(cell_states_edge_RNNs, 0, list_of_temporal_edges_bike)
                        h_temporal_bike, c_temporal_bike = self.humanhumanEdgeRNN_temporal_bike.forward(edges_temporal_start_end_bike,
                                                                                         hidden_temporal_start_end_bike,
                                                                                         cell_temporal_start_end_bike)

                        hidden_states_edge_RNNs[list_of_temporal_edges_bike.data] = h_temporal_bike
                        cell_states_edge_RNNs[list_of_temporal_edges_bike.data] = c_temporal_bike
                        hidden_states_nodes_from_edges_temporal[list_of_temporal_nodes_bike] = h_temporal_bike

                    if len(temporal_edges_car):
                        list_of_temporal_edges_car = Variable(torch.LongTensor([x[0]*numNodes + x[0] for x in temporal_edges_car]).cuda())
                        list_of_temporal_nodes_car = torch.LongTensor([x[0] for x in temporal_edges_car]).cuda()
                        edges_temporal_start_end_car = torch.index_select(edges_current, 0, list_of_temporal_edges_car)
                        hidden_temporal_start_end_car = torch.index_select(hidden_states_edge_RNNs, 0, list_of_temporal_edges_car)
                        cell_temporal_start_end_car = torch.index_select(cell_states_edge_RNNs, 0, list_of_temporal_edges_car)
                        h_temporal_car, c_temporal_car = self.humanhumanEdgeRNN_temporal_car.forward(edges_temporal_start_end_car,
                                                                                         hidden_temporal_start_end_car,
                                                                                         cell_temporal_start_end_car)

                        hidden_states_edge_RNNs[list_of_temporal_edges_car.data] = h_temporal_car
                        cell_states_edge_RNNs[list_of_temporal_edges_car.data] = c_temporal_car
                        hidden_states_nodes_from_edges_temporal[list_of_temporal_nodes_car] = h_temporal_car


                # Spatial Edges
                if len(spatial_edges) != 0:
                    # Get the spatial edges
                    list_of_spatial_edges = Variable(torch.LongTensor([x[0]*numNodes + x[1] for x in edgeIDs if x[0] != x[1]]).cuda())
                    # Get nodes associated with the spatial edges
                    list_of_spatial_nodes = np.array([x[0] for x in edgeIDs if x[0] != x[1]])

                    # Get the corresponding edge features
                    edges_spatial_start_end = torch.index_select(edges_current, 0, list_of_spatial_edges)
                    # Get the corresponding hidden states
                    hidden_spatial_start_end = torch.index_select(hidden_states_edge_RNNs, 0, list_of_spatial_edges)
                    # Get the corresponding cell states
                    cell_spatial_start_end = torch.index_select(cell_states_edge_RNNs, 0, list_of_spatial_edges)

                    # Do forward pass through spatialedgeRNN
                    h_spatial, c_spatial = self.humanhumanEdgeRNN_spatial.forward(edges_spatial_start_end, hidden_spatial_start_end, cell_spatial_start_end)

                    # Update the hidden state and cell state
                    hidden_states_edge_RNNs[list_of_spatial_edges.data] = h_spatial
                    cell_states_edge_RNNs[list_of_spatial_edges.data] = c_spatial

                    # pass it to attention module
                    # For each node
                    for node in range(numNodes):
                        # Get the indices of spatial edges associated with this node

                        l = np.where(list_of_spatial_nodes == node)[0]
                        if len(l) == 0:
                            # If the node has no spatial edges, nothing to do
                            continue

                        l = torch.LongTensor(l).cuda()
                        # What are the other nodes with these edges?
                        node_others = [x[1] for x in edgeIDs if x[0] == node and x[0] != x[1]]
                        # If it has spatial edges
                        # Get its corresponding temporal edgeRNN hidden state
                        h_node = hidden_states_nodes_from_edges_temporal[node]

                        # Do forward pass through attention module
                        hidden_attn_weighted, attn_w = self.attn.forward(h_node.view(1, -1), h_spatial[l])

                        # Store the attention weights
                        attn_weights[framenum][node] = (attn_w.data.cpu().numpy(), node_others)

                        # Store the output of attention module in temporary tensor
                        hidden_states_nodes_from_edges_spatial[node] = hidden_attn_weighted

            # If there are nodes in this frame
            if len(nodeIDs) != 0:

                # Get list of nodes
                # list_of_nodes = Variable(torch.LongTensor(nodeIDs).cuda())# 获取当前帧中的物体个数
                #
                # nodes_current_selected = torch.index_select(nodes_current, 0, list_of_nodes) # nodes_current: 包含了所有的instance, 这里有43个
                #
                #
                # # Get the hidden and cell states of the corresponding nodes
                # hidden_nodes_current = torch.index_select(hidden_states_node_RNNs, 0, list_of_nodes)
                # cell_nodes_current = torch.index_select(cell_states_node_RNNs, 0, list_of_nodes)
                #
                # # Get the temporal edgeRNN hidden states corresponding to these nodes
                # h_temporal_other = hidden_states_nodes_from_edges_temporal[list_of_nodes.data]
                # h_spatial_other = hidden_states_nodes_from_edges_spatial[list_of_nodes.data]
                #
                # # Do a forward pass through nodeRNN
                # outputs[framenum * numNodes + list_of_nodes.data], h_nodes, c_nodes = self.humanNodeRNN.forward(nodes_current_selected, h_temporal_other, h_spatial_other, hidden_nodes_current, cell_nodes_current)
                # # h_nodes: (23, 128), 23指的是当前frame的物体个数
                #
                # # Update the hidden and cell states
                # hidden_states_node_RNNs[list_of_nodes.data] = h_nodes
                # cell_states_node_RNNs[list_of_nodes.data] = c_nodes

                ####################################### for ped #############################################
                h_nodes_car = []
                h_nodes_bike = []
                h_nodes_ped = []
                c_nodes_car = []
                c_nodes_ped = []
                c_nodes_bike = []

                if len(ins_Node_ped):
                    list_of_nodes_ped = Variable(torch.LongTensor(ins_Node_ped).cuda())
                    nodes_current_selected_ped = torch.index_select(nodes_current, 0, list_of_nodes_ped)
                    hidden_nodes_current_ped = torch.index_select(hidden_states_node_RNNs, 0, list_of_nodes_ped)
                    cell_nodes_current_ped = torch.index_select(cell_states_node_RNNs, 0, list_of_nodes_ped)
                    h_temporal_other_ped = hidden_states_nodes_from_edges_temporal[list_of_nodes_ped.data]
                    h_spatial_other_ped = hidden_states_nodes_from_edges_spatial[list_of_nodes_ped.data]

                    outputs_ped, h_nodes_ped, c_nodes_ped = self.humanNodeRNN_ped.forward(
                        nodes_current_selected_ped, h_temporal_other_ped, h_spatial_other_ped, hidden_nodes_current_ped,
                        cell_nodes_current_ped)


                ######################################## for bike ###########################################


                if len(ins_Node_bike):
                    list_of_nodes_bike = Variable(torch.LongTensor(ins_Node_bike).cuda())
                    nodes_current_selected_bike = torch.index_select(nodes_current, 0, list_of_nodes_bike)
                    hidden_nodes_current_bike = torch.index_select(hidden_states_node_RNNs, 0, list_of_nodes_bike)
                    cell_nodes_current_bike = torch.index_select(cell_states_node_RNNs, 0, list_of_nodes_bike)
                    h_temporal_other_bike = hidden_states_nodes_from_edges_temporal[list_of_nodes_bike.data]
                    h_spatial_other_bike = hidden_states_nodes_from_edges_spatial[list_of_nodes_bike.data]

                    outputs_bike, h_nodes_bike, c_nodes_bike = self.humanNodeRNN_bike.forward(
                        nodes_current_selected_bike, h_temporal_other_bike, h_spatial_other_bike, hidden_nodes_current_bike,
                        cell_nodes_current_bike)


                ######################################## for car #############################################

                if len(ins_Node_car):
                    list_of_nodes_car = Variable(torch.LongTensor(ins_Node_car).cuda())
                    nodes_current_selected_car = torch.index_select(nodes_current, 0, list_of_nodes_car)
                    hidden_nodes_current_car = torch.index_select(hidden_states_node_RNNs, 0, list_of_nodes_car)
                    cell_nodes_current_car = torch.index_select(cell_states_node_RNNs, 0, list_of_nodes_car)
                    h_temporal_other_car = hidden_states_nodes_from_edges_temporal[list_of_nodes_car.data]
                    h_spatial_other_car = hidden_states_nodes_from_edges_spatial[list_of_nodes_car.data]

                    outputs_car, h_nodes_car, c_nodes_car = self.humanNodeRNN_car.forward(
                        nodes_current_selected_car, h_temporal_other_car, h_spatial_other_car, hidden_nodes_current_car,
                        cell_nodes_current_car)





                # superNode_ped = []
                # superNode_bike = []
                # superNode_car = []
                # for instance in range(len(nodes_current_selected)):
                #     #pdb.set_trace()
                #     current_instance = nodes_current_selected[instance]
                #     current_type = current_instance[2].cpu().data[0]
                #     #print('{}'.format(nodes_current_selected[instance]))
                #     if current_type > 0.55 and current_type < 0.65:
                #         superNode_ped.append(instance)
                #     elif current_type > 0.75 and current_type < 0.85:
                #         superNode_bike.append(instance)
                #     elif current_type > 0.95 and current_type < 1.05:
                #         superNode_car.append(instance)

                #pdb.set_trace()
                superNode_ped = Variable(torch.LongTensor(ins_Node_ped).cuda()) # the number of ped in the frame
                superNode_bike = Variable(torch.LongTensor(ins_Node_bike).cuda()) # the number of bike in the frame
                superNode_car = Variable(torch.LongTensor(ins_Node_car).cuda()) # the number of car in the frame
                #print('{}:{}:{}:{}:{}'.format(len(nodes_current_selected), len(nodeIDs), len(superNode_ped), len(superNode_bike), len(superNode_car)))
                #if len(superNode_ped)==0:
                 #   pdb.set_trace()
                # super_ped_h = torch.index_select(h_nodes,0,superNode_ped)
                # super_ped_c = torch.index_select(c_nodes,0,superNode_ped)
                # super_bike_h = torch.index_select(h_nodes,0,superNode_bike)
                # super_bike_c = torch.index_select(c_nodes,0,superNode_bike)
                # super_car_h = torch.index_select(h_nodes,0,superNode_car)
                # super_car_c = torch.index_select(c_nodes,0,superNode_car)
                # super_ped_h: 1; super_bike_h: 1; super_car_h:19
                # superEdges = {}
                superEdge_ids = []
                superNodes, superEdges, superNode_ids = self.superSRNN.set_Super_Graph(h_nodes_ped, c_nodes_ped, h_nodes_bike, c_nodes_bike, h_nodes_car, c_nodes_car)

                if framenum > 0 :
                    for node_i in range(len(superNode_ids)):
                         current_node = superNode_ids[node_i]
                         if self.last_superNode_ids.count(current_node)>0 :
                             superEdge_ids.append((current_node,current_node))
                             superEdges[current_node] = superNodes[current_node] - self.last_superNodes[current_node]
                # superEdges Size(): 3 * 128 for Three superNodes
                self.last_superNodes = superNodes
                self.last_superNode_ids = superNode_ids

                superNodeOutputs,_,_,_,_ = self.superSRNN.forward(superNodes, superEdges, superNode_ids, superEdge_ids, hidden_states_superNode_RNNs, hidden_states_superEdge_RNNs, cell_states_superNode_RNNs, cell_states_superEdge_RNNs)




                """
                superNode_ped, bike, car: the index of instance in the frame
                h_nodes: [total_instance_in_current_frame, 128]
                superNodeOutputs: [3, 128]
                """
                if len(ins_Node_ped):
                    second_output_ped = self.humanNodeRNN_ped.secondForward(superNode_ped, h_nodes_ped,
                                                                            superNodeOutputs, 0L)
                    outputs[framenum * numNodes + list_of_nodes_ped.data] = second_output_ped

                if len(ins_Node_bike):
                    second_output_bike = self.humanNodeRNN_bike.secondForward(superNode_bike, h_nodes_bike,
                                                                              superNodeOutputs, 1L)
                    outputs[framenum * numNodes + list_of_nodes_bike.data] = second_output_bike

                if len(ins_Node_car):
                    second_output_car = self.humanNodeRNN_car.secondForward(superNode_car, h_nodes_car,
                                                                            superNodeOutputs, 2L)
                    outputs[framenum * numNodes + list_of_nodes_car.data] = second_output_car

        # Reshape the outputs carefully
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size).cuda())
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs, attn_weights
