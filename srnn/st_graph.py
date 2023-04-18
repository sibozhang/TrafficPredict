# -*- coding: utf-8 -*-
'''
ST-graph data structure script for the structural RNN implementation
Takes a batch of sequences and generates corresponding ST-graphs

Author : Anirudh Vemula
Date : 15th March 2017
'''
import numpy as np
from helper import getVector, getMagnitudeAndDirection
import pdb
import operator

class ST_GRAPH():

    def __init__(self, batch_size=50, seq_length=5):
        '''
        Initializer function for the ST graph class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        '''
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.nodes = [{} for i in range(batch_size)]
        self.edges = [{} for i in range(batch_size)]

    def reset(self):
        self.nodes = [{} for i in range(self.batch_size)]
        self.edges = [{} for i in range(self.batch_size)]

    def readGraph(self, source_batch):
        '''
        Main function that constructs the ST graph from the batch data
        params:
        source_batch : List of lists of numpy arrays. Each numpy array corresponds to a frame in the sequence.
        '''
        for sequence in range(self.batch_size):
            # source_seq is a list of numpy arrays
            # where each numpy array corresponds to a single frame
            source_seq = source_batch[sequence]
            #pdb.set_trace()
            for framenum in range(self.seq_length):
                # Each frame is a numpy array
                # each row in the array is of the form
                # pedID, x, y
                frame = source_seq[framenum]
                #print('----------------{}'.format(framenum))
                # Add nodes
                for ped in range(frame.shape[0]):
                    pedID = frame[ped, 0]
                    t = frame[ped, 1]
                    x = frame[ped, 2]
                    y = frame[ped, 3]
                    z = frame[ped, 4]
                    v1 = frame[ped, 5]
                    v2 = frame[ped, 6]
                    v3 = frame[ped, 7]
                    width = frame[ped, 8]
                    length = frame[ped, 9]
                    heading = frame[ped, 10]
                    #pos = (t, x, y,z,v1,v2,v3,width,length, heading)
                    #print(t)
                    #if t >1:
                     #   pdb.set_trace()
                    pos = (x,y,t)
                    #print(pos)
                    if pedID not in self.nodes[sequence]:
                        node_type = t
                        node_id = pedID
                        node_pos_list = {}
                        node_pos_list[framenum] = pos
                        self.nodes[sequence][pedID] = ST_NODE(node_type, node_id, node_pos_list)
                    else:

                        self.nodes[sequence][pedID].addPosition(pos, framenum)

                        # Add Temporal edge between the node at current time-step
                        # and the node at previous time-step
                        edge_id = (pedID, pedID)
                        #pdb.set_trace()
                        pos_edge = (self.nodes[sequence][pedID].getLastPosition(framenum,pos), pos)
                        if edge_id not in self.edges[sequence]:
                            edge_type = 'H-H/T'
                            edge_pos_list = {}
                            # ASSUMPTION: Adding temporal edge at the later time-step
                            edge_pos_list[framenum] = pos_edge
                            self.edges[sequence][edge_id] = ST_EDGE(edge_type, edge_id, edge_pos_list)
                        else:
                            self.edges[sequence][edge_id].addPosition(pos_edge, framenum)

                # ASSUMPTION:
                # Adding spatial edges between all pairs of pedestrians.
                # TODO:
                # Can be pruned by considering pedestrians who are close to each other
                # Add spatial edges
                for ped_in in range(frame.shape[0]):
                    for ped_out in range(ped_in+1, frame.shape[0]):
                        pedID_in = frame[ped_in, 0]
                        pedID_out = frame[ped_out, 0]

                        t = frame[ped_in, 1]
                        x = frame[ped_in, 2]
                        y = frame[ped_in, 3]
                        z = frame[ped_in, 4]
                        v1 = frame[ped_in, 5]
                        v2 = frame[ped_in, 6]
                        v3 = frame[ped_in, 7]
                        width = frame[ped_in, 8]
                        length = frame[ped_in, 9]
                        heading = frame[ped_in, 10]
                        #pos_in = (t, x, y, z, v1, v2, v3, width, length, heading)
                        pos_in = (x,y,t)
                        t = frame[ped_out, 1]
                        x = frame[ped_out, 2]
                        y = frame[ped_out, 3]
                        z = frame[ped_out, 4]
                        v1 = frame[ped_out, 5]
                        v2 = frame[ped_out, 6]
                        v3 = frame[ped_out, 7]
                        width = frame[ped_out, 8]
                        length = frame[ped_out, 9]
                        heading = frame[ped_out, 10]
                        #pos_out = (t, x, y, z, v1, v2, v3, width, length, heading)
                        pos_out = (x,y,t)
                        pos = (pos_in, pos_out)
                        edge_id = (pedID_in, pedID_out)
                        # ASSUMPTION:
                        # Assuming that pedIDs always are in increasing order in the input batch data
                        if edge_id not in self.edges[sequence]:
                            edge_type = 'H-H/S'
                            edge_pos_list = {}
                            edge_pos_list[framenum] = pos
                            self.edges[sequence][edge_id] = ST_EDGE(edge_type, edge_id, edge_pos_list)
                        else:
                            self.edges[sequence][edge_id].addPosition(pos, framenum)

    def printGraph(self):
        '''
        Print function for the graph
        For debugging purposes
        '''
        for sequence in range(self.batch_size):
            nodes = self.nodes[sequence]
            edges = self.edges[sequence]

            print('Printing Nodes')
            print('===============================')
            for node in nodes.values():
                node.printNode()
                print('--------------')

            print
            print('Printing Edges')
            print('===============================')
            for edge in edges.values():
                edge.printEdge()
                print('--------------')

    def getSequence(self):
        '''
        Gets the sequence
        '''
        nodes = self.nodes[0]
        edges = self.edges[0]

        numNodes = len(nodes.keys())
        list_of_nodes = {}

        retNodes = np.zeros((self.seq_length, numNodes, 3))
        retEdges = np.zeros((self.seq_length, numNodes*numNodes, 3))  # Diagonal contains temporal edges
        retNodePresent = [[] for c in range(self.seq_length)]
        retEdgePresent = [[] for c in range(self.seq_length)]

        #pdb.set_trace()

        for i, ped in enumerate(nodes.keys()):
            list_of_nodes[ped] = i
            pos_list = nodes[ped].node_pos_list
            for framenum in range(self.seq_length):
                if framenum in pos_list:
                    retNodePresent[framenum].append(i)
                    retNodes[framenum, i, :] = list(pos_list[framenum])

        for ped, ped_other in edges.keys():
            i, j = list_of_nodes[ped], list_of_nodes[ped_other]
            edge = edges[(ped, ped_other)]

            if ped == ped_other:
                # Temporal edge
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        retEdgePresent[framenum].append((i, j))
                        retEdges[framenum, i*(numNodes) + j, :] = getVector(edge.edge_pos_list[framenum])
            else:
                # Spatial edge
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        retEdgePresent[framenum].append((i, j))
                        retEdgePresent[framenum].append((j, i))
                        # the position returned is a tuple of tuples

                        retEdges[framenum, i*numNodes + j, :] = getVector(edge.edge_pos_list[framenum])
                        retEdges[framenum, j*numNodes + i, :] = -np.copy(retEdges[framenum, i*(numNodes) + j, :])

        #pdb.set_trace()
        return retNodes, retEdges, retNodePresent, retEdgePresent


class ST_NODE():

    def __init__(self, node_type, node_id, node_pos_list):
        '''
        Initializer function for the ST node class
        params:
        node_type : Type of the node (Human or Obstacle)
        node_id : Pedestrian ID or the obstacle ID
        node_pos_list : Positions of the entity associated with the node in the sequence
        '''
        self.node_type = node_type
        self.node_id = node_id
        self.node_pos_list = node_pos_list

    def getPosition(self, index):
        '''
        Get the position of the node at time-step index in the sequence
        params:
        index : time-step
        '''
        assert(index in self.node_pos_list)
        return self.node_pos_list[index]

    def getLastPosition(self, index, pos):
        #print('index:{}:{}'.format(index,len(self.node_pos_list)))
        if len(self.node_pos_list)>1:
            count = 0
            sort_key = sorted(self.node_pos_list.keys())
            for key in sort_key:
                if count==len(sort_key)-2 :
                    current_frame = key
                    value = self.node_pos_list[key]
                    gap = index - current_frame
                    #print('count, gap: {}:{}'.format(count, gap))
                    if gap > 1:
                        #pdb.set_trace()
                        #print(self.node_pos_list.keys())
                        #print(sort_key)
                        new_x = ((gap-1)*pos[0]+value[0])/gap
                        new_y = ((gap-1)*pos[1]+value[1])/gap

                        #new_x = ((gap-1)*pos[1]+value[1])/gap
                        #new_y = ((gap-1)*pos[2]+value[2])/gap
                        #new_z = ((gap-1)*pos[3]+value[3])/gap
                        #new_v1 = ((gap-1)*pos[4]+value[4])/gap
                        #new_v2 = ((gap-1)*pos[5]+value[5])/gap
                        #new_v3 = ((gap-1)*pos[6]+value[6])/gap
                        #new_w = ((gap-1)*pos[7]+value[7])/gap
                        #new_l = ((gap-1)*pos[8]+value[8])/gap
                        #new_head = ((gap-1)*pos[9]+value[9])/gap

                        #new_pos = (pos[0],new_x,new_y,new_z,new_v1,new_v2,new_v3,new_w,new_l,new_head)
                        new_pos = (new_x, new_y, pos[2])
                        return new_pos
                count+=1

        assert(index-1 in self.node_pos_list)
        return self.node_pos_list[index-1]




    def getType(self):
        '''
        Get node type
        '''
        return self.node_type

    def getID(self):
        '''
        Get node ID
        '''
        return self.node_id

    def addPosition(self, pos, index):
        '''
        Add position to the pos_list at a specific time-step
        params:
        pos : A tuple (x, y)
        index : time-step
        '''
        assert(index not in self.node_pos_list)
        self.node_pos_list[index] = pos

    def printNode(self):
        '''
        Print function for the node
        For debugging purposes
        '''
        print('Node type:', self.node_type, 'with ID:', self.node_id, 'with positions:', self.node_pos_list.values(), 'at time-steps:', self.node_pos_list.keys())


class ST_EDGE():

    def __init__(self, edge_type, edge_id, edge_pos_list):
        '''
        Inititalizer function for the ST edge class
        params:
        edge_type : Type of the edge (Human-Human or Human-Obstacle)
        edge_id : Tuple (or set) of node IDs involved with the edge
        edge_pos_list : Positions of the nodes involved with the edge
        '''
        self.edge_type = edge_type
        self.edge_id = edge_id
        self.edge_pos_list = edge_pos_list

    def getPositions(self, index):
        '''
        Get Positions of the nodes at time-step index in the sequence
        params:
        index : time-step
        '''
        assert(index in self.edge_pos_list)
        return self.edge_pos_list[index]

    def getType(self):
        '''
        Get edge type
        '''
        return self.edge_type

    def getID(self):
        '''
        Get edge ID
        '''
        return self.edge_id

    def addPosition(self, pos, index):
        '''
        Add a position to the pos_list at a specific time-step
        params:
        pos : A tuple (x, y)
        index : time-step
        '''
        assert(index not in self.edge_pos_list)
        self.edge_pos_list[index] = pos

    def printEdge(self):
        '''
        Print function for the edge
        For debugging purposes
        '''
        print('Edge type:', self.edge_type, 'between nodes:', self.edge_id, 'at time-steps:', self.edge_pos_list.keys())
