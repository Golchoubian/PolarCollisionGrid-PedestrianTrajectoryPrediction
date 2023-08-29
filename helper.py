
import torch
import numpy as np
from model_CollisionGrid import CollisionGridModel
from model_SocialLSTM import SocialModel
from model_VanillaLSTM import VLSTMModel
from torch.autograd import Variable
import math
import itertools


# one time set dictionary for a exist key
class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if not key in self:
            super(WriteOnceDict, self).__setitem__(key, value)

def position_change_seq(x_seq, PedsList_seq, lookup_seq):
    #substract each frame value from its previosu frame to create displacment data.
    first_values_dict = WriteOnceDict()
    vectorized_x_seq = x_seq.clone()
    first_presence_flag = [0]*(x_seq.shape[1])
    latest_pos = [0]*(x_seq.shape[1])

    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            first_values_dict[ped] = frame[lookup_seq[ped], 0:2]
            if first_presence_flag[lookup_seq[ped]] == 0: # this frame is the first frame where this pedestrian apears
                vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - first_values_dict[ped][0:2] # this should always give (0,0)
                latest_pos[lookup_seq[ped]] = frame[lookup_seq[ped], 0:2]
                first_presence_flag[lookup_seq[ped]] = 1
            else:
                vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - latest_pos[lookup_seq[ped]]
                latest_pos[lookup_seq[ped]] = frame[lookup_seq[ped], 0:2]
    
    return vectorized_x_seq, first_values_dict

def vectorize_seq(x_seq, PedsList_seq, lookup_seq):
    #substract first frame value to all frames for a ped.Therefore, convert absolute pos. to relative pos.
    first_values_dict = WriteOnceDict()
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            first_values_dict[ped] = frame[lookup_seq[ped], 0:2]
            vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - first_values_dict[ped][0:2]
    
    return vectorized_x_seq, first_values_dict


def getCoef(outputs):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    '''
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr

def Gaussian2DLikelihood(outputs, targets, nodesPresent, look_up):
    '''
    params:
    outputs : predicted locations
    targets : true locations
    nodesPresent : True nodes present in each frame in the sequence
    look_up : lookup table for determining which ped is in which array index

    '''
    seq_length = outputs.size()[0]
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

    loss = 0
    counter = 0

    for framenum in range(seq_length):

        nodeIDs = nodesPresent[framenum]
        nodeIDs = [int(nodeID) for nodeID in nodeIDs]

        for nodeID in nodeIDs:
            nodeID = look_up[nodeID]
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss


def get_model(index, arguments, infer = False): 
    # return a model given index and arguments
    if index == 1:
        return SocialModel(arguments, infer)
    elif index == 4:
        return CollisionGridModel(arguments, infer)
    elif index == 3:
        return VLSTMModel(arguments, infer)
    else:
        raise ValueError('Invalid model index!')
   

def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent, look_up):
    '''
    Parameters
    ==========

    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation

    nodesPresent : a list of nodeIDs present in the frame
    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]

    numNodes = mux.size()[1]
    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    converted_node_present = [look_up[node] for node in nodesPresent]
    for node in range(numNodes):
        if node not in converted_node_present:
            continue
        mean = [o_mux[node], o_muy[node]]
        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], 
                [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        mean = np.array(mean, dtype='float')
        cov = np.array(cov, dtype='float')
        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y 

def revert_seq(x_seq, PedsList_seq, lookup_seq, first_values_dict):
    # convert velocity array to absolute position array
    absolute_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + first_values_dict[ped][0:2]

    return absolute_x_seq



def revert_postion_change_seq2(x_seq, PedsList_seq, lookup_seq, first_values_dict, orig_x_seq, obs_length, infer=False):
    # convert displacement array to absolute position array
    absolute_x_seq = x_seq.clone()
    first_presence_flag = [0]*(x_seq.shape[1])
    latest_pos = [0]*(x_seq.shape[1])
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            if first_presence_flag[lookup_seq[ped]] == 0: # this frame is the first frame where this pedestrian apears
                absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + first_values_dict[ped][0:2]
                latest_pos[lookup_seq[ped]] = absolute_x_seq[ind, lookup_seq[ped], 0:2] 
                # for the first frame this absolute_x_seq is same as the orig_x_seq since frame is [0,0]
                first_presence_flag[lookup_seq[ped]] = 1
            else:
                absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + latest_pos[lookup_seq[ped]]
                if (infer==True and ind>obs_length): # we have to rely on the algorithm's own prediction for the next state
                    latest_pos[lookup_seq[ped]] = absolute_x_seq[ind, lookup_seq[ped], 0:2]
                else: # we use the ground truth that we have
                    latest_pos[lookup_seq[ped]] = orig_x_seq[ind, lookup_seq[ped], 0:2]

    return absolute_x_seq


def get_num_collisions_hetero(ret_nodes, NodesPresent, ObsNodesPresent, using_cuda, look_up, Veh_nodes, VehList, lookup_veh, threshold):


    collision_count = 0
    all_two_cases = 0
    num_nodes = ret_nodes.shape[1]
    num_vehs = Veh_nodes.shape[1]
    two_agent_combination =  itertools.product(list(range(num_nodes)),list(range(num_vehs)))

    lookup_indxToid = reverse_dict(look_up)
    lookup_veh_indxToid = reverse_dict(lookup_veh)

    for ped_ind, veh_ind in two_agent_combination:
        # check if the two agent's have any collision along their predicted trajectory

        idped = lookup_indxToid[ped_ind]
        idveh = lookup_veh_indxToid[veh_ind]

        if (idped not in ObsNodesPresent): 
            # we do not count on the prediction of pedestrians that we did not have any information from them during the observation period
            continue
        
        all_two_cases += 1
        for t in range(len(ret_nodes)):
            if ((idped in NodesPresent[t]) and (idveh in VehList[t])): # if both agent are present in this timestep
                pre_pos_ped = ret_nodes[t][ped_ind][:2]
                pos_veh = Veh_nodes[t][veh_ind][:2]
                dis = pre_pos_ped - pos_veh
                if torch.norm(dis, p=2) < threshold:
                    collision_count += 1
                    break # one time of collision is enough

    return collision_count, all_two_cases



def get_num_collisions_homo(ret_nodes, NodesPresent, ObsNodesPresent, using_cuda, look_up, threshold):


    collision_count = 0
    all_two_cases = 0
    num_nodes = ret_nodes.shape[1]
    two_agent_combination = list(itertools.combinations(list(range(num_nodes)), 2))

    lookup_indxToid = reverse_dict(look_up)

    for two_agent in two_agent_combination:

        # check if the two agent's have any collision along their predicted trajectory
        agentA = two_agent[0]
        agentB = two_agent[1]
        idA = lookup_indxToid[agentA]
        idB = lookup_indxToid[agentB]

        if (idA not in ObsNodesPresent) or (idB not in ObsNodesPresent): 
            # we do not count on the prediction of pedestrians that we did not have any information from them during the observation period
            continue
        
        all_two_cases += 1
        for t in range(len(ret_nodes)):
            if ((idA in NodesPresent[t]) and (idB in NodesPresent[t])): # if both agent are present in this timestep
                pre_posA = ret_nodes[t][agentA][:2]
                pre_posB = ret_nodes[t][agentB][:2]
                dis = pre_posB - pre_posA
                if torch.norm(dis, p=2) < threshold:
                    collision_count += 1
                    break # one time of collision is enough

    return collision_count, all_two_cases


def get_mean_error(ret_nodes, nodes, assumedNodesPresent, ObsNodesPresent, using_cuda, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length)
    if using_cuda:
        error = error.cuda()

    for tstep in range(pred_length):
        counter = 0

        for nodeID in assumedNodesPresent[tstep]:
            nodeID = int(nodeID)

           
            if nodeID not in ObsNodesPresent: 
                # This is for elimiating those agents that did not have any observation data 
                # and only appread during the predictino length. We want to exclude then from the error calculation process
                continue

            nodeID = look_up[nodeID]


            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]

            error[tstep] += torch.norm(pred_pos - true_pos, p=2)
            counter += 1

        if counter != 0:
            error[tstep] = error[tstep] / counter

    return torch.mean(error)



def get_final_error(ret_nodes, nodes, assumedNodesPresent, ObsNodesPresent, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index


    Returns
    =======

    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = 0
    counter = 0

    # Last time-step
    tstep = pred_length - 1
    for nodeID in assumedNodesPresent[tstep]:
        nodeID = int(nodeID)


        if nodeID not in ObsNodesPresent: # When this will happen?!
            continue

        nodeID = look_up[nodeID]

        
        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]
        
        error += torch.norm(pred_pos - true_pos, p=2)
        counter += 1
        
    if counter != 0:
        error = error / counter
            
    return error


def sequence_error(outputs, x_orig, Pedlist, look_up, using_cuda, first_values_dict,obs_length):

    seq_len = outputs.shape[0]
    num_ped = outputs.shape[1]
    pred_seq = Variable(torch.zeros(seq_len, num_ped, 2))
    if using_cuda:
        pred_seq = pred_seq.cuda()

    for tstep in range(seq_len):
        mux, muy, sx, sy, corr = getCoef(outputs[tstep,:,:].view(1, num_ped, 5))
        next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, Pedlist[tstep], look_up)
        pred_seq[tstep,:,0] = next_x
        pred_seq[tstep,:,1] = next_y

    Pedlist_whole_seq = sum(Pedlist, [])
    
    # When working with displacement the following two lines should be added:
    pred_seq_abs = revert_postion_change_seq2(pred_seq.data.cpu(), Pedlist, look_up, first_values_dict, x_orig, obs_length)
    total_error = get_mean_error(pred_seq_abs.data, x_orig.data, Pedlist, Pedlist_whole_seq, using_cuda, look_up)

    return total_error, pred_seq_abs

def get_hausdorff_distance(ret_nodes, nodes, assumedNodesPresent, ObsNodesPresent, using_cuda, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : The largest distance from each predicted point to any point on the ground truth trajectory (Modified Hausdorff Distance: MHD)
    '''
    pred_length = ret_nodes.size()[0]
    PedsPresent = list(set(sum(assumedNodesPresent, []))) # getting a list of all non repeating agetn ids present in the prediction length
    Valid_PedsPresent = [i for i in PedsPresent if i in ObsNodesPresent] # elimiating those agents that did not have any observation data and
    # only appread during the prediction length
    num_agents = len(Valid_PedsPresent)

    if (num_agents == 0):
        return None
    else:
        error = torch.zeros(num_agents)
    
        if using_cuda:
            error = error.cuda()

        count = 0
        for nodeID in Valid_PedsPresent:
        
            nodeID = int(nodeID)
            present_frames = [i for i, id_list in enumerate(assumedNodesPresent) if nodeID in id_list]
            
            nodeID = look_up[nodeID]
            pred_traj = ret_nodes[present_frames, nodeID, :]
            true_traj = nodes[present_frames, nodeID, :]

            error_t = torch.zeros(len(present_frames))
            for tstep in range(len(present_frames)):
                pred_pos = pred_traj[tstep,:]
                error_t[tstep] = torch.max(torch.norm(true_traj - pred_pos, p=2, dim=1)) 
                # the maximum distance between this predicted timestep and the ground truth trajectory
            
            # maximum among all the predicted time steps distance to groud truth for this agent
            error[count] = max(error_t)
            count += 1

        MHD = sum(error)/count

        return MHD



def get_velocity_errors(ret_nodes, nodes, assumedNodesPresent, ObsNodesPresent, using_cuda, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : Mean distance between predicted speed and true speed & Mean distance between predicted heaing and true heading
    '''
    pred_length = ret_nodes.size()[0]
    error_speed = torch.zeros(pred_length)
    error_heading = torch.zeros(pred_length)

    if using_cuda:
        error_speed = error_speed.cuda()
        error_heading = error_heading.cuda()

    for tstep in range(pred_length): 
        counter = 0

        for nodeID in assumedNodesPresent[tstep]:
            nodeID = int(nodeID)

          
            if nodeID not in ObsNodesPresent:
                continue

            nodeID = look_up[nodeID]


            pred_v = ret_nodes[tstep, nodeID, :]
            true_v = nodes[tstep, nodeID, :]

            pred_speed = torch.norm(pred_v, p=2)
            true_speed = torch.norm(true_v, p=2)  

            error_speed[tstep] += torch.pow((pred_speed - true_speed), 2)

            dot = true_v[0]*pred_v[0] + true_v[1]*pred_v[1] # dot is propotional to cos(theta)
            det = true_v[0]*pred_v[1] - pred_v[0]*true_v[1] # det (|V_e V_n|) is propotional to sin(theta)
            angle = math.atan2(det,dot) * (180/math.pi) # the value is between -180 and 180
            error_heading[tstep] += angle**2

            counter += 1

        if counter != 0:
            error_speed[tstep] = error_speed[tstep] / counter
            error_heading[tstep] = error_heading[tstep] / counter

    return torch.mean(error_speed), sum(error_heading)/len(error_heading)


def available_frame_extraction(agent_indx, pedlist_seq, lookup_seq):
    
    key_list = list(lookup_seq.keys())
    val_list = list(lookup_seq.values())
    indx_position = val_list.index(agent_indx)
    agent_id = key_list[indx_position]

    present_frames = []

    for f in range(len(pedlist_seq)):
        if agent_id in pedlist_seq[f]:
            present_frames.append(f)

    return present_frames



def reverse_dict(lookup):

    reversedDict = dict()
    key_list = list(lookup.keys()) # the agent id
    val_list = list(lookup.values()) # the index number of that agent in the tensor
    n = len(key_list)
    for i in range(n):
        key = val_list[i]
        val = key_list[i]
        reversedDict[key] = val
    
    return reversedDict



