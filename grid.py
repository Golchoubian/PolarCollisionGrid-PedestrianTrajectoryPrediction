import numpy as np
import torch
import itertools
from torch.autograd import Variable

def getGridMask(frame, num_person, neighborhood_size, grid_size, is_occupancy = False):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 2 matrix with each row being [x, y] 
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    num_person : number of people that exist in the given frame
    is_occupancy: A flag using for calculation of accupancy map

    '''
    mnp = num_person 

    if is_occupancy:
        frame_mask = np.zeros((mnp, grid_size**2))
    else:
        frame_mask = np.zeros((mnp, mnp, grid_size**2))
    frame_np =  frame.data.numpy()

  
    width_bound, height_bound = (neighborhood_size), (neighborhood_size) 

    #instead of 2 inner loop, we check all possible 2-permutations which is 2 times faster.
    list_indices = list(range(0, mnp)) 
    
    for real_frame_index, other_real_frame_index in itertools.permutations(list_indices, 2): 

        current_x, current_y = frame_np[real_frame_index, 0], frame_np[real_frame_index, 1] 

        width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
        height_low, height_high = current_y - height_bound/2, current_y + height_bound/2

        other_x, other_y = frame_np[other_real_frame_index, 0], frame_np[other_real_frame_index, 1] 
        
        if (other_x >= width_high) or (other_x < width_low) or (other_y >= height_high) or (other_y < height_low):
                # Ped not in surrounding, so binary mask should be zero
                continue
        # If in surrounding, calculate the grid cell
        cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size)) 
        cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size)) # The orign is at the left bottom

        if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0: 
                # When the neighbour excatly resides in the out boudary (edge case for the equal sign)
                continue

        if is_occupancy:
            frame_mask[real_frame_index, cell_x + cell_y*grid_size] = 1 
        else:
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y*grid_size] = 1 

    return frame_mask


def getSequenceGridMask(sequence, pedlist_seq, neighborhood_size, grid_size, using_cuda, lookup_seq, is_occupancy=False):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A matrix of shape SL x MNP x 3  
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    is_occupancy: A flag using for calculation of accupancy map
    '''
    sl = len(sequence)
    sequence_mask = []

    for i in range(sl):
   
        pedlist_seq[i] = [int(_x_seq) for _x_seq in pedlist_seq[i]]
        current_ped_list = pedlist_seq[i].copy()
        converted_pedlist = [lookup_seq[_x_seq] for _x_seq in current_ped_list]
        list_of_x_seq = Variable(torch.LongTensor(converted_pedlist))

        # Get the portion of the sequence[i] that has only pedestrians present in that specific frame
        current_x_seq = torch.index_select(sequence[i], 0, list_of_x_seq)

        mask = Variable(torch.from_numpy(getGridMask(current_x_seq, len(pedlist_seq[i]), neighborhood_size, grid_size, is_occupancy)).float())
        if using_cuda:
            mask = mask.cuda()
        sequence_mask.append(mask)

    return sequence_mask


def getGridMask_heterogeneous(frame, num_person, frame_veh, num_veh, neighborhood_size, grid_size, is_occupancy = False):

    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 2 matrix with each row being [x, y] 
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    num_person : number of people that exist in the given frame
    is_occupancy: A flag using for calculation of accupancy map

    '''
    mnp = num_person 
    mnv = num_veh
    
    if is_occupancy:
        frame_mask = np.zeros((mnp, grid_size**2))
    else:
        frame_mask = np.zeros((mnp, mnv, grid_size**2))
    frame_np =  frame.data.numpy()
    frame_nv =  frame_veh.data.numpy()
   
    width_bound, height_bound = (neighborhood_size)*2, (neighborhood_size)*2

    #instead of 2 inner loop, we check all possible 2-permutations which is 2 times faster.
    list_indices = list(range(0, mnp)) 
    list_indices_veh = list(range(0, mnv)) 
  
    
    for real_frame_index, other_real_frame_index in itertools.product(list_indices,list_indices_veh): 

        current_x, current_y = frame_np[real_frame_index, 0], frame_np[real_frame_index, 1] 

        width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
        height_low, height_high = current_y - height_bound/2, current_y + height_bound/2

        other_x, other_y = frame_nv[other_real_frame_index, 0], frame_nv[other_real_frame_index, 1] 
        
        if (other_x >= width_high) or (other_x < width_low) or (other_y >= height_high) or (other_y < height_low):
                # Veh not in ped's surrounding, so binary mask should be zero
                continue
        # If in surrounding, calculate the grid cell
        cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size)) 
        cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size)) # The orign is at the left bottom

        if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0: 
                # When the neighbour excatly resides in the out boudary (edge case for the equal sign)
                continue

        if is_occupancy:
            frame_mask[real_frame_index, cell_x + cell_y*grid_size] = 1 
        else:
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y*grid_size] = 1 

    return frame_mask


    
def getSequenceGridMask_heterogeneous(sequence, pedlist_seq, sequence_veh, vehlist_seq, neighborhood_size, grid_size, using_cuda, lookup_seq, lookup_seq_veh, is_occupancy=False):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A matrix of shape SL x MNP x 3  
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    is_occupancy: A flag using for calculation of accupancy map
    '''
    sl = len(sequence)
    sequence_mask = []

    for i in range(sl):
        
        pedlist_seq[i] = [int(_x_seq) for _x_seq in pedlist_seq[i]]
        current_ped_list = pedlist_seq[i].copy()
        converted_pedlist = [lookup_seq[_x_seq] for _x_seq in current_ped_list]
        list_of_x_seq = Variable(torch.LongTensor(converted_pedlist))
        # Get the portion of the sequence[i] that has only pedestrians present in that specific frame
        current_x_seq = torch.index_select(sequence[i], 0, list_of_x_seq)

        
        vehlist_seq[i] = [int(_x_seq) for _x_seq in vehlist_seq[i]]
        current_veh_list = vehlist_seq[i].copy()
        converted_vehlist = [lookup_seq_veh[_x_seq] for _x_seq in current_veh_list]
        list_of_x_seq_veh = Variable(torch.LongTensor(converted_vehlist))
        current_x_seq_veh = torch.index_select(sequence_veh[i], 0, list_of_x_seq_veh)

        mask = Variable(torch.from_numpy(getGridMask_heterogeneous(current_x_seq, len(pedlist_seq[i]), current_x_seq_veh, len(vehlist_seq[i]),  neighborhood_size, grid_size, is_occupancy)).float()) # Mahsa
        if using_cuda:
            mask = mask.cuda()
        sequence_mask.append(mask)

    return sequence_mask