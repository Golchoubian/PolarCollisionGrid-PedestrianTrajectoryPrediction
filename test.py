import argparse
import pickle
import os
import time 

import torch
from torch.autograd import Variable

from utils.DataLoader import DataLoader
from utils.grid import getSequenceGridMask, getGridMask, getSequenceGridMask_heterogeneous, getGridMask_heterogeneous
from utils.Interaction import getInteractionGridMask, getSequenceInteractionGridMask
from utils.helper import * # want to use its get_model()
# from utils.helper import sample_gaussian_2d
from matplotlib import pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=6, # 6 for HBS
                            help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=6, # 6 for HBS
                            help='Predicted length of the trajectory')

    # cuda support
    parser.add_argument('--use_cuda', action="store_true", default=True,
                            help='Use GPU or not')      
    # number of iteration                  
    parser.add_argument('--iteration', type=int, default=200,
                            help='Number of saved models (during training) for testing their performance here \
                                  (smallest test errror will be selected)')

    # ============================================================================================
    #       change the following three arguments according to the model you want to test
    # ============================================================================================

    # method selection. this have to match with the training method manually
    parser.add_argument('--method', type=int, default=4,
                            help='Method of lstm being used (1 = social lstm, 3 = vanilla lstm, 4 = collision grid)')

    # Model to be loaded (saved model (the epoch #) during training
    # with the best performace according to previous invesigation on valiquation set)
    parser.add_argument('--epoch', type=int, default=181, # PV-CollisionGrid: 181, SocialLSTM: 185, VanillaLSTM: 192,
                            help='Epoch of model to be loaded') # oblation models:  V-CollisionGrid: 188, P-CollisionGrid: 172
    
    # The number of samples to be generated for each test data, when reporting its performance
    parser.add_argument('--sample_size', type=int, default=20,
                            help='The number of sample trajectory that will be generated')
        

    sample_args = parser.parse_args()

    seq_length = sample_args.obs_length + sample_args.pred_length

    dataloader = DataLoader(1, seq_length, infer=True, filtering=True)
    dataloader.reset_batch_pointer()

    # Define the path for the config file for saved args
    prefix = 'Store_Results/'
    save_directory_pre = os.path.join(prefix, 'model/')
    if sample_args.method == 1:
        save_directory = os.path.join(save_directory_pre, 'SocialLSTM/') 
    elif sample_args.method == 3:
        save_directory = os.path.join(save_directory_pre, 'VanillaLSTM/')
    elif sample_args.method == 4:
        save_directory = os.path.join(save_directory_pre, 'CollisionGrid/')
    else:
        raise ValueError('The selected method is not defined!')

    with open(os.path.join(save_directory,'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    model_name = "LSTM"
    method_name = "SOCIALLSTM" # Attention: This name has not been changed for different models used. (ToDO later)
    save_tar_name = method_name+"_lstm_model_" 
    if saved_args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    plot_directory = os.path.join(prefix, 'plot/')
    plot_test_file_directory = 'test'

    iteration_result = []
    iteration_total_error = []
    iteration_final_error = []
    iteration_MHD_error = []
    iteration_speed_error = []
    iteration_heading_error = []
    iteration_collision_percent = []
    iteration_collision_percent_pedped = []
    iteration_collision_percent_pedveh = []
    
    smallest_err = 100000
    smallest_err_iter_num = -1

    # Use "range(0, sample_args.iteration):" when willing to find the best model during training
    # in that case uncomment line 109 and comment line 110
    # This iteration is for testing the results for different stages of the trained model
    # (the stored paramters of the model at different iterations)
    for iteration in [0]: # range(0, sample_args.iteration): 
        # Initialize net
        net = get_model(sample_args.method, saved_args, True)

        if sample_args.use_cuda:        
            net = net.cuda()  

        # Get the checkpoint path for loading the trained model
        # checkpoint_path = os.path.join(save_directory, save_tar_name+str(iteration)+'.tar')
        checkpoint_path = os.path.join(save_directory, save_tar_name+str(sample_args.epoch)+'.tar')
        if os.path.isfile(checkpoint_path):
            print('Loading checkpoint')
            checkpoint = torch.load(checkpoint_path)
            model_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print('Loaded checkpoint at epoch', model_epoch)
        else:
            raise ValueError('The seleted model checkpoint does not exists in the specified directory!')

        # Initialzing some of the parameters 
        
        results = []
        
        # Variable to maintain total error
        total_error = 0
        final_error = 0
        MHD_error = 0
        speed_error = 0
        heading_error = 0

        num_collision_homo = 0
        all_num_cases_homo = 0
        num_collision_hetero = 0
        all_num_cases_hetero = 0


        x_WholeBatch, numPedsList_WholeBatch, PedsList_WholeBatch, x_veh_WholeBatch, numVehsList_WholeBatch,\
            VehsList_WholeBatch, grids_WholeBatch, grids_veh_WholeBatch, grids_TTC_WholeBatch, grids_TTC_veh_WholeBatch \
                 = dataloader.batch_creater(False, sample_args.method, suffle=False)

        None_count = 0
        for batch in range(dataloader.num_batches):
            start = time.time()
            # Get data
            # x, y, d , numPedsList, PedsList, x_veh, numVehsList, VehsList = dataloader.next_batch()
            x, numPedsList, PedsList = x_WholeBatch[batch], numPedsList_WholeBatch[batch], PedsList_WholeBatch[batch]
            x_veh, numVehsList, VehsList = x_veh_WholeBatch[batch], numVehsList_WholeBatch[batch], VehsList_WholeBatch[batch]

            x_seq, numPedsList_seq, PedsList_seq = x[0], numPedsList[0], PedsList[0] 
            x_seq_veh , numVehsList_seq, VehsList_seq = x_veh[0], numVehsList[0], VehsList[0]

            
            # dense vector creation
            x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
            x_seq_veh, lookup_seq_veh = dataloader.convert_proper_array(x_seq_veh, numVehsList_seq, VehsList_seq, veh_flag=True)

            # will be used for error calculation
            orig_x_seq = x_seq.clone() 
            orig_x_seq_veh = x_seq_veh.clone()

            # grid mask calculation
            if  saved_args.method == 1: # social lstm   
                grid_seq = getSequenceGridMask(x_seq, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size,
                                                saved_args.use_cuda, lookup_seq) 
            
            elif saved_args.method == 4: # collision grid
                grid_seq, grid_TTC_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq, PedsList_seq,
                                                                        saved_args.TTC, saved_args.D_min, saved_args.num_sector,
                                                                        False, lookup_seq, lookup_seq)
                grid_seq_veh_in_ped, grid_TTC_veh_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq_veh,
                                                                                        VehsList_seq, saved_args.TTC_veh,
                                                                                        saved_args.D_min_veh, saved_args.num_sector,
                                                                                        False, lookup_seq, lookup_seq_veh,
                                                                                        is_heterogeneous=True, is_occupancy=False)
            
            elif saved_args.method == 3:
                grid_seq = None

            
            # vectorize datapoints
            x_seq, first_values_dict = position_change_seq(x_seq, PedsList_seq, lookup_seq)
         
            if sample_args.use_cuda:
                x_seq = x_seq.cuda()
                if saved_args.method == 4:
                    x_seq_veh = x_seq_veh.cuda()


            sample_error = []
            sample_final_error = []
            sample_MHD_error = []
            sample_speed_error = []
            sample_heading_error = []
            sample_ret_x_seq = []
            sample_dist_param_seq = []
            sample_num_coll_hetero = []
            sample_num_coll_homo = []
            sample_all_num_cases_homo = []
            sample_all_num_cases_hetero = []

            for sample_num in range(sample_args.sample_size):


                # The sample function
                if saved_args.method == 3: # vanilla lstm
                    # Extract the observed part of the trajectories
                    obs_traj, obs_PedsList_seq = x_seq[:sample_args.obs_length], PedsList_seq[:sample_args.obs_length]
                    ret_x_seq, dist_param_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq, PedsList_seq,
                                                        saved_args, dataloader, lookup_seq, numPedsList_seq, saved_args.gru,
                                                        first_values_dict, orig_x_seq)

                elif saved_args.method == 4: # collision grid
                    # Extract the observed part of the trajectories
                    obs_traj, obs_PedsList_seq, obs_grid, obs_grid_TTC = x_seq[:sample_args.obs_length], \
                                                                        PedsList_seq[:sample_args.obs_length], \
                                                                        grid_seq[:sample_args.obs_length], \
                                                                        grid_TTC_seq[:sample_args.obs_length]
                    obs_grid_veh_in_ped, obs_grid_TTC_veh = grid_seq_veh_in_ped[:sample_args.obs_length], \
                                                            grid_TTC_veh_seq[:sample_args.obs_length]
                  

                    ret_x_seq, dist_param_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq,
                                                        PedsList_seq, saved_args, dataloader, lookup_seq,
                                                        numPedsList_seq, saved_args.gru, first_values_dict,
                                                        orig_x_seq, obs_grid, x_seq_veh, VehsList_seq, 
                                                        lookup_seq_veh, obs_grid_veh_in_ped,
                                                        obs_grid_TTC, obs_grid_TTC_veh)

                elif saved_args.method == 1: # soial lstm
                    # Extract the observed part of the trajectories
                    obs_traj, obs_PedsList_seq, obs_grid = x_seq[:sample_args.obs_length], \
                                                           PedsList_seq[:sample_args.obs_length], \
                                                           grid_seq[:sample_args.obs_length]
                 
                    ret_x_seq, dist_param_seq = sample(obs_traj, obs_PedsList_seq, sample_args, net, x_seq,
                                                        PedsList_seq, saved_args, dataloader, lookup_seq,
                                                        numPedsList_seq, saved_args.gru, first_values_dict,
                                                        orig_x_seq, obs_grid, x_seq_veh, VehsList_seq, 
                                                        lookup_seq_veh, None, None, None)


                ret_x_seq = revert_postion_change_seq2(ret_x_seq.cpu(), PedsList_seq, lookup_seq, first_values_dict,
                                                        orig_x_seq, sample_args.obs_length, infer=True)
                dist_param_seq[:,:,0:2] = revert_postion_change_seq2(dist_param_seq[:,:,0:2].cpu(), PedsList_seq, 
                                                                     lookup_seq, first_values_dict, orig_x_seq, 
                                                                     sample_args.obs_length, infer=True)

                # Getting the agent_ids of those present in the observation section. 
                # error should be calculated only for those agents that their data exists in the observation part
                PedsList_obs = sum(PedsList_seq[:sample_args.obs_length], []) # contains duplicat but does not make any problem


                sample_error_batch = get_mean_error(ret_x_seq[sample_args.obs_length:,:,:2].data,
                                                    orig_x_seq[sample_args.obs_length:,:,:2].data,
                                                    PedsList_seq[sample_args.obs_length:], PedsList_obs, False, lookup_seq)
                sample_final_error_batch = get_final_error(ret_x_seq[sample_args.obs_length:,:,:2].data, 
                                                           orig_x_seq[sample_args.obs_length:,:,:2].data, 
                                                           PedsList_seq[sample_args.obs_length:],PedsList_obs, lookup_seq)
                sample_MHD_error_batch = get_hausdorff_distance(ret_x_seq[sample_args.obs_length:,:,:2].data, 
                                                                orig_x_seq[sample_args.obs_length:,:,:2].data, 
                                                                PedsList_seq[sample_args.obs_length:], PedsList_obs, False, lookup_seq)
                sample_speed_error_batch, sample_heading_error_batch = get_velocity_errors(ret_x_seq[sample_args.obs_length:,:,2:4].data,
                                                                                            orig_x_seq[sample_args.obs_length:,:,2:4].data,
                                                                                            PedsList_seq[sample_args.obs_length:], PedsList_obs,
                                                                                            False, lookup_seq)
                sample_num_coll_homo_batch, sample_all_num_cases_homo_batch = get_num_collisions_homo(ret_x_seq[sample_args.obs_length:,:,:2].data,
                                                                                                       PedsList_seq[sample_args.obs_length:], 
                                                                                                       PedsList_obs, False, lookup_seq,
                                                                                                        saved_args.D_min)
                sample_num_coll_hetero_batch, sample_all_num_cases_hetero_batch = get_num_collisions_hetero(   
                                                                                                ret_x_seq[sample_args.obs_length:,:,:2].data,
                                                                                                PedsList_seq[sample_args.obs_length:], 
                                                                                                PedsList_obs, False, lookup_seq,
                                                                                                x_seq_veh[sample_args.obs_length:,:,:2].cpu().data,
                                                                                                VehsList_seq[sample_args.obs_length:], 
                                                                                                lookup_seq_veh, saved_args.D_min_veh)

                sample_error.append(sample_error_batch)
                sample_final_error.append(sample_final_error_batch)
                sample_MHD_error.append(sample_MHD_error_batch)
                sample_speed_error.append(sample_speed_error_batch)
                sample_heading_error.append(sample_heading_error_batch)
                sample_ret_x_seq.append((ret_x_seq.clone()).data.cpu().numpy())
                sample_dist_param_seq.append((dist_param_seq.clone()).data.cpu().numpy())
                sample_num_coll_homo.append(sample_num_coll_homo_batch)
                sample_all_num_cases_homo.append(sample_all_num_cases_homo_batch)
                sample_num_coll_hetero.append(sample_num_coll_hetero_batch)
                sample_all_num_cases_hetero.append(sample_all_num_cases_hetero_batch)
                

            # Deciding the best sample based on the average displacement error
            min_ADE = min(sample_error)
            min_index = sample_error.index(min_ADE)

            total_error += sample_error[min_index] # or min_ADE
            final_error += sample_final_error[min_index]
            MHD_error += sample_MHD_error[min_index]
            speed_error += sample_speed_error[min_index]
            heading_error += sample_heading_error[min_index]
            num_collision_homo += sample_num_coll_homo[min_index]
            all_num_cases_homo += sample_all_num_cases_homo[min_index]
            num_collision_hetero += sample_num_coll_hetero[min_index]
            all_num_cases_hetero += sample_all_num_cases_hetero[min_index]


            end = time.time()

            print('Current file : ', dataloader.get_file_name(0),' Processed trajectory number : ', batch+1,
                   'out of', dataloader.num_batches, 'trajectories in time', end - start)

            if sample_args.method == 3:
                results.append((orig_x_seq.data.cpu().numpy(), sample_ret_x_seq[min_index], PedsList_seq, lookup_seq , None, 
                                sample_args.obs_length, sample_dist_param_seq[min_index], orig_x_seq_veh.data.cpu().numpy(),
                                VehsList_seq, lookup_seq_veh, None, None))
            elif sample_args.method == 4:
                results.append((orig_x_seq.data.cpu().numpy(), sample_ret_x_seq[min_index], PedsList_seq, lookup_seq , None, 
                                sample_args.obs_length, sample_dist_param_seq[min_index], orig_x_seq_veh.data.cpu().numpy(), 
                                VehsList_seq, lookup_seq_veh, grid_seq, grid_seq_veh_in_ped))
            elif sample_args.method == 1:
                results.append((orig_x_seq.data.cpu().numpy(), sample_ret_x_seq[min_index], PedsList_seq, lookup_seq , None,
                                 sample_args.obs_length, sample_dist_param_seq[min_index], orig_x_seq_veh.data.cpu().numpy(), 
                                 VehsList_seq, lookup_seq_veh, grid_seq, None))

        iteration_result.append(results)
        iteration_total_error.append(total_error.data.cpu()/ dataloader.num_batches)
        iteration_final_error.append(final_error.data.cpu()/ dataloader.num_batches)
        iteration_MHD_error.append(MHD_error.data.cpu()/ (dataloader.num_batches-None_count))
        iteration_speed_error.append(speed_error.data.cpu()/ dataloader.num_batches)
        iteration_heading_error.append(heading_error.data.cpu()/ dataloader.num_batches)
        iteration_collision_percent.append((num_collision_homo+num_collision_hetero)/(all_num_cases_homo+all_num_cases_hetero))
        iteration_collision_percent_pedped.append(num_collision_homo/all_num_cases_homo)
        iteration_collision_percent_pedveh.append(num_collision_hetero/all_num_cases_hetero)
        
        print('Iteration:' ,iteration+1,' Total testing (prediction sequence) mean error of the model is ', total_error / dataloader.num_batches) 
        print('Iteration:' ,iteration+1,'Total testing final error of the model is ', final_error / dataloader.num_batches)
        print('Iteration:' ,iteration+1,'Total tresting (prediction sequence) hausdorff distance error of the model is ', 
                                                                                    MHD_error / (dataloader.num_batches-None_count))
        print('Iteration:' ,iteration+1,'Total tresting (prediction sequence) speed error of the model is ', speed_error / dataloader.num_batches)
        print('Iteration:' ,iteration+1,'Total tresting final heading error of the model is ', heading_error / dataloader.num_batches)
        print('Iteration:' ,iteration+1,'Overll percentage of collision of the model is ', iteration_collision_percent[-1])
        print('Iteration:' ,iteration+1,'Percentage of collision between pedestrians of the model is ', iteration_collision_percent_pedped[-1])
        print('Iteration:' ,iteration+1,'Percentage of collision between pedestrians and vehicles of the model is ',
                                                                                     iteration_collision_percent_pedveh[-1])
        # print('None count for MHD calculation:', None_count)

        
        if total_error<smallest_err:
            print("**********************************************************")
            print('Best iteration has been changed. Previous best iteration: ', smallest_err_iter_num+1, 'Error: ',
                   smallest_err / dataloader.num_batches)
            print('New best iteration : ', iteration+1, 'Error: ',total_error / dataloader.num_batches)
            smallest_err_iter_num = iteration
            smallest_err = total_error

    dataloader.write_to_plot_file(iteration_result[smallest_err_iter_num], os.path.join(plot_directory, plot_test_file_directory)) 

    print("==================================================")
    print("==================================================")
    print("==================================================")
    print('Best final iteration : ', smallest_err_iter_num+1)
    print('ADE: ', smallest_err.item() / dataloader.num_batches)
    print('FDE: ', iteration_final_error[smallest_err_iter_num].item())
    print('MHD: ', iteration_MHD_error[smallest_err_iter_num].item())
    print('Speed error: ',iteration_speed_error[smallest_err_iter_num].item()**0.5)
    print('Heading error: ', iteration_heading_error[smallest_err_iter_num].item()**0.5)
    print('Collision percentage: ', round(iteration_collision_percent[smallest_err_iter_num], 4) * 100)
    print('Collision percentage (ped-ped): ', round(iteration_collision_percent_pedped[smallest_err_iter_num], 4) * 100)
    print('Collision percentage (ped-veh): ', round(iteration_collision_percent_pedveh[smallest_err_iter_num], 4) * 100)
    


def sample(x_seq, Pedlist, args, net, true_x_seq, true_Pedlist, saved_args, dataloader, look_up, num_pedlist, is_gru,
            first_values_dict, orig_x_seq, grid=None, x_seq_veh=None, Vehlist=None, look_up_veh=None,
            grid_veh_in_ped=None, grid_TTC=None, grid_TTC_veh=None):
    '''
    The sample function
    params:
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    true_x_seq: True positions
    true_Pedlist: The true peds present in each frame
    saved_args: Training arguments
    '''

    # Number of peds in the sequence 
    # This will also include peds that do not exist in the observation length but come into scene in the prediction time interval
    numx_seq = len(look_up)   
    if look_up_veh is not None:
        numx_seq_veh = len(look_up_veh)
        
    with torch.no_grad():
        # Construct variables for hidden and cell states
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
        if not is_gru:
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if args.use_cuda:
                cell_states = cell_states.cuda()
        else:
            cell_states = None


        ret_x_seq = Variable(torch.zeros(args.obs_length+args.pred_length, numx_seq, x_seq.shape[2]))
        dist_param_seq = Variable(torch.zeros(args.obs_length+args.pred_length, numx_seq, 5))

        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()
            # dist_param_seq = dist_param_seq.cuda()

        # For the observed part of the trajectory
        for tstep in range(args.obs_length-1):
            if grid is None: 
               # Do a forward prop
                out_obs, hidden_states, cell_states = net(x_seq[tstep,:,:2].view(1, numx_seq, 2), hidden_states, cell_states, 
                                                          [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            elif (args.method == 4):
                # Do a forward prop 
                # We give the frames one by one as input. 
                # Unlike the training where we give the whole frames at once and it was iterated in the model
                grid_t = grid[tstep]
                grid_veh_in_ped_t = grid_veh_in_ped[tstep]
                grid_TTC_t = grid_TTC[tstep]
                grid_TTC_veh_t = grid_TTC_veh[tstep]
                if args.use_cuda:
                    grid_t = grid_t.cuda()
                    grid_veh_in_ped_t = grid_veh_in_ped_t.cuda()
                    grid_TTC_t = grid_TTC_t.cuda()
                    grid_TTC_veh_t = grid_TTC_veh_t.cuda()
                out_obs, hidden_states, cell_states = net(x_seq[tstep,:,:].view(1, numx_seq, x_seq.shape[2]), [grid_t],
                                                           hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader,
                                                            look_up, x_seq_veh[tstep,:,:].view(1, numx_seq_veh, x_seq_veh.shape[2]),
                                                            [grid_veh_in_ped_t], [Vehlist[tstep]], look_up_veh, [grid_TTC_t], [grid_TTC_veh_t])

            elif (args.method == 1):
                grid_t = grid[tstep]
                out_obs, hidden_states, cell_states = net(x_seq[tstep,:,:].view(1, numx_seq, x_seq.shape[2]), [grid_t.cpu()],
                                                           hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader,
                                                            look_up, x_seq_veh[tstep,:,:].view(1, numx_seq_veh, x_seq_veh.shape[2]), None,
                                                            [Vehlist[tstep]], look_up_veh)

            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(out_obs.cpu())

            # Storing the paramteres of the distriution for plotting
            dist_param_seq[tstep + 1, :, :] = out_obs.clone()

            # Sample from the bivariate Gaussian
            next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, true_Pedlist[tstep], look_up)

            ret_x_seq[tstep + 1, :, 0] = next_x
            ret_x_seq[tstep + 1, :, 1] = next_y

            # One could also assign the mean to the next state instead of sampling from the distrbution. 
            # for that one should comment the above three lines and uncomment the following lines 
            # next_x_mean = mux
            # next_y_mean = muy
            # ret_x_seq[tstep + 1, :, 0] = next_x_mean
            # ret_x_seq[tstep + 1, :, 1] = next_y_mean

    
        # Last seen grid
        if grid is not None: # not vanilla lstm
            prev_grid = grid[-1].clone()
            if (args.method == 4):
                prev_grid_veh_in_ped = grid_veh_in_ped[-1].clone()
                prev_TTC_grid = grid_TTC[-1].clone()
                prev_TTC_grid_veh = grid_TTC_veh[-1].clone()

        # constructing the speed change and deviation feautres for time step obs_length-1 
        # that is going ot be used in the next for loop and also calculating that for each new time step on the following for loop
        ret_x_seq[tstep + 1, :, 2] = x_seq[-1,:,2]
        ret_x_seq[tstep + 1, :, 3] = x_seq[-1,:,3]
        ret_x_seq[tstep + 1, :, 5] = x_seq[-1,:,5]
        ret_x_seq[tstep + 1, :, 6] = x_seq[-1,:,6]
        ret_x_seq[tstep + 1, :, 7] = x_seq[-1,:,7]
        ret_x_seq[tstep + 1, :, 8] = x_seq[-1,:,8]

        timestamp = dataloader.timestamp

        # For the predicted part of the trajectory
        for tstep in range(args.obs_length-1, args.pred_length + args.obs_length-1):
            # Do a forward prop
            if grid is None: # vanilla lstm
                outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, ret_x_seq.shape[2]), hidden_states, 
                                                          cell_states, [true_Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            else:
                if (args.method == 4):
                    outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, ret_x_seq.shape[2]), [prev_grid],
                                                               hidden_states, cell_states, [true_Pedlist[tstep]], [num_pedlist[tstep]],
                                                                dataloader, look_up, x_seq_veh[tstep,:,:].view(1, numx_seq_veh, x_seq_veh.shape[2]), 
                                                                [prev_grid_veh_in_ped], [Vehlist[tstep]], look_up_veh,  [prev_TTC_grid],
                                                                [prev_TTC_grid_veh])
                elif (args.method == 1):
                     outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, ret_x_seq.shape[2]), [prev_grid.cpu()],
                                                                hidden_states, cell_states, [true_Pedlist[tstep]], [num_pedlist[tstep]],
                                                                dataloader, look_up, x_seq_veh[tstep,:,:].view(1, numx_seq_veh, x_seq_veh.shape[2]),
                                                                None, [Vehlist[tstep]], look_up_veh)
      
            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(outputs.cpu())

            # Storing the paramteres of the distriution
            dist_param_seq[tstep + 1, :, :] = torch.stack((mux, muy, sx, sy, corr),2) 

            # Sample from the bivariate Gaussian
            next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, true_Pedlist[tstep], look_up)

            # Store the predicted position
            ret_x_seq[tstep + 1, :, 0] = next_x
            ret_x_seq[tstep + 1, :, 1] = next_y

            # One could also assign the mean to the next state instead of sampling from the distrbution. 
            # for that one should comment the above three lines and uncomment the following lines 

            # next_x_mean = mux
            # next_y_mean = muy
            # ret_x_seq[tstep + 1, :, 0] = next_x_mean
            # ret_x_seq[tstep + 1, :, 1] = next_y_mean


            # Preparing a ret_x_seq that is covnerted back to the original frame by adding the first position.
            # This will be used for grid calculation
            ret_x_seq_convert = ret_x_seq.clone() 
            ret_x_seq_convert = revert_postion_change_seq2(ret_x_seq_convert.cpu(), true_Pedlist, look_up, first_values_dict,
                                                            orig_x_seq, saved_args.obs_length, infer=True)


            ret_x_seq_convert[tstep + 1, :, 2] = (ret_x_seq_convert[tstep + 1, :, 0] - ret_x_seq_convert[tstep, :, 0]) / timestamp # vx 
            ret_x_seq_convert[tstep + 1, :, 3] = (ret_x_seq_convert[tstep + 1, :, 1] - ret_x_seq_convert[tstep, :, 1]) / timestamp # vy
            # updating the velocity data in ret_x_seq accordingly
            ret_x_seq[tstep + 1, :, 2] = ret_x_seq_convert[tstep + 1, :, 2].clone()
            ret_x_seq[tstep + 1, :, 3] = ret_x_seq_convert[tstep + 1, :, 3].clone()

            
            # claculating the rest of the features that will be used in the interaction tensor (speed change and deviation) 
            ret_x_seq_convert[tstep + 1, :, 5] = (ret_x_seq_convert[tstep + 1, :, 2] - ret_x_seq_convert[tstep, :, 2]) / timestamp # ax
            ret_x_seq_convert[tstep + 1, :, 6] = (ret_x_seq_convert[tstep + 1, :, 3] - ret_x_seq_convert[tstep, :, 3]) / timestamp # ay
            speed_next_t = torch.pow((torch.pow(ret_x_seq_convert[tstep + 1, :, 2], 2) + torch.pow(ret_x_seq_convert[tstep + 1, :, 3], 2)),0.5)
            speed_t = torch.pow((torch.pow(ret_x_seq_convert[tstep, :, 2], 2) + torch.pow(ret_x_seq_convert[tstep, :, 3], 2)),0.5)
            ret_x_seq[tstep + 1, :, 7] = speed_next_t - speed_t # speed difference
            dot_vel = torch.mul(ret_x_seq_convert[tstep, :, 2], ret_x_seq_convert[tstep+1, :, 2]) + \
                        torch.mul(ret_x_seq_convert[tstep, :, 3], ret_x_seq_convert[tstep+1, :, 3])
            det_vel = torch.mul(ret_x_seq_convert[tstep, :, 2],  ret_x_seq_convert[tstep+1, :, 3]) - \
                        torch.mul(ret_x_seq_convert[tstep, :, 3], ret_x_seq_convert[tstep+1, :, 2])
            ret_x_seq[tstep + 1, :, 8] = torch.atan2(det_vel,dot_vel) * 180/np.pi # deviation angel


            # List of x_seq at the last time-step (assuming they exist until the end)
            true_Pedlist[tstep+1] = [int(_x_seq) for _x_seq in true_Pedlist[tstep+1]]
            next_ped_list = true_Pedlist[tstep+1].copy()
            converted_pedlist = [look_up[_x_seq] for _x_seq in next_ped_list]
            list_of_x_seq = Variable(torch.LongTensor(converted_pedlist))

            #Get their predicted positions
            current_x_seq = torch.index_select(ret_x_seq_convert[tstep+1], 0, list_of_x_seq)

            if grid is not None: # not vanilla lstm

                if args.method == 4:
                    Vehlist[tstep+1] = [int(_x_seq_veh) for _x_seq_veh in Vehlist[tstep+1]]
                    next_veh_list = Vehlist[tstep+1].copy()
                    converted_vehlist = [look_up_veh[_x_seq_veh] for _x_seq_veh in next_veh_list]
                    list_of_x_seq_veh = Variable(torch.LongTensor(converted_vehlist))
                    if args.use_cuda:
                        list_of_x_seq_veh = list_of_x_seq_veh.cuda()
                    current_x_seq_veh = torch.index_select(x_seq_veh[tstep+1], 0, list_of_x_seq_veh)
          
            
                if  args.method == 1: #social lstm 
                    prev_grid = getGridMask(current_x_seq.data.cpu(), len(true_Pedlist[tstep+1]),saved_args.neighborhood_size, saved_args.grid_size) 
                  
                elif args.method == 4: # Collision grid
                    prev_grid, prev_TTC_grid = getInteractionGridMask(current_x_seq.data.cpu(), current_x_seq.data.cpu(), saved_args.TTC,
                                                                       saved_args.D_min, saved_args.num_sector)
                    prev_grid_veh_in_ped, prev_TTC_grid_veh = getInteractionGridMask(current_x_seq.data.cpu(),  current_x_seq_veh.data.cpu(),
                                                                                      saved_args.TTC_veh, saved_args.D_min_veh, 
                                                                                      saved_args.num_sector, is_heterogeneous = True,
                                                                                      is_occupancy = False)


                prev_grid = Variable(torch.from_numpy(prev_grid).float())
              
                if args.method == 4:
                    prev_grid_veh_in_ped = Variable(torch.from_numpy(prev_grid_veh_in_ped).float())
                    prev_TTC_grid = Variable(torch.from_numpy(prev_TTC_grid).float())
                    prev_TTC_grid_veh = Variable(torch.from_numpy(prev_TTC_grid_veh).float())
                    if args.use_cuda:
                        prev_grid = prev_grid.cuda()
                        prev_grid_veh_in_ped = prev_grid_veh_in_ped.cuda()
                        # if args.method == 4:
                        prev_TTC_grid = prev_TTC_grid.cuda()
                        prev_TTC_grid_veh = prev_TTC_grid_veh.cuda()


        return ret_x_seq, dist_param_seq



if __name__ == '__main__':
    main()


    

