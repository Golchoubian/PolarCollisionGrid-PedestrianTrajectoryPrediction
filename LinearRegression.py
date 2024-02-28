import argparse
import os
import time 
import torch
from torch.autograd import Variable
from utils.DataLoader import DataLoader
from utils.helper import * 
from sklearn.linear_model import LinearRegression

def main():

    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=6, # 6 for HBS
                            help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=6, # 6 for HBS
                            help='Predicted length of the trajectory')
    parser.add_argument('--method', type=int, default=3, # we keep this 3 as for vanilla lstm to not consider grid in the dataloader when loading the dataset
                            help='Method of lstm will be used (1 = social lstm, 3 = vanilla lstm, 4 = collision grid)') 

    
    # Minimum acceptalbe distance between two pedestrians
    parser.add_argument('--D_min', type=int, default=0.7, 
                        help='Minimum distance for which the TTC is calculated')
    # Minimum acceptalbe distance between a pedstrian and a vehicle
    parser.add_argument('--D_min_veh', type=int, default=1.0, 
                        help='Minimum distance for which the TTC is calculated')


    sample_args = parser.parse_args()

    seq_length = sample_args.obs_length + sample_args.pred_length

    dataloader = DataLoader(1, seq_length, infer=True, filtering=True)
    dataloader.reset_batch_pointer()

    # Define the path for the config file for saved args
    prefix = 'Store_Results/'
 
    plot_directory = os.path.join(prefix, 'plot/')
    plot_test_file_directory = 'test'
    
    results = []
    
    # Variable to maintain total error
    total_error = 0
    final_error = 0
    MHD_error = 0
    speed_error = 0
    heading_error = 0

    num_coll_homo = 0
    all_num_cases_homo = 0
    num_coll_hetero = 0
    all_num_cases_hetero = 0

    x_WholeBatch, numPedsList_WholeBatch, PedsList_WholeBatch, x_veh_WholeBatch, \
                  numVehsList_WholeBatch, VehsList_WholeBatch, _, _, _, _ = \
                        dataloader.batch_creater(False, sample_args.method, suffle=False)

    None_count = 0
    for batch in range(dataloader.num_batches):
        start = time.time()
        # Get data
        x, numPedsList, PedsList = x_WholeBatch[batch], numPedsList_WholeBatch[batch], PedsList_WholeBatch[batch]
        x_veh, numVehsList, VehsList = x_veh_WholeBatch[batch], numVehsList_WholeBatch[batch], VehsList_WholeBatch[batch]

        x_seq, numPedsList_seq, PedsList_seq = x[0], numPedsList[0], PedsList[0] 
        x_seq_veh , numVehsList_seq, VehsList_seq = x_veh[0], numVehsList[0], VehsList[0]

        #dense vector creation
        x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
        x_seq_veh, lookup_seq_veh = dataloader.convert_proper_array(x_seq_veh, numVehsList_seq, VehsList_seq, veh_flag=True)
            
        #will be used for error calculation
        orig_x_seq = x_seq.clone() 
        orig_x_seq_veh = x_seq_veh.clone()
        
        num_agent = x_seq.shape[1]
        ret_x_seq = np.zeros((seq_length, num_agent, 4))

        for agent in range(num_agent):
            
            curr_agent_x = x_seq[:sample_args.obs_length,agent,0].reshape(-1,1)
            curr_agent_y = x_seq[:sample_args.obs_length,agent,1].reshape(-1,1)
            t_obs = np.arange(sample_args.obs_length).reshape(-1,1)
            t = np.arange(seq_length).reshape(-1,1)

            model_x = LinearRegression().fit(t_obs,curr_agent_x)
            model_y = LinearRegression().fit(t_obs,curr_agent_y)

            pred_x= np.squeeze(model_x.predict(t))
            pred_y= np.squeeze(model_y.predict(t))

            ret_x_seq[:,agent,0] = pred_x
            ret_x_seq[:,agent,1] = pred_y

            ret_x_seq[:,agent,2] = np.ediff1d(pred_x, to_begin=(pred_x[1]-pred_x[0])) / dataloader.timestamp # v_x
            ret_x_seq[:,agent,3] = np.ediff1d(pred_y, to_begin=(pred_y[1]-pred_y[0])) / dataloader.timestamp # v_y

        # Getting the agent_ids of those present in the observation section. 
        # Error should be calculated only for those agents that their data exists in the observation part
        ret_x_seq = torch.from_numpy(ret_x_seq)
        PedsList_obs = sum(PedsList_seq[:sample_args.obs_length], []) 

        total_error += get_mean_error(ret_x_seq[sample_args.obs_length:,:,:2].data, orig_x_seq[sample_args.obs_length:,:,:2].data,
                                       PedsList_seq[sample_args.obs_length:], PedsList_obs, False, lookup_seq)
        final_error += get_final_error(ret_x_seq[sample_args.obs_length:,:,:2].data, orig_x_seq[sample_args.obs_length:,:,:2].data,
                                    PedsList_seq[sample_args.obs_length:],PedsList_obs, lookup_seq)
        MHD_error_batch = get_hausdorff_distance(ret_x_seq[sample_args.obs_length:,:,:2].data, orig_x_seq[sample_args.obs_length:,:,:2].data,
                                                  PedsList_seq[sample_args.obs_length:], PedsList_obs, False, lookup_seq)
        speed_error_batch, heading_error_batch = get_velocity_errors(ret_x_seq[sample_args.obs_length:,:,2:4].data,
                                                                      orig_x_seq[sample_args.obs_length:,:,2:4].data,
                                                                      PedsList_seq[sample_args.obs_length:], PedsList_obs, False, lookup_seq)
        
        num_coll_homo_batch, all_num_cases_homo_batch = get_num_collisions_homo(ret_x_seq[sample_args.obs_length:,:,:2].data,
                                                                                 PedsList_seq[sample_args.obs_length:], PedsList_obs, 
                                                                                 False, lookup_seq, sample_args.D_min)
        num_coll_hetero_batch, all_num_cases_hetero_batch = get_num_collisions_hetero(ret_x_seq[sample_args.obs_length:,:,:2].data,
                                                                                       PedsList_seq[sample_args.obs_length:], 
                                                                                       PedsList_obs, False, lookup_seq,
                                                                                       x_seq_veh[sample_args.obs_length:,:,:2].cpu().data,
                                                                                       VehsList_seq[sample_args.obs_length:], lookup_seq_veh,
                                                                                       sample_args.D_min_veh)

        
        if (MHD_error_batch != None):
            MHD_error += MHD_error_batch
        else:
            None_count += 1
        speed_error += speed_error_batch
        heading_error += heading_error_batch

        num_coll_homo += num_coll_homo_batch
        all_num_cases_homo += all_num_cases_homo_batch
        num_coll_hetero += num_coll_hetero_batch
        all_num_cases_hetero += all_num_cases_hetero_batch

        end = time.time()

        print('Current file : ', dataloader.get_file_name(0),' Processed trajectory number : ', batch+1, 'out of', dataloader.num_batches,
               'trajectories in time', end - start)

        results.append((orig_x_seq.data.cpu().numpy(), ret_x_seq.data.cpu().numpy(), PedsList_seq, lookup_seq , None, sample_args.obs_length, [],
                         orig_x_seq_veh.data.cpu().numpy(), VehsList_seq, lookup_seq_veh, None, None))

    print("==================================================")
    print("==================================================")
    print("==================================================")
    print('ADE: ', total_error / dataloader.num_batches)
    print('FDE: ', final_error / dataloader.num_batches)
    print('MHD: ', MHD_error / (dataloader.num_batches-None_count))
    print('Speed error: ', (speed_error / dataloader.num_batches)**0.5)
    print('Heading error: ', (heading_error / dataloader.num_batches)**0.5)
    # print('Collision percentage: ', (num_coll_homo+num_coll_hetero)/(all_num_cases_homo+all_num_cases_hetero) * 100)
    # print('Collision percentage (ped-ped): ', num_coll_homo/all_num_cases_homo * 100)
    # print('Collision percentage (ped-veh): ', num_coll_hetero/all_num_cases_hetero * 100)
    dataloader.write_to_plot_file(results, os.path.join(plot_directory, plot_test_file_directory)) 


if __name__ == '__main__':
    main()


    

