
import argparse
import pickle
from utils.helper import *

def main():
    
    parser = argparse.ArgumentParser() 

    # method selection
    parser.add_argument('--method', type=int, default=7,
                            help='Method you want to display its test result  \
                            (1 = social-lstm, 3 = vanilla-lstm, 4 = PV-collisionGrid, 5 = Linear-regression \
                             6 = P-collisionGrid, 7 = V-collisionGrid)') 
    # Minimum acceptalbe distance between two pedestrians
    parser.add_argument('--D_min', type=int, default=0.7, 
                        help='Minimum distance for which the TTC is calculated')
    # Minimum acceptalbe distance between a pedstrian and a vehicle
    parser.add_argument('--D_min_veh', type=int, default=1.0, 
                        help='Minimum distance for which the TTC is calculated')
    args = parser.parse_args()


    file_path_PV_collisionGrid = "Store_Results/plot/test/CollisionGrid/PV-CollisionGrid/test_results.pkl"
    file_path_P_collisionGrid = "Store_Results/plot/test/CollisionGrid/P-CollisionGrid/test_results.pkl"
    file_path_V_collisionGrid = "Store_Results/plot/test/CollisionGrid/V-CollisionGrid/test_results.pkl"
    file_path_SocialLSTM = "Store_Results/plot/test/SocialLSTM/test_results.pkl"
    file_path_VLSTM = "Store_Results/plot/test/VLSTM/test_results.pkl"
    file_path_LR = "Store_Results/plot/test/LR/test_results.pkl"


    if args.method == 1:
        file_path = file_path_SocialLSTM
        print("====== Social LSTM results ======")
    elif args.method == 3:
        file_path = file_path_VLSTM
        print("====== Vanilla LSTM results ======")
    elif args.method == 4:
        file_path = file_path_PV_collisionGrid
        print("====== PV_CollisionGrid results ======")
    elif args.method == 5:
        file_path = file_path_LR
        print("====== Linear Regression results ======")
    elif args.method == 6:
        file_path = file_path_P_collisionGrid
        print("====== P_CollisionGrid results (oblation study)======")
    elif args.method == 7:
        file_path = file_path_V_collisionGrid
        print("====== V_CollisionGrid results (oblation study)======")
    else:
        raise ValueError("Invalid method number")
    
    try:
        f = open(file_path, 'rb')
    except FileNotFoundError:
        print("File not found: %s"%file_path_PV_collisionGrid)

    results = pickle.load(f)

    # print("====== The total number of data in the test set is: " + str(len(results)) + ' ========')

    ave_error = []
    final_error = []
    MHD_error = []
    speed_error = []
    heading_error = []
    num_coll_homo = []
    all_num_cases_homo = []
    num_coll_hetero = []
    all_num_cases_hetero = []

    for i in range(len(results)): # each i is one sample or batch (since batch_size is 1 during test)

        results_i = results[i]
        true_trajectories = results_i[0] 
        pred_trajectories = results_i[1]
        PedsList_seq = results_i[2]
        lookup_seq = results_i[3]
        obs_length = results_i[5]
        dist_param_seq = results_i[6]
        true_trajectories_veh = results_i[7]
        VehsList_seq = results_i[8]
        lookup_seq_veh = results_i[9]
        grid_seq = results_i[10]
        grid_seq_veh = results_i[11]

        pred_trajectories = torch.from_numpy(pred_trajectories)
        true_trajectories = torch.from_numpy(true_trajectories)
        true_trajectories_veh = torch.from_numpy(true_trajectories_veh)

        
        PedsList_obs = sum(PedsList_seq[:obs_length], [])
        error_batch = get_mean_error(pred_trajectories[obs_length:,:,:2], true_trajectories[obs_length:,:,:2],
                                      PedsList_seq[obs_length:], PedsList_obs, False, lookup_seq)
        final_error_batch = get_final_error(pred_trajectories[obs_length:,:,:2], true_trajectories[obs_length:,:,:2],
                                             PedsList_seq[obs_length:],PedsList_obs, lookup_seq)
        MHD_error_batch = get_hausdorff_distance(pred_trajectories[obs_length:,:,:2], true_trajectories[obs_length:,:,:2],
                                                  PedsList_seq[obs_length:], PedsList_obs, False, lookup_seq)
        speed_error_batch, heading_error_batch = get_velocity_errors(pred_trajectories[obs_length:,:,2:4],
                                                                      true_trajectories[obs_length:,:,2:4], 
                                                                      PedsList_seq[obs_length:], PedsList_obs, 
                                                                      False, lookup_seq)
        num_coll_homo_batch, all_num_cases_homo_batch = get_num_collisions_homo(pred_trajectories[obs_length:,:,:2],
                                                                                 PedsList_seq[obs_length:], PedsList_obs, 
                                                                                 False, lookup_seq, args.D_min)
        num_coll_hetero_batch, all_num_cases_hetero_batch = get_num_collisions_hetero(pred_trajectories[obs_length:,:,:2],
                                                                                       PedsList_seq[obs_length:], PedsList_obs,
                                                                                        False, lookup_seq,
                                                                                        true_trajectories_veh[obs_length:,:,:2],
                                                                                        VehsList_seq[obs_length:], lookup_seq_veh,
                                                                                        args.D_min_veh)

        ave_error.append(error_batch)
        final_error.append(final_error_batch)
        MHD_error.append(MHD_error_batch)
        speed_error.append(speed_error_batch)
        heading_error.append(heading_error_batch)
       
        num_coll_homo.append(num_coll_homo_batch)
        all_num_cases_homo.append(all_num_cases_homo_batch)
        num_coll_hetero.append(num_coll_hetero_batch)
        all_num_cases_hetero.append(all_num_cases_hetero_batch)


    print('Average displacement error (ADE) of the model is: ', round(sum(ave_error).item() / len(results),4)) 
    print('Final displacement error (FDE) of the model is: ', round(sum(final_error).item() / len(results), 4))
    print('Hausdorff distance error (MHD) of the model is: ', round(sum(MHD_error).item() / len(results), 4))
    print('Speed error (SE) of the model is: ', round((sum(speed_error).item() / len(results))**0.5, 4))
    print('Average heading error (HE) of the model is: ', round((sum(heading_error).item() /len(results))**0.5, 2))
    # print('Overll percentage of collision of the model is ', 
    #       round((sum(num_coll_homo)+sum(num_coll_hetero))/(sum(all_num_cases_homo)+sum(all_num_cases_hetero)) * 100, 4))
    # print('Percentage of collision between pedestrians of the model is ',
    #       round(sum(num_coll_homo)/sum(all_num_cases_homo) * 100, 4))
    # print('Percentage of collision between pedestrians and vehicles of the model is ',
    #       round(sum(num_coll_hetero)/sum(all_num_cases_hetero) * 100, 4))


if __name__ == '__main__':
    main()


