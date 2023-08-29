
import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import string

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_trajecotries(true_trajectories,pred_trajectories,pred_trajectories_SocialLSTM,
                      pred_trajectories_VLSTM,pred_trajectories_LR,obs_len,batch,
                      dist_param_seq,PedsList_seq,lookup,true_trajectories_veh=None,
                      VehsList_seq=None,lookup_seq_veh=None, grid_seq=None, grid_seq_veh=None,
                      grid_seq_SocialLSTM=None, is_train=False, frame_i=5):

    num_ped = pred_trajectories.shape[1]
    seq_len = pred_trajectories.shape[0]
    if true_trajectories_veh is not None:
        num_veh = true_trajectories_veh.shape[1]
    else:
        num_veh = 0
   

    plt.figure()
    plt.xlabel("x (m)", fontsize=12)
    plt.ylabel("y (m)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    min_x = pred_trajectories[:,:,0].min() - 10
    max_x = pred_trajectories[:,:,0].max() + 10
    min_y = pred_trajectories[:,:,1].min() - 10
    max_y = pred_trajectories[:,:,1].max() + 10

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)


    for agent_index in range(num_ped): # for each agent plotting its trajecotry for the frames the agent is present

        # finding the id of the agent usign the lookup dictionary, 
        # this is the opposite of what we were usally doing (geting the index from the agent_id in ped_list_seq)
        id_list = list(lookup.keys())
        index_list = list(lookup.values())
        position = index_list.index(agent_index)
        agent_id = id_list[position]

        # Plotting the observation part
        # frames that the agent is present 
        obs_frame_nums = []
        for frame in range(0,obs_len):
            if agent_id in PedsList_seq[frame]:
                obs_frame_nums.append(frame)

        pred_frame_nums = []
        for frame in range(obs_len-1,seq_len):
            if agent_id in PedsList_seq[frame]:
                pred_frame_nums.append(frame)

            
        ### Predictions

        max_size = 6
        min_size = 1
        # prediction of CollisionGrid
        for f in pred_frame_nums:
            marker_size = min_size + ((max_size-min_size)/seq_len * f)
            plt.plot(pred_trajectories[f,agent_index,0], pred_trajectories[f,agent_index,1],
                      c='b', ls=":", marker='d', markersize=marker_size)
        plt.plot(pred_trajectories[pred_frame_nums,agent_index,0], pred_trajectories[pred_frame_nums,agent_index,1],
                  c='b', ls="-", linewidth=1.0)
        
    
        # prediction of SocialLSTM
        for f in pred_frame_nums:
            marker_size = min_size + ((max_size-min_size)/seq_len * f)
            plt.plot(pred_trajectories_SocialLSTM[f,agent_index,0], pred_trajectories_SocialLSTM[f,agent_index,1],
                      c='r', ls=":", marker='^', markersize=marker_size)
        plt.plot(pred_trajectories_SocialLSTM[pred_frame_nums,agent_index,0], pred_trajectories_SocialLSTM[pred_frame_nums,agent_index,1],
                  c='r', ls="-", linewidth=1.0)

        # prediction of Vanilla LSTM
        for f in pred_frame_nums:
            marker_size = min_size + ((max_size-min_size)/seq_len * f)
            plt.plot(pred_trajectories_VLSTM[f,agent_index,0], pred_trajectories_VLSTM[f,agent_index,1],
                      c='g', ls=":", marker='p', markersize=marker_size)
        plt.plot(pred_trajectories_VLSTM[pred_frame_nums,agent_index,0], pred_trajectories_VLSTM[pred_frame_nums,agent_index,1], 
                    c='g', ls="-", linewidth=1.0) 

        # prediction of LR
        for f in pred_frame_nums[1:]:
            marker_size = min_size + ((max_size-min_size)/seq_len * f)
            plt.plot(pred_trajectories_LR[f,agent_index,0], pred_trajectories_LR[f,agent_index,1],
                      c='y', ls=":", marker='P', markersize=marker_size)
        plt.plot(pred_trajectories_LR[pred_frame_nums,agent_index,0], pred_trajectories_LR[pred_frame_nums,agent_index,1],
                  c='y', ls="-", linewidth=1.0) 
        
        
        ### Ground truth:
        alpha_val = 1.0 # transparancy value
        all_frame = obs_frame_nums + pred_frame_nums
        plt.plot(true_trajectories[all_frame,agent_index,0], true_trajectories[all_frame,agent_index,1],
                  c='0.0', linewidth=2.0, alpha=alpha_val)

      
    max_size_veh = 6
    min_size_veh = 2

    for veh_ind in range(num_veh):

        id_list_veh = list(lookup_seq_veh.keys())
        index_list_veh = list(lookup_seq_veh.values())
        position_veh = index_list_veh.index(veh_ind)
        veh_id = id_list_veh[position_veh]
        pres_frame_nums = []
        for frame in range(seq_len):
            if veh_id in VehsList_seq[frame]:
                pres_frame_nums.append(frame)
                marker_size = min_size_veh + ((max_size_veh-min_size_veh)/seq_len * frame)
                plt.plot(true_trajectories_veh[frame,veh_ind,0], true_trajectories_veh[frame,veh_ind,1], 
                         c='0.3', marker='o', markersize=marker_size)
        plt.plot(true_trajectories_veh[pres_frame_nums,veh_ind,0], true_trajectories_veh[pres_frame_nums,veh_ind,1], 
                 c='0.3', linewidth=2.0)

    
    # =========================================================================================================================================
    # =============================================================== Neigbors ================================================================
    # =========================================================================================================================================

    alphabet = list(string.ascii_uppercase)
    label_font_size = 12


    if grid_seq != None:

        ego_agent_indx_in_pedlist = 0

        for indx, nodes_pre in enumerate(PedsList_seq[frame_i]):
            agent_i = lookup[nodes_pre]
            label = "Ped " + alphabet[indx]
            if indx == ego_agent_indx_in_pedlist: # This is our ego agent for which we want to plot its neighbours
                ego_agent = agent_i # index number of the ego agent in the trajectroy data tesnor of the sequence 
                plt.plot(pred_trajectories[frame_i, agent_i,0], pred_trajectories[frame_i, agent_i,1], 
                         color='b', marker="*", markersize=11)
            
            elif (any(grid_seq[frame_i][ego_agent_indx_in_pedlist,indx,:])): # This agent is in the grid of the ego agent.
                plt.plot(pred_trajectories[frame_i, agent_i,0], pred_trajectories[frame_i, agent_i,1], 
                         color='b', marker="s", markersize=7)
              
            else: # This agent is presnet but not in the gird of the ego agent
                plt.plot(pred_trajectories[frame_i, agent_i,0], pred_trajectories[frame_i, agent_i,1], 
                         markerfacecolor='none', markeredgecolor='b', marker="s", markersize=7)
            


        for indx, nodes_pre in enumerate(VehsList_seq[frame_i]):
            label_veh = "Veh " + alphabet[indx]
            agent_i = lookup_seq_veh[nodes_pre]
            if (any(grid_seq_veh[frame_i][ego_agent_indx_in_pedlist,indx,:])): # This agent is in the grid of the ego agent.
                plt.plot(true_trajectories_veh[frame_i, agent_i,0], true_trajectories_veh[frame_i, agent_i,1], 
                         color='0.3', marker="s", markersize=7)
           
            else: # This agent is presnet but not in the gird of the ego agent
                plt.plot(true_trajectories_veh[frame_i, agent_i,0], true_trajectories_veh[frame_i, agent_i,1], 
                         markerfacecolor='none', markeredgecolor='0.3', marker="s", markersize=7)
               

    # legends
    plt.plot(-100,-100, c='b', marker='d', label='Collision Grid')
    plt.plot(-100,-100, c='r', marker='^', label='Social LSTM')
    plt.plot(-100,-100, c='g', marker='p', label='Vanilla LSTM')
    plt.plot(-100,-100, c='y', marker='P', label='Linear Regression')
    plt.plot(-100,-100, c='0.0', marker='_', label='Ground truth')
    plt.legend(loc="lower right", prop={'size': 13}, ncol=1)
    
    if is_train:
        plt.savefig("Store_Results/plot/train/plt/compare/%d.png"%batch, dpi=200)
    else:
        plt.savefig("Store_Results/plot/test/plt/compare/%d.png"%batch, dpi=200)
    plt.close()


def Loss_Plot(train_batch_num, error_batch, loss_batch, file_name, x_axis_label):

    
    plt.subplot(2,1,1)
    plt.plot(train_batch_num, error_batch, 'b', linewidth=2.0, label="error")
    plt.ylabel("error")

    plt.subplot(2,1,2)
    plt.plot(train_batch_num, loss_batch, 'k', linewidth=2.0, label="loss")
    plt.xlabel(x_axis_label)
    plt.ylabel("loss")

    plt.savefig("Store_Results/plot/train/"+file_name+".png")
    plt.close()
 


def main():


    file_path_collisionGrid = "Store_Results/plot/test/CollisionGrid/test_results.pkl"
    file_path_SocialLSTM = "Store_Results/plot/test/SocialLSTM/test_results.pkl"
    file_path_VLSTM = "Store_Results/plot/test/VLSTM/test_results.pkl"
    file_path_LR = "Store_Results/plot/test/LR/test_results.pkl"

    try:
        f_collisionGrid = open(file_path_collisionGrid, 'rb')
    except FileNotFoundError:
        print("File not found: %s"%file_path_collisionGrid)

    try:
        f_SocialLSTM = open(file_path_SocialLSTM, 'rb')
    except FileNotFoundError:
        print("File not found: %s"%file_path_SocialLSTM)

    try:
        f_VLSTM = open(file_path_VLSTM, 'rb')
    except FileNotFoundError:
        print("File not found: %s"%file_path_VLSTM)

    try:
        f_LR = open(file_path_LR, 'rb')
    except FileNotFoundError:
        print("File not found: %s"%file_path_LR)

    results = pickle.load(f_collisionGrid)
    results_SocialLSTM = pickle.load(f_SocialLSTM)
    results_VLSTM = pickle.load(f_VLSTM)
    results_LR = pickle.load(f_LR)

    print("====== The total number of data in the test set is: " + str(len(results)) + ' ========')

    for i in range(0, len(results), 50): # plotting the samples in the test set

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

        results_SocialLSTM_i = results_SocialLSTM[i]
        pred_trajectories_SocialLSTM = results_SocialLSTM_i[1]
        grid_seq_SocialLSTM = results_SocialLSTM_i[10]

        results_VLSTM_i = results_VLSTM[i]
        pred_trajectories_VLSTM = results_VLSTM_i[1]

        results_LR_i = results_LR[i]
        pred_trajectories_LR = results_LR_i[1]


        plot_trajecotries(true_trajectories,pred_trajectories,pred_trajectories_SocialLSTM,
                          pred_trajectories_VLSTM,pred_trajectories_LR,obs_length,i,
                          dist_param_seq,PedsList_seq,lookup_seq,true_trajectories_veh,
                          VehsList_seq,lookup_seq_veh, grid_seq,grid_seq_veh,grid_seq_SocialLSTM)

if __name__ == '__main__':
    main()


