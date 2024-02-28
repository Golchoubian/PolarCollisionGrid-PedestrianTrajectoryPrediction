import torch 
from utils.DataLoader import DataLoader
from utils.helper import *
from utils.grid import getSequenceGridMask, getSequenceGridMask_heterogeneous 
from utils.Interaction import getInteractionGridMask, getSequenceInteractionGridMask
from torch.autograd import Variable
import time
import argparse
import os
import pickle
from visualization import Loss_Plot 

'''
## Acknowledgements
This project is builds upon the codebase from [social-lstm](https://github.com/quancore/social-lstm),
developed by "quancore" as a pytorch implementation of the Social LSTM model proposed by Alahi et al.
The Social LSTM model itself is also used as a baseline for comparison with our propsed CollisionGrid model.
'''


def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')
    # Length of sequence to be considered
    parser.add_argument('--seq_length', type=int, default=12, # 12 for HBS (obs: 6, pred: 6)
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=6,
                        help='prediction length')
    parser.add_argument('--obs_length', type=int, default=6,
                        help='Observed length of the trajectory')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=200, 
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now.
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.5, 
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')

    # Dimension of the embeddings parameter for actions
    parser.add_argument('--embedding_size_action', type=int, default=32,
                        help='Embedding dimension for the actions')

    # For the SocialLSTM:
    # Size of neighborhood to be considered parameter # 
    parser.add_argument('--neighborhood_size', type=int, default=8,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')


    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=27,
                        help='Maximum Number of Pedestrians')

    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    # Cuda parameter
    parser.add_argument('--use_cuda', action="store_true", default=True,
                        help='Use GPU or not')
    # GRU parameter
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # drive option
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # number of validation will be used
    parser.add_argument('--num_validation', type=int, default=2,
                        help='Total number of validation dataset for validate accuracy')
    # frequency of validation
    parser.add_argument('--freq_validation', type=int, default=1,
                        help='Frequency number(epoch) of validation using validation data')
    # frequency of optimazer learning decay
    parser.add_argument('--freq_optimizer', type=int, default=8,
                        help='Frequency number(epoch) of learning decay for optimizer')
    # store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
    parser.add_argument('--store_grid', action="store_true", default=True, 
                        help='Whether store grids and use further epoch')

    # Size of neighborhood for vehilces in pedestrians grid 
    parser.add_argument('--neighborhood_size_veh_in_ped', type=int, default=64,
                        help='Neighborhood size to be considered for social grid (the grid that considers vehicles)')
    # Size of the social grid parameter for vehilces in pedestrians grid
    parser.add_argument('--grid_size_veh_in_ped', type=int, default=4,
                        help='Grid size of the social grid (the grid that considers vehicles)')

    # The lateral size of the social grid, the number of divisions of the circle around the agent for specifying the approach angle
    parser.add_argument('--num_sector', type=int, default=8, 
                        help='The number of circle division for distinguishing approach angle')
    # Minimum time to collisions to be considered, the num of TTC is the radial size of the social grid mask
    parser.add_argument('--TTC', type=int, default=[9], # [10]
                        help='Minimum time to collisions to be considerd for the social grid')
    # Minimum acceptalbe distance between two pedestrians
    parser.add_argument('--D_min', type=int, default=0.7, 
                        help='Minimum distance for which the TTC is calculated')
    # Minimum time to collisions to be considered for ped-veh interaction, the num of TTC is the radial size of the social grid mask of veh in ped
    parser.add_argument('--TTC_veh', type=int, default=[8],
                        help='Minimum time to collisions to be considerd for the social grid')
    # Minimum acceptalbe distance between a pedstrian and a vehicle
    parser.add_argument('--D_min_veh', type=int, default=1.0,
                        help='Minimum distance for which the TTC is calculated')
    # method selection
    parser.add_argument('--method', type=int, default=4,
                            help='Method of lstm will be used (1 = social lstm, 3 = vanilla lstm, 4 = collision grid)') 


    args = parser.parse_args()

    train(args)



def train(args):


    model_name = "LSTM"
    method_name = "SOCIALLSTM" # Attention: This name has not been changed for different models used. (ToDO later)
    save_tar_name = method_name+"_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

     # Log directory
    prefix = 'Store_Results/'
    log_directory = os.path.join(prefix, 'log/')
    plot_directory = os.path.join(prefix, 'plot/') 

    # Logging files
    log_file_curve = open(os.path.join(log_directory,'log_curve.txt'), 'w+')

    # model directory
    save_directory = os.path.join(prefix, 'model/') 

    # Save the arguments in the config file
    with open(os.path.join(save_directory,'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file 
    def checkpoint_path(x):
        return os.path.join(save_directory, save_tar_name+str(x)+'.tar')



    # Create the data loader object. This object would preprocess the data in terms of
    # batches each of size args.batch_size, and of length args.seq_length
    dataloader = DataLoader(args.batch_size, args.seq_length, infer=False, filtering=True)


    # model creation
    net = get_model(args.method,args)
    if args.use_cuda:
        net = net.cuda()

    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, weight_decay=args.lambda_param)
    # optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param) 
    # optimizer = torch.optim.Adam(net.parameters(), weight_decay=args.lambda_param)


    if args.store_grid:
        print("////////////////////////////")
        print("Starting the off line grid caculation all at once")
        grid_cal_start = time.time()
        dataloader.grid_creation(args)
        grid_cal_end = time.time()
        print("grid calculation is finished")
        print("grid calculation time for all the data: {} seconds".format(grid_cal_end - grid_cal_start))  
        print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\")

    num_batch = 0


    start_train_loop = time.time()
    err_batch_list = []
    loss_batch_list = []
    train_batch_num_list = []
    loss_epoch_list = []
    err_epoch_list = []

    # Training
    for epoch in range(args.num_epochs):
        
        print('**************** Training epoch beginning ******************')
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0
        err_epoch = 0

        # changing the order of the sequence if shuffle in on
        x_WholeBatch, numPedsList_WholeBatch, PedsList_WholeBatch, x_veh_WholeBatch, numVehsList_WholeBatch, \
            VehsList_WholeBatch, grids_WholeBatch, grids_veh_WholeBatch, grids_TTC_WholeBatch, grids_TTC_veh_WholeBatch = \
                dataloader.batch_creater(args.store_grid, args.method, suffle=True)

        # For each batch
        for batch in range(dataloader.num_batches):
            start = time.time()

            # Get batch data
            x, numPedsList, PedsList = x_WholeBatch[batch], numPedsList_WholeBatch[batch], PedsList_WholeBatch[batch]
            x_veh, numVehsList, VehsList = x_veh_WholeBatch[batch], numVehsList_WholeBatch[batch], VehsList_WholeBatch[batch]
            if args.store_grid:
                grids_batch, grids_veh_batch = grids_WholeBatch[batch], grids_veh_WholeBatch[batch]
                if (args.method == 4):
                    grids_TTC_batch, grids_TTC_veh_batch = grids_TTC_WholeBatch[batch], grids_TTC_veh_WholeBatch[batch]


            loss_batch = 0
            err_batch = 0

            # Zero out gradients
            net.zero_grad()
            optimizer.zero_grad()

            # For each sequence
            for sequence in range(dataloader.batch_size):
              
                x_seq , numPedsList_seq, PedsList_seq = x[sequence], numPedsList[sequence], PedsList[sequence]
                x_seq_veh , numVehsList_seq, VehsList_seq = x_veh[sequence], numVehsList[sequence], VehsList[sequence]

                #dense vector creation
                x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq) 
                # order of featurs in x_seq: x, y, vx, vy, timestamp, ax, ay 
                x_seq_veh, lookup_seq_veh = dataloader.convert_proper_array(x_seq_veh, numVehsList_seq, VehsList_seq, veh_flag=True)
              
                x_seq_orig = x_seq.clone()
                x_seq_veh_orig = x_seq_veh.clone()

                if args.store_grid:
                   grid_seq =  grids_batch[sequence]
                   grid_seq_veh_in_ped = grids_veh_batch[sequence]
                   if args.method == 4:
                    grid_TTC_seq = grids_TTC_batch[sequence]
                    grid_TTC_veh_seq = grids_TTC_veh_batch[sequence]
                
                else:
                    if args.method == 1: # Social LSTM
                        grid_seq = getSequenceGridMask(x_seq, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda, lookup_seq)
                        grid_seq_veh_in_ped = getSequenceGridMask_heterogeneous(x_seq, PedsList_seq, x_seq_veh, VehsList_seq,
                                                                                 args.neighborhood_size_veh_in_ped, args.grid_size_veh_in_ped, 
                                                                                 args.use_cuda, lookup_seq, lookup_seq_veh, False)
                    
                    elif args.method ==4: # CollisionGird
                        grid_seq, grid_TTC_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq, PedsList_seq, args.TTC,
                                                                                args.D_min, args.num_sector, args.use_cuda, 
                                                                                lookup_seq, lookup_seq)
                        grid_seq_veh_in_ped, grid_TTC_veh_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq_veh, VehsList_seq,
                                                                                                args.TTC_veh, args.D_min_veh, args.num_sector,
                                                                                                args.use_cuda, lookup_seq, lookup_seq_veh,
                                                                                                is_heterogeneous=True, is_occupancy=False) 
                        
                x_seq, first_values_dict = position_change_seq(x_seq, PedsList_seq, lookup_seq)
                x_seq_veh, first_values_dict_veh = position_change_seq(x_seq_veh, VehsList_seq, lookup_seq_veh)

                if args.use_cuda:                    
                    x_seq = x_seq.cuda()
                    x_seq_veh = x_seq_veh.cuda()

                y_seq = x_seq[1:,:,:2]
                x_seq = x_seq[:-1,:,:]
                numPedsList_seq = numPedsList_seq[:-1]
             
                y_seq_veh = x_seq_veh[1:,:,:2]
                x_seq_veh = x_seq_veh[:-1,:,:]
                numVehsList_seq = numVehsList_seq[:-1]
               

                if args.method != 3: # not Vanilla LSTM 
                    grid_seq_plot = grid_seq[1:]
                    grid_seq_veh_plot = grid_seq_veh_in_ped[1:]

                    grid_seq = grid_seq[:-1]
                    grid_seq_veh_in_ped = grid_seq_veh_in_ped[:-1]
                    
                if args.method == 4:
                    grid_TTC_seq = grid_TTC_seq[:-1]
                    grid_TTC_veh_seq = grid_TTC_veh_seq[:-1]

                #number of peds in this sequence per frame
                numNodes = len(lookup_seq) 


                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    hidden_states = hidden_states.cuda()

                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    cell_states = cell_states.cuda()

                # Forward prop
                if args.method == 3: # Vanillar LSTM
                    outputs, _, _ = net(x_seq, hidden_states, cell_states, PedsList_seq[:-1], numPedsList_seq ,dataloader, lookup_seq) 
                elif args.method == 4: # Collision Grid
                    outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, PedsList_seq[:-1], numPedsList_seq ,dataloader,
                                            lookup_seq, x_seq_veh, grid_seq_veh_in_ped, VehsList_seq[:-1], lookup_seq_veh, grid_TTC_seq,
                                            grid_TTC_veh_seq) 
                elif args.method == 1: # Social LSTM
                    outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, PedsList_seq[:-1], numPedsList_seq ,dataloader, 
                                            lookup_seq, x_seq_veh, grid_seq_veh_in_ped, VehsList_seq[:-1], lookup_seq_veh) 
                else:
                    raise ValueError("Method is not defined")
              
                # Compute loss
                loss = Gaussian2DLikelihood(outputs, y_seq, PedsList_seq[1:], lookup_seq)
                loss = loss / dataloader.batch_size
                loss_batch += loss.item()

                # Compute gradients
                # Cumulating gradient until we reach our required batch size and then updating one the weights
                loss.backward()
            
                # # Clip gradients 
                # torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                err, pred_seq = sequence_error(outputs.cpu(), x_seq_orig[1:,:,:2], PedsList_seq[1:], lookup_seq, args.use_cuda,
                                                first_values_dict, args.obs_length)
                err_batch += err.item()


            # Update parameters
            optimizer.step()
            
            end = time.time()
            loss_batch = loss_batch 
            err_batch = err_batch / dataloader.batch_size
            err_batch_list.append(err_batch)
            loss_batch_list.append(loss_batch)
            loss_epoch += loss_batch
            err_epoch += err_batch
            num_batch+=1

            print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                    args.num_epochs * dataloader.num_batches,
                                                                                    epoch,
                                                                                    loss_batch, end - start))

            train_batch_num = epoch * dataloader.num_batches + batch
            train_batch_num_list.append(train_batch_num)
            if (train_batch_num%50 == 0):
                Loss_Plot(train_batch_num_list, err_batch_list, loss_batch_list, "loss_plot_batch", "training batch number")

        loss_epoch /= dataloader.num_batches
        err_epoch /= dataloader.num_batches
        loss_epoch_list.append(loss_epoch)
        err_epoch_list.append(err_epoch)
        Loss_Plot(range(epoch+1), err_epoch_list, loss_epoch_list, "loss_plot_epoch", "epoch")

        # Log loss values
        log_file_curve.write("Training epoch: "+str(epoch)+" loss: "+str(loss_epoch)+" error: "+str(err_epoch)+'\n')

        # Save the model after each epoch, with a file name that has the number of epoch at the end of the name (x)
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    end_train_loop = time.time()
    train_time = end_train_loop - start_train_loop
    print("The whole trainig time for {} iteraction was {} seconds".format(args.num_epochs,train_time))


if __name__ == '__main__':
    main()