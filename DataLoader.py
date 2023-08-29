
import os
import pandas as pd
import numpy as np
import pickle
import math
from torch.autograd import Variable
import torch
import random

from grid import getSequenceGridMask, getSequenceGridMask_heterogeneous 
from Interaction import getInteractionGridMask, getSequenceInteractionGridMask

class DataLoader():

    def __init__(self, batch_size=5, seq_length=20, infer=False, filtering=False):


        self.downsample_step = 1 # consider the data every x steps
        fps = 2 / self.downsample_step # (CITR: 29.97, DUT:23.98, ETH/UCY: 2.5, HBS: 2)
        self.timestamp = 1 / fps

        base_dataset = ['Data/HBS/hbs.csv']

        self.base_data_dirs = base_dataset

        # Number of datasets
        self.numDatasets = len(self.base_data_dirs)

        # Store the arguments
        self.infer = infer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.orig_seq_lenght = seq_length

        # array for keepinng target ped ids for each sequence
        self.target_ids = []
        self.veh_ids = []
        
        # Define the path in which the process data would be stored
        self.data_file = os.path.join('Data/train/', "trajectories.cpkl")
        self.data_file_tr = os.path.join('Data/train/', "trajectories_train.cpkl")        
        self.data_file_te = os.path.join('Data/test/', "trajectories_test.cpkl")
        self.data_file_vl = os.path.join('Data/validation/', "trajectories_val.cpkl")

        if not(os.path.exists(self.data_file_te)):
            #all the files will be created at the first run. So if one doesn't exist it means that none exists

            print("Creating pre-processed data from raw data")
            self.frame_preprocess(self.base_data_dirs, self.data_file)

            # Reset all the data pointers of the dataloader object
            self.reset_batch_pointer(valid=False) 
            self.reset_batch_pointer(valid=True) 
            self.load_preprocessed(self.data_file, self.data_file_tr, self.data_file_te, infer)
            self.load_preprocessed2(self.data_file_tr, filtering)

        elif (infer == False):
            self.reset_batch_pointer(valid=False) 
            self.reset_batch_pointer(valid=True) 
            self.load_preprocessed2(self.data_file_tr, filtering)
        elif (infer == True):
            self.reset_batch_pointer(valid=False) 
            self.reset_batch_pointer(valid=True) 
            self.load_preprocessed2(self.data_file_te, filtering)



    def frame_preprocess(self, data_dirs, data_file):
        '''
        Function that will pre-process the csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file_tr : The file into which all the pre-processed training data needs to be stored
        data_file_te : The file into which all the pre-processed testing data needs to be stored

        '''
        # all_frame_data would be a list of list of numpy arrays corresponding to each dataset
        # Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row
        # containing pedID, x, y
        all_frame_data = []
        # Validation frame data
        valid_frame_data = []
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []
        valid_numPeds_data= []
        # numPeds_data would be a list of lists corresponding to each dataset
        # Each list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []
        

        #each list includes ped ids of this frame
        pedsList_data = []
        valid_pedsList_data = []
        # target ped ids for each sequence
        target_ids = []
        orig_data = []

        # creating the same lists for the vehicle data
        all_frame_data_veh = []
        numVehs_data = []
        vehsList_data = []
        target_ids_veh = []
        orig_data_veh = []


        # Index of the current dataset
        dataset_index = 0


        # For each dataset (each video file of each dataset)
        for directory in data_dirs:

            # Load the data from the txt file
            print("Now processing: ", directory)
            
            column_names = ['frame_id','agent_id','pos_x','pos_y', 'vel_x', 'vel_y', 'label', 'timestamp']

            df_orig = pd.read_csv(directory, dtype={'frame_id':'int','agent_id':'int', 'label':'str'}, usecols=column_names)
            df_orig = df_orig[column_names] # changing the order of the columns as specifed inn the "columns_names"

            # downsampling the data
            min_frame = df_orig["frame_id"].min()
            max_frame = df_orig["frame_id"].max()
            considered_frames = np.arange(min_frame, max_frame, self.downsample_step).tolist()
            df = df_orig.loc[df_orig['frame_id'].isin(considered_frames)]


            df_ped_orig = df.loc[(df['label'].isin(['pedestrian', 'ped']))] # !!!! Check these lables for each new dataset you want to add
            df_veh_orig = df.loc[(df['label'].isin(['car', 'cart', 'veh']))]  # !!!! Check these lables for each new dataset you want to add
            self.target_ids = np.array(df_ped_orig.drop_duplicates(subset={'agent_id'}, keep='first', inplace=False)['agent_id'])
            self.target_veh_ids = np.array(df_veh_orig.drop_duplicates(subset={'agent_id'}, keep='first', inplace=False)['agent_id'])


            # adding acceleration and other feature as new columns to the dataframe calculated
            # through the changes of velocity and other related formulas
            df_ped = self.add_features(df_ped_orig, self.timestamp)
            df_veh = self.add_features(df_veh_orig, self.timestamp)
            # print(df_ped.head())

            # convert pandas -> numpy array
            data = np.array(df_ped.drop(['label'], axis=1))
            data_veh = np.array(df_veh.drop(['label'], axis=1))

            # keep original copy of file
            orig_data.append(data)
            orig_data_veh.append(data_veh)

            data = np.swapaxes(data,0,1) # chaning the column and row positions of a array.
            # So now the frame id will propogate thorugh the columns and the frames are all in the first row
            data_veh = np.swapaxes(data_veh,0,1)
            
            # get frame numbers
            frameList_rep = data[0, :].tolist() 
            frameList = list(dict.fromkeys(frameList_rep)) # removing the repeated frames and sorting them at the same time
            frameList.sort() # sorting the list

            # Add the list of frameIDs to the frameList_data
            frameList_data.append(frameList) 

            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            valid_numPeds_data.append([]) 
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])

            # list of peds for each frame
            pedsList_data.append([])
            valid_pedsList_data.append([])

            target_ids.append(self.target_ids)

            # Doing the same for veh data
            numVehs_data.append([])
            all_frame_data_veh.append([])
            vehsList_data.append([])
            target_ids_veh.append(self.target_veh_ids)

            for ind, frame in enumerate(frameList):

                # Extract all pedestrians in current frame
                pedsInFrame = data[: , data[0, :] == frame]
                vehsInFrame = data_veh[: , data_veh[0, :] == frame]
            
                # Extract peds list
                pedsList = pedsInFrame[1, :].tolist() # Grabbing the agent_id of all the pedestrians in this specific frame
                vehsList = vehsInFrame[1, :].tolist()

                # Initialize the row of the numpy array
                pedsWithPos = []
                vehsWithPos = []

                # For each ped in the current frame
                for ped in pedsList:
                    # Extract their x and y positions
                    current_x = pedsInFrame[2, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_vx = pedsInFrame[4, pedsInFrame[1, :] == ped][0]
                    current_vy = pedsInFrame[5, pedsInFrame[1, :] == ped][0]
                    timestamp = pedsInFrame[6, pedsInFrame[1, :] == ped][0]
                    current_ax = pedsInFrame[7, pedsInFrame[1, :] == ped][0]
                    current_ay = pedsInFrame[8, pedsInFrame[1, :] == ped][0]
                    last_speed_change = pedsInFrame[9, pedsInFrame[1, :] == ped][0]
                    last_deviation_angle = pedsInFrame[10, pedsInFrame[1, :] == ped][0]

                    # Add their pedID, x, y to the row of the numpy array
                    pedsWithPos.append([ped, current_x, current_y, current_vx, current_vy, timestamp, current_ax, current_ay,
                                         last_speed_change, last_deviation_angle])

                # For each veh in the current frame
                for veh in vehsList:
                    # Extract their x and y positions
                    current_x_veh = vehsInFrame[2, vehsInFrame[1, :] == veh][0]
                    current_y_veh = vehsInFrame[3, vehsInFrame[1, :] == veh][0]
                    current_vx_veh = vehsInFrame[4, vehsInFrame[1, :] == veh][0]
                    current_vy_veh = vehsInFrame[5, vehsInFrame[1, :] == veh][0]
                    timestamp_veh = vehsInFrame[6, vehsInFrame[1, :] == veh][0]
                    current_ax_veh = vehsInFrame[7, vehsInFrame[1, :] == veh][0]
                    current_ay_veh = vehsInFrame[8, vehsInFrame[1, :] == veh][0]
                    last_speed_change_veh = vehsInFrame[9, vehsInFrame[1, :] == veh][0]
                    last_deviation_angle_veh = vehsInFrame[10, vehsInFrame[1, :] == veh][0]


                    # Add their pedID, x, y to the row of the numpy array
                    vehsWithPos.append([veh, current_x_veh, current_y_veh, current_vx_veh, current_vy_veh, timestamp_veh,
                                         current_ax_veh, current_ay_veh, last_speed_change_veh, last_deviation_angle_veh])


                # Add the details of all the peds in the current frame to all_frame_data 
                all_frame_data[dataset_index].append(np.array(pedsWithPos))
                pedsList_data[dataset_index].append(pedsList)
                numPeds_data[dataset_index].append(len(pedsList))

                all_frame_data_veh[dataset_index].append(np.array(vehsWithPos))
                vehsList_data[dataset_index].append(vehsList)
                numVehs_data[dataset_index].append(len(vehsList))

            dataset_index += 1
  
        # Save the arrays in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_numPeds_data, valid_frame_data,
                      pedsList_data, valid_pedsList_data, target_ids, orig_data, all_frame_data_veh, 
                      numVehs_data, vehsList_data, target_ids_veh, orig_data_veh), f, protocol=2)
        f.close()


    def load_preprocessed(self, data_file, data_file_tr, data_file_te, infer = False): 
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file containig all the trajecotries
        data_file_tr: the path to the pickled data file that we want to store our training portion of data
        data_file_te: the path to the pickled data file that we want to store our testing portion of data
        '''
       
        print("Loading all the dataset: ", data_file)

        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0] # all_frame_data  from the frame_preprocess function
        self.frameList = self.raw_data[1] # frameList_data  from the frame_preprocess function
        self.numPedsList = self.raw_data[2] # numPeds_data  from the frame_preprocess function
        self.valid_numPedsList = self.raw_data[3] # valid_numPeds_data  from the frame_preprocess function
        self.valid_data = self.raw_data[4] # valid_frame_data  from the frame_preprocess function
        self.pedsList = self.raw_data[5] # pedsList_data  from the frame_preprocess function
        self.valid_pedsList = self.raw_data[6] # valid_pedsList_data  from the frame_preprocess function
        self.target_ids = self.raw_data[7] # target_ids  from the frame_preprocess function
        self.orig_data = self.raw_data[8] # orig_data  from the frame_preprocess function
        
        self.data_veh = self.raw_data[9]
        self.numVehsList = self.raw_data[10]
        self.vehsList = self.raw_data[11]
        self.target_ids_veh = self.raw_data[12]
        self.orig_data_veh = self.raw_data[13]

        counter = 0
        valid_counter = 0
        print('Sequence size(frame) ------>',self.seq_length)
        print('One batch size (frame)--->-', self.batch_size*self.seq_length)

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            dataset_name = self.base_data_dirs[dataset].split('/')[-1]
            all_frame_data_veh = self.data_veh[dataset]
            # calculate number of sequence 
            num_seq_in_dataset = int(len(all_frame_data)) - (self.seq_length) # We allow overlap between the sequences that we extract,
          
            print('Data from the whole dataset(name, # frame, #sequence)--> ', dataset_name, ':', len(all_frame_data),':', (num_seq_in_dataset))

            # Increment the counter with the number of sequences in the current dataset
            counter += num_seq_in_dataset

        # Calculate the number of batches
        num_batches = int(counter/self.batch_size)
        num_seqences = counter

        print('Total number of batches:', num_batches)
        print('Total number of sequences:', num_seqences)

    
        sequence_x = []
        sequence_numPedsList = []
        sequence_PedsList = []
        sequence_x_veh = []
        sequence_numVehsList = []
        sequence_VehsList = []

        print("Deviding the data into sequences and storing them")

        for i in range(num_seqences):

            x, _, _ , numPedsList, PedsList, x_veh, numVehsList, VehsList = self.next_batch()
            sequence_x.append(x[0])
            sequence_numPedsList.append(numPedsList[0])
            sequence_PedsList.append(PedsList[0])
            sequence_x_veh.append(x_veh[0])
            sequence_numVehsList.append(numVehsList[0])
            sequence_VehsList.append(VehsList[0])

        print("Finished deviding the data into sequences")

        print("Extracting a the first portion of these sequences for test")
        # For HBS we get the first 31% of the data for testing (first 10 minute) and the rest for training
        test_data_portion = 0.31
        num_test_seq = int(test_data_portion*num_seqences)
        print('Number of seqeunces for testing: ', num_test_seq)
        print("*************")
        indx_list_test = list(range(0,num_test_seq))
        indx_list_train = list(range(num_test_seq,num_seqences)) 
        
        
        # creating and storing the portion of the data that will be used for testing
        sequence_x_test = [sequence_x[i] for i in indx_list_test]
        sequence_numPedsList_test = [sequence_numPedsList[i] for i in indx_list_test]
        sequence_PedsList_test = [sequence_PedsList[i] for i in indx_list_test]
        sequence_x_veh_test = [sequence_x_veh[i] for i in indx_list_test]
        sequence_numVehsList_test = [sequence_numVehsList[i] for i in indx_list_test]
        sequence_VehsList_test = [sequence_VehsList[i] for i in indx_list_test]

        f = open(data_file_te, "wb")
        pickle.dump((sequence_x_test, sequence_numPedsList_test, sequence_PedsList_test,
                      sequence_x_veh_test, sequence_numVehsList_test, sequence_VehsList_test),
                       f, protocol=2)
        f.close()

        # creating and storing the portion of the data that will be used for training
        sequence_x_train = [sequence_x[i] for i in indx_list_train]
        sequence_numPedsList_train = [sequence_numPedsList[i] for i in indx_list_train]
        sequence_PedsList_train = [sequence_PedsList[i] for i in indx_list_train]
        sequence_x_veh_train = [sequence_x_veh[i] for i in indx_list_train]
        sequence_numVehsList_train = [sequence_numVehsList[i] for i in indx_list_train]
        sequence_VehsList_train = [sequence_VehsList[i] for i in indx_list_train]

        f2 = open(data_file_tr, "wb")
        pickle.dump((sequence_x_train, sequence_numPedsList_train, sequence_PedsList_train, 
                     sequence_x_veh_train, sequence_numVehsList_train, sequence_VehsList_train), 
                     f2, protocol=2)
        f2.close()


    def load_preprocessed2(self, data_file, filtering=False): 

        print("Loading the data from: ", data_file)

        f = open(data_file, 'rb')
        self.seqdata = pickle.load(f)
        f.close()

        self.sequence_x = self.seqdata[0] 
        self.sequence_numPedsList = self.seqdata[1]
        self.sequence_PedsList = self.seqdata[2]
        self.sequence_x_veh = self.seqdata[3]
        self.sequence_numVehsList = self.seqdata[4]
        self.sequence_VehsList = self.seqdata[5]

        if filtering :
            print("________ Filtering process started _______")
            self.filtering()
      
        self.num_batches = int(len(self.sequence_x)/self.batch_size)
        self.num_seqences = len(self.sequence_x)

        self.num_features = self.sequence_x[0][0].shape[1] - 1 #  -1 because to not consider agent id in the count 

        print('Total number of batches for this pahse:', self.num_batches)
        print('Total number of sequences for this phase:', self.num_seqences)

     
    def batch_creater(self, pre_calculated_grids, method, suffle=True):

        '''
        creating a list of all batches for the whole training.
        x is a list of batches for ped position data. Each batch itself is a list of arrays.
        This list has a lenght equal to seq_lenght each array being the data in a frame
        This strucutre is repeated for all other data as well
        The whole list of seqeucnes at the begining are shuffled if shuffle is ON to have
        different data in each batch at every episoid when this is function is called
        '''
        x = []
        numPedsList = []
        PedsList = []
        x_veh = []
        numVehsList = []
        VehsList = []

        Grids = []
        Grids_veh = []
        Grids_TTC = []
        Grids_TTC_veh = []
        
        # shuffle multiple lists with same order
        if suffle:
            if (pre_calculated_grids == True):
                if (method == 4):
                    all_data = list(zip(self.sequence_x, self.sequence_numPedsList, self.sequence_PedsList, 
                                        self.sequence_x_veh, self.sequence_numVehsList, self.sequence_VehsList,
                                        self.grids, self.grids_veh, self.grids_TTC, self.grids_TTC_veh))         
                    random.shuffle(all_data)
                    sequence_x, sequence_numPedsList, sequence_PedsList, sequence_x_veh, sequence_numVehsList, \
                                sequence_VehsList, grids, grids_veh, grids_TTC, grids_TTC_veh= zip(*all_data)
                
                elif method == 1: # social LSTM. We do not have TTC grids for this method
                    all_data = list(zip(self.sequence_x, self.sequence_numPedsList, self.sequence_PedsList, 
                                        self.sequence_x_veh, self.sequence_numVehsList, self.sequence_VehsList, 
                                        self.grids, self.grids_veh))         
                    random.shuffle(all_data)
                    sequence_x, sequence_numPedsList, sequence_PedsList, sequence_x_veh, sequence_numVehsList, \
                                sequence_VehsList, grids, grids_veh = zip(*all_data)

            else:
                all_data = list(zip(self.sequence_x, self.sequence_numPedsList, self.sequence_PedsList,
                                     self.sequence_x_veh, self.sequence_numVehsList, self.sequence_VehsList))
                random.shuffle(all_data)
                sequence_x, sequence_numPedsList, sequence_PedsList, sequence_x_veh, sequence_numVehsList,  \
                            sequence_VehsList = zip(*all_data) # the outputs are tuple here. So need to be covnerted back to list
            
            sequence_x, sequence_numPedsList, sequence_PedsList,  = list(sequence_x), list(sequence_numPedsList), list(sequence_PedsList)
            sequence_x_veh, sequence_numVehsList, sequence_VehsList = list(sequence_x_veh), list(sequence_numVehsList), list(sequence_VehsList)
            if (pre_calculated_grids == True):
                grids, grids_veh = list(grids), list(grids_veh)
                if (method == 4): # Collision Grid
                    grids_TTC, grids_TTC_veh = list(grids_TTC), list(grids_TTC_veh)
       
        else:
            sequence_x =  self.sequence_x.copy()
            sequence_numPedsList = self.sequence_numPedsList.copy()
            sequence_PedsList = self.sequence_PedsList.copy()
            sequence_x_veh = self.sequence_x_veh.copy()
            sequence_numVehsList =  self.sequence_numVehsList.copy()
            sequence_VehsList = self.sequence_VehsList.copy()
            if (pre_calculated_grids == True):
                grids = self.grids.copy()
                grids_veh = self.grids_veh.copy()
                if (method == 4):
                    grids_TTC = self.grids_TTC.copy()
                    grids_TTC_veh = self.grids_TTC_veh.copy()
     
        seq = 0
        while (seq+self.batch_size)<= self.num_seqences: # leaving away the last sequences that cannot create a whole batch

                x_batch = []
                numPedsList_batch = []
                PedsList_batch = []
                x_batch_veh = []
                numVehsList_batch = []
                VehsList_batch = []
                if (pre_calculated_grids == True):
                    grids_batch = []
                    grids_veh_batch = []
                    grids_TTC_batch = []
                    grids_TTC_veh_batch = []

                for i in range(self.batch_size):
                    x_batch.append(sequence_x[seq])
                    numPedsList_batch.append(sequence_numPedsList[seq])
                    PedsList_batch.append(sequence_PedsList[seq])
                    x_batch_veh.append(sequence_x_veh[seq])
                    numVehsList_batch.append(sequence_numVehsList[seq])
                    VehsList_batch.append(sequence_VehsList[seq])
                    if (pre_calculated_grids == True):
                        grids_batch.append(grids[seq])
                        grids_veh_batch.append(grids_veh[seq])
                        if (method == 4):
                            grids_TTC_batch.append(grids_TTC[seq])
                            grids_TTC_veh_batch.append(grids_TTC_veh[seq])
                    seq = seq+1

                x.append(x_batch)
                numPedsList.append(numPedsList_batch)
                PedsList.append(PedsList_batch)
                x_veh.append(x_batch_veh)
                numVehsList.append(numVehsList_batch)
                VehsList.append(VehsList_batch)
                if (pre_calculated_grids == True):
                    Grids.append(grids_batch)
                    Grids_veh.append(grids_veh_batch)
                    if (method == 4):
                        Grids_TTC.append(grids_TTC_batch)
                        Grids_TTC_veh.append(grids_TTC_veh_batch)

        return x, numPedsList, PedsList, x_veh, numVehsList, VehsList, Grids, Grids_veh, Grids_TTC, Grids_TTC_veh


    def next_batch(self):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []

        # pedlist per sequence
        numPedsList_batch = []

        # pedlist per sequence
        PedsList_batch = []

        #return target_id
        target_ids = []

        # repeating the same for the vehicles
        x_batch_veh = []
        numVehsList_batch = []
        VehsList_batch = []
        target_ids_veh = []

        # Iteration index
        i = 0
        while i < 1: # self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.data[self.dataset_pointer] 
            numPedsList = self.numPedsList[self.dataset_pointer]
            pedsList = self.pedsList[self.dataset_pointer]
            
            frame_data_veh = self.data_veh[self.dataset_pointer] 
            numVehsList = self.numVehsList[self.dataset_pointer]
            vehsList = self.vehsList[self.dataset_pointer]
            
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            # if idx + self.seq_length-1 < len(frame_data):
            if idx + self.seq_length-1 < len(frame_data): 
                # All the data in this sequence
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_numPedsList = numPedsList[idx:idx+self.seq_length]
                seq_PedsList = pedsList[idx:idx+self.seq_length]

                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length] 

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data) 
                numPedsList_batch.append(seq_numPedsList)
                PedsList_batch.append(seq_PedsList)
             
                seq_source_frame_data_veh = frame_data_veh[idx:idx+self.seq_length]
                seq_numVehsList = numVehsList[idx:idx+self.seq_length]
                seq_VehsList = vehsList[idx:idx+self.seq_length]

                x_batch_veh.append(seq_source_frame_data_veh)
                numVehsList_batch.append(seq_numVehsList)
                VehsList_batch.append(seq_VehsList)
                
                # self.frame_pointer += self.seq_length
                self.frame_pointer += 1

                d.append(self.dataset_pointer) 
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=False)
    
        return x_batch, y_batch, d, numPedsList_batch, PedsList_batch, x_batch_veh, numVehsList_batch, VehsList_batch


    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        
        if not valid:
            
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
            print("*******************")
            print("now processing: %s"% self.get_file_name())
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0
            print("*******************")
            print("now processing: %s"% self.get_file_name(pointer_type = 'valid'))

        
    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0
        

    def get_file_name(self, offset=0, pointer_type = 'train'):
        #return file name of processing or pointing by dataset pointer
        #if pointer_type is 'train':
        if (pointer_type =='train'):
            return self.base_data_dirs[self.dataset_pointer+offset].split('/')[-1]
        
 
    def convert_proper_array(self, x_seq, num_pedlist, pedlist, veh_flag = False):
        #converter function to appropriate format. Instead of direcly use ped ids, we are mapping ped ids to
        #array indices using a lookup table for each sequence -> speed
        #output: seq_lenght (real sequence lenght+1)*max_ped_id+1 (biggest id number in the sequence)*2 (x,y)
        
        #get unique ids from sequence
        unique_ids = pd.unique(np.concatenate(pedlist).ravel().tolist()).astype(int)
        # create a lookup table which maps ped ids -> array indices
        lookup_table = dict(zip(unique_ids, range(0, len(unique_ids))))

        seq_data = np.zeros(shape=(self.seq_length, len(lookup_table), self.num_features)) 


        # create new structure of array
        for ind, frame in enumerate(x_seq):
            if num_pedlist[ind] != 0:
                corr_index = [lookup_table[x] for x in frame[:, 0]]
                seq_data[ind, corr_index,:] = frame[:,1:] # adding vx and vy, timestep, ax and ay

        return_arr = Variable(torch.from_numpy(np.array(seq_data)).float())

        return return_arr, lookup_table


    def get_len_of_dataset(self):
        # return the number of dataset in the mode
        return len(self.data)

    def get_frame_sequence(self, frame_lenght):
        #begin and end of predicted fram numbers in this seq.
        begin_fr = (self.frame_pointer - frame_lenght)
        end_fr = (self.frame_pointer)
        frame_number = self.orig_data[self.dataset_pointer][begin_fr:end_fr, 0].transpose()
        return frame_number

   
    def write_to_plot_file(self, data, path):
        # write plot file for further visualization in pkl format
        self.reset_batch_pointer()
        file_name = "test_results.pkl"
        print("Writing to plot file  path: %s, file_name: %s"%(path, file_name))
        with open(os.path.join(path, file_name), 'wb') as f:
                pickle.dump(data, f)


    def add_features(self, df, timestamp):
        '''
        df: is a data frame to which we want to add a new column
        timestamp: the time difference between two consecutive frames/rows in the dataset that will be used for calculation of derivatives
        '''
        df = df.sort_values(by=['agent_id', 'frame_id'])
        df['a_x'] = df.groupby(['agent_id'])['vel_x'].diff().fillna(0) /timestamp
        df['a_y'] = df.groupby(['agent_id'])['vel_y'].diff().fillna(0) /timestamp

        # Generating some action features to be used
        df['speed'] = (df['vel_x']**2 + df['vel_y']**2)**0.5
        df['speed_diff'] = df.groupby(['agent_id'])['speed'].diff().fillna(0)


        df['prev_vel_x'] = df.groupby(['agent_id'])['vel_x'].shift()
        df.prev_vel_x.fillna(df.vel_x, inplace=True)
        df['prev_vel_y'] = df.groupby(['agent_id'])['vel_y'].shift()
        df.prev_vel_y.fillna(df.vel_y, inplace=True)
    
        df['dot_vel'] = df['vel_x']*df['prev_vel_x'] + df['vel_y']*df['prev_vel_y']
        df['det_vel'] = df['prev_vel_x']*df['vel_y'] - df['prev_vel_y']*df['vel_x']
        df['deviation_angle'] = np.arctan2(df['det_vel'], df['dot_vel']) * 180/np.pi

        del df['speed']
        del df['prev_vel_x']
        del df['prev_vel_y']
        del df['dot_vel']
        del df['det_vel']
       
        return df


    def grid_creation(self, args):

        grids = []
        grids_veh = []
        grids_TTC = []
        grids_TTC_veh = []

        for seq in range(len(self.sequence_x)):

            x_seq , numPedsList_seq, PedsList_seq = self.sequence_x[seq], self.sequence_numPedsList[seq], self.sequence_PedsList[seq]
            x_seq_veh , numVehsList_seq, VehsList_seq = self.sequence_x_veh[seq], self.sequence_numVehsList[seq], self.sequence_VehsList[seq]

            #dense vector creation
            x_seq, lookup_seq = self.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq) 
            # order of featurs in x_seq: x, y, vx, vy, timestamp, ax, ay 
            x_seq_veh, lookup_seq_veh = self.convert_proper_array(x_seq_veh, numVehsList_seq, VehsList_seq, veh_flag=True)

            if args.method == 1: # Social LSTM
                grid_seq = getSequenceGridMask(x_seq, PedsList_seq, args.neighborhood_size, args.grid_size, False, lookup_seq) 
                grid_seq_veh_in_ped = getSequenceGridMask_heterogeneous(x_seq, PedsList_seq, x_seq_veh, VehsList_seq,
                                                                         args.neighborhood_size_veh_in_ped,
                                                                         args.grid_size_veh_in_ped, False, lookup_seq,
                                                                        lookup_seq_veh, False)
                
            elif args.method ==4: # CollisionGird
                grid_seq, grid_TTC_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq, PedsList_seq, args.TTC,
                                                                         args.D_min, args.num_sector, False,
                                                                         lookup_seq, lookup_seq) 
                                                                        # Mahsa: I will not use cuda to not get out of memory.
                                                                        #  I will move to cude within the model itself
                grid_seq_veh_in_ped, grid_TTC_veh_seq = getSequenceInteractionGridMask(x_seq, PedsList_seq, x_seq_veh, VehsList_seq,
                                                                                        args.TTC_veh, args.D_min_veh, 
                                                                                        args.num_sector, False, lookup_seq, lookup_seq_veh, 
                                                                                        is_heterogeneous=True, is_occupancy=False) 

            grids.append(grid_seq)
            grids_veh.append(grid_seq_veh_in_ped)
            if args.method ==4: # CollisionGird:
                grids_TTC.append(grid_TTC_seq)
                grids_TTC_veh.append(grid_TTC_veh_seq)


        self.grids = grids
        self.grids_veh = grids_veh
        self.grids_TTC = grids_TTC
        self.grids_TTC_veh = grids_TTC_veh


    def filtering(self):
        '''
        This function filters the sequence data to include only 
        pedestrian and vehicles that are present throughout the 
        whole sequence. This means focusing only on the common 
        ped ids in the pedList throughout the whole seqences 
        '''

        filt_sequence_x = []
        filt_sequence_numPedsList = []
        filt_sequence_PedsList = [] 
        filt_sequence_x_veh = [] 
        filt_sequence_numVehsList = [] 
        filt_sequence_VehsList = [] 

        count = 0
        for i in range(len(self.sequence_x)):
            PedList_seq = self.sequence_PedsList[i]
            common_ids = set(PedList_seq[0])
            for pedlist in PedList_seq[1:]:
                common_ids.intersection_update(pedlist)
            common_ids = list(common_ids)

            if (len(common_ids) == 0): # no common ped exsits in this sequence, so we remove this sequence compeletly
                count += 1
                continue

            filt_seq_x = []
            filt_seq_numPedsList = []
            filt_seq_PedsList = []

            for frame in self.sequence_x[i]:
                filterd_frame = frame[np.isin(frame[:,0], common_ids)]
                filt_seq_x.append(filterd_frame)
                filt_seq_numPedsList.append(len(common_ids))
                filt_seq_PedsList.append(common_ids)
            
            filt_sequence_x.append(filt_seq_x)
            filt_sequence_numPedsList.append(filt_seq_numPedsList)
            filt_sequence_PedsList.append(filt_seq_PedsList)
            filt_sequence_x_veh.append(self.sequence_x_veh[i])
            filt_sequence_numVehsList.append(self.sequence_numVehsList[i])
            filt_sequence_VehsList.append(self.sequence_VehsList[i])

        self.sequence_x = filt_sequence_x[:]
        self.sequence_numPedsList = filt_sequence_numPedsList[:]
        self.sequence_PedsList = filt_sequence_PedsList[:]
        self.sequence_x_veh = filt_sequence_x_veh[:]
        self.sequence_numVehsList = filt_sequence_numVehsList[:]
        self.sequence_VehsList = filt_sequence_VehsList[:]
        print(count, " number of seqeunces were deleted because of not having a single pedestrian present throughout the whole sequence")
        print("________ Filtering process finished ______")