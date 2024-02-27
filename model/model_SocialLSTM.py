import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

class SocialModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 2 
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size 
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.gru = args.gru

        self.grid_size_veh_in_ped = args.grid_size_veh_in_ped 

        # The LSTM cell for pedestrians
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        if self.gru:
            self.cell = nn.GRUCell(2*self.embedding_size, self.rnn_size) # Mahsa: same here !!!


        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.rnn_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    

    def getSocialTensor(self, grid, hidden_states, veh_tensor = False):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds / OR the embedded state of vehicles
        '''
        # Number of peds
        numNodes = grid.size()[0]

        # Construct the variable
        if (veh_tensor == True): 
            social_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size, self.embedding_size)) 
        else: # for ped in ped grid.
            social_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size, self.rnn_size)) 
    
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        if (veh_tensor == True): 
            social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.embedding_size) 
        else:
            social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size) 
        return social_tensor
            
   
    def forward(self, *args):

        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
    
        
        # Construct the output variable
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]

        if self.gru:
            cell_states = None

        PedsList = args[4]
        num_pedlist = args[5]
        dataloader = args[6]
        look_up = args[7]

        input_data_veh = args[8]
        grids_veh = args[9]
        VehsList = args[10]
        look_up_veh = args[11]

        numNodes = len(look_up)
        outputs = Variable(torch.zeros((self.seq_length-1) * numNodes, self.output_size))
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum,frame in enumerate(input_data):

            # Peds present in the current frame

            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]
            # nodeIDs_veh = [int(nodeID_veh) for nodeID_veh in VehsList[framenum]]

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue


            # List of nodes
            list_of_nodes = [look_up[x] for x in nodeIDs]
            # list_of_nodes_veh = [look_up_veh[x_veh] for x_veh in nodeIDs_veh]

            corr_index = Variable((torch.LongTensor(list_of_nodes)))
            if self.use_cuda:            
                corr_index = corr_index.cuda()

            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes,:2] # Getting only the x and y of each pedestrian for the input. Leaving th vx and vy
            # nodes_current_veh = input_data_veh[framenum,list_of_nodes_veh,5:7]
           
            # Get the corresponding grid masks
            grid_current = grids[framenum] 
            # grid_current_veh = grids_veh[framenum]             


            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)


            if not self.gru:
                cell_states_current = torch.index_select(cell_states, 0, corr_index)

            # Compute the social tensor
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current.cpu()) 
            if self.use_cuda:
                social_tensor = social_tensor.cuda()

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

    
            # Concat input
            # Uncomment this later when adding the vehicle !!!!!
            # concat_embedded = torch.cat((input_embedded, tensor_embedded,tensor_embedded_veh), 1)  
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            if not self.gru:
                # One-step of the LSTM
                h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
            else:
                h_nodes = self.cell(concat_embedded, (hidden_states_current))


            # Compute the output
            outputs[framenum*numNodes + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros((self.seq_length-1), numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length-1):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states