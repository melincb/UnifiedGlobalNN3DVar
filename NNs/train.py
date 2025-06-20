# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script with methods for model training.

@author: Uroš Perkan
"""

#%% Libraries

# TorchMetrics https://torchmetrics.readthedocs.io/en/stable/pages/quickstart.html

import os
import copy
import time
#import psutil
import resource
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import data_preparation
import torch_scalers

# %% ===================================== 
#           CREATE CUSTOM DATASET
#    =====================================

class S2SDataset(Dataset):
    """
    Fields and their annotations_file.txt are stored in
    .../Data dir.
    
    Creates a pytorch dataset, which retrieves our datasets
    features and labels one sample at a time. 
    We can index Dataset manually like training_data[idx].
    
    ----------------------------------------------------
    Args
    ----------------------------------------------------
    ROOT_ANNOTATIONS: (str)        - list of main paths to directory with annotations_file.
    ANNOTATIONS_NAME: (str)        - list of names of annotations file (same order as ROOT_ANNOTATIONS)
    fields_indexes: (dict)         - dictionary with keys "input" and "target":
                                     input --> array of integers i which point to
                                     all input data files we wish to have in our 
                                     dataset (diferent for training, testing,
                                     predicting...) e.g. [0,1,...,N-2]
                                     output --> same but for target data files 
                                     e.g. [1,2,...,N-1] (t+dt)
    SCALERS:                       - list of torch scalers for scaling data - same order as ANNOTATIONS_NAME
    transform_name:                - Name of transformation:
                                        --> { "ToTensor", "ToScaledTensor",
                                              "FieldPad" }
    target_transform_name:         - Same for target transformation:
    transform: (callable)          - (ptional) transform to be applied on a
                                     input field. 
    target_transform: (callable)   - (optional) transform to be applied on a
                                     target (output) field. 
    EastWest_pad (optional) :      - number of east-west padding to apply to
                                     field (default = 0)                          
    NorthSouth_pad (optional) :    - number of north-south padding to apply
                                     to field (default = 0)  
    input_sequence_length :        - length of time series to be averaged as an input
                                     the model (t0, t-1, t-2,...). Has to be at
                                     least 1 (t0)
    TARGET_VARIABLES:              - list of True, False - True if variable should be contained in
                                     target fields                   
    """

    # Overwrite Datasets __init__, __len__ and __getitem__ methods
    def __init__(self, root_annotations, ANNOTATIONS_NAME, fields_indexes,
                 SCALERS = None, transform_name="ToTensor", 
                 target_transform_name="ToTensor",
                 transform=None, target_transform=None,
                 EastWest_pad=0,  NorthSouth_pad=0,
                 input_sequence_length = 1, 
                 averaging_sequence_length = 1,
                 TARGET_VARIABLES=None): 

        self.root = root_annotations
        self.NAME = ANNOTATIONS_NAME
        self.input_fields_indexes = fields_indexes['input']
        self.target_fields_indexes = fields_indexes['output']
        self.SCALERS = SCALERS
        assert len(self.SCALERS) == len(self.NAME), "SCALERS lenght is not the same as ANNOTATIONS_NAME"
        self.transform_name = transform_name
        self.target_transform_name = target_transform_name
        self.transform = transform
        self.target_transform = target_transform
        self.EastWest_pad = EastWest_pad        
        self.NorthSouth_pad = NorthSouth_pad
        assert input_sequence_length > 0, "Length of time series input is zero"        
        self.input_sequence_length = input_sequence_length
        self.averaging_sequence_length = averaging_sequence_length
        self.TARGET_VARIABLES = TARGET_VARIABLES

    def __len__(self):
        """Num of samples in dataset."""
        return len(self.input_fields_indexes)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index 
        fields_indexes[idxc].
        Based on the index it identifies the fields locations on disk (both
        input and target field). Then it calls transform and target_transform
        functions (if applicable) and transforms fields. It returns tensor field
        and coresponding target tensor field.
        """

        # Load fields annotations (path to dataset fields)
        ANNOTATION_FILES = []
        for i in range(len(self.NAME)):
            annotation_file = np.loadtxt(
                os.path.join(self.root, self.NAME[i]),
                dtype=np.str_)
            ANNOTATION_FILES.append(annotation_file)

        # Input and target fields serial number
        input_fields_index = self.input_fields_indexes[idx] 
        target_fields_index = self.target_fields_indexes[idx] 
        
        # # Input fields
        # INPUT_FIELDS = []
        # k = 0
        # for i in range(self.input_sequence_length):
        #     intermediate_fields = []
        #     for j in range(self.averaging_sequence_length):
        #         # path = annotation_file_geopotentital[index-k-1]
        #         # field = np.load(path)
        #         path = annotation_file_geopotencital[input_fields_index-k]
        #         field = np.load(path)
        #         intermediate_fields.append(field)
        #         k += 1
        #     INPUT_FIELDS.append(np.average(np.array(intermediate_fields), axis=0))
        # INPUT_FIELDS = list(np.flip(np.array(INPUT_FIELDS), axis=0))

        # Input fields
        INPUT_FIELDS = [] # [var1[t], var1[t-1],...,var1[t-isl+1],
                          #  var2[t], var2[t-1],...,var2[t-isl+1],
                          #                   ...
                          #  vark[t], vark[t-1],...,vark[t-isl+1],
                          # ]
        INPUT_SCALERS = []
        for a in range(len(ANNOTATION_FILES)):
            k = 0
            for i in range(self.input_sequence_length):
                intermediate_fields = []
                for j in range(self.averaging_sequence_length):
                    path = ANNOTATION_FILES[a][input_fields_index-k]
                    field = np.load(path)
                    intermediate_fields.append(field)
                    k += 1
                INPUT_FIELDS.append(np.average(np.array(intermediate_fields), axis=0))
                INPUT_SCALERS.append(self.SCALERS[a])

            # INPUT_FIELDS = list(np.flip(np.array(INPUT_FIELDS), axis=0))

        # Last sequence_length indexes
        #input_start_idx = input_fields_index - self.input_sequence_length + 1        
        #input_end_idx = input_fields_index + 1        
        target_start_idx = target_fields_index
        target_end_idx = target_fields_index + self.averaging_sequence_length

        # Target fields (properly averaged)
        TARGET_FIELDS = [] # [var1[t+1], var2[t+1], ..., var_ntar[t+1]]
        TARGET_SCALERS = []
        for a in range(len(ANNOTATION_FILES)):
            if self.TARGET_VARIABLES[a]:
                intermediate_fields = []
                for i in range(target_start_idx, target_end_idx):
                    intermediate_fields.append(np.load(ANNOTATION_FILES[a][i]))
                TARGET_FIELDS.append(np.average(np.array(intermediate_fields), axis=0))
                TARGET_SCALERS.append(self.SCALERS[a])

        # input_field = np.average(np.array(INPUT_FIELDS), axis=0)
        # target_field = np.average(np.array(TARGET_FIELDS), axis=0)

        # # Scalers
        # SCALERS = [] # The same shape as INPUT_FIELDS
        # for a in range(len(ANNOTATION_FILES)):
        #     for i in range(self.input_sequence_length):
        #         SCALERS.append(self.SCALERS[a])    

        # Apply transformations
        if self.transform:
            if self.transform_name == "ToScaledTensor":
                for i in range(len(INPUT_FIELDS)):    
                    input_field = self.transform(INPUT_FIELDS[i], INPUT_SCALERS[i])
                    INPUT_FIELDS[i] = input_field
            elif self.transform_name == "ToTensor":
                for i in range(len(INPUT_FIELDS)):    
                    input_field = self.transform(INPUT_FIELDS[i])
                    INPUT_FIELDS[i] = input_field

        if self.target_transform:
            if self.target_transform_name == "ToScaledTensor":
                for i in range(len(TARGET_FIELDS)):
                    target_field = self.target_transform(TARGET_FIELDS[i], TARGET_SCALERS[i])
                    TARGET_FIELDS[i] = target_field
            elif self.target_transform_name == "ToTensor":
                for i in range(len(TARGET_FIELDS)):
                    target_field = self.target_transform(TARGET_FIELDS[i])
                    TARGET_FIELDS[i] = target_field

        input_field = torch.cat(INPUT_FIELDS, 0)
        target_field = torch.cat(TARGET_FIELDS, 0)

        return {'input': input_field, 'target': target_field}

class ToTensor(object):
    """Convert ndarrays to torch.tensors."""

    def __call__(self, ndarray):
        # Check if ndarray is of shape (F,H,W)
        if len(ndarray.shape) == 3:
            return torch.from_numpy(ndarray)
        else:
            H = ndarray.shape[0]
            W = ndarray.shape[1]
            ndarray = ndarray.reshape(1,H,W)
        return torch.from_numpy(ndarray)

class DecreaseDensity(object):
    """
    Reduces resolution by keeping every density_reduction_number.
    """

    def __call__(self, two_dim_array, 
                 density_reduction_number=4, transform=ToTensor()):
        arr = two_dim_array
        n = density_reduction_number
        decreased_arr = arr[::n, ::n]
        decreased_tensor = transform(decreased_arr)
        return decreased_tensor

class ToScaledTensor(object):
    """
    Convert ndarrays to scaled torch.tensors.
    """

    def __call__(self, ndarray, scaler):
        """Args: scaler instance"""
        # Check if ndarray is of shape (F,H,W)
        if len(ndarray.shape) == 3:
            tensor = torch.from_numpy(ndarray)
            transformed_tensor = scaler.transform(tensor)
            return transformed_tensor
        else:
            H = ndarray.shape[0]
            W = ndarray.shape[1]
            ndarray = ndarray.reshape(1,H,W)
            tensor = torch.from_numpy(ndarray)
            transformed_tensor = scaler.transform(tensor)            
        return torch.from_numpy(ndarray)

def Pad(twodarray, EastWest_pad, NorthSouth_pad):
    
    # NORTH AND SOUTH PADDING
    if NorthSouth_pad > 0:
        # Top of matrix transformation
        top = np.flip(twodarray[0:NorthSouth_pad,:], axis=0)
        top = np.roll(top, shift=int(top.shape[1]/2), axis=1)
        # Bottom of matrix transformation
        bottom = np.flip(twodarray[-NorthSouth_pad:,:], axis=0)
        bottom = np.roll(bottom, shift=int(bottom.shape[1]/2), axis=1)
        # Stack together
        arr = np.vstack((top, twodarray))
        arr = np.vstack((arr, bottom))
    else:
        arr = copy.deepcopy(twodarray)

    # EAST AND WEST PADDING
    if EastWest_pad == 0:
        return arr
    else:
        # Pad all 4 sides
        arr = np.pad(arr, (EastWest_pad, EastWest_pad), 'wrap')
        # Top and bottom
        arr = arr[EastWest_pad:-EastWest_pad,:]
        return arr

if __name__ == "__main__":
    a = np.array([[i+j for j in range(10)] for i in range(5)])
    print(a)
    print(a.shape)
    pad = Pad(a, 1, 0)
    print(pad)
    print(pad.shape)

class FieldPad(object):
    """
    Reduces resolution by keeping every density_reduction_number.
    """

    def __call__(self, twodarray, scaler, EastWest_pad,
                 NorthSouth_pad, transform=ToScaledTensor()):
        
        arr = Pad(twodarray, EastWest_pad, NorthSouth_pad)
        return transform(arr, scaler)

# Check if dataset works
if __name__ == "__main__":
    
    ROOT_ANNOTATIONS = [ 
        "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density",
        "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density",
        "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density",
    ]
    ANNOTATIONS_NAME = [ 
        "geopotential_annotation_file.txt",
        "tau_300_700_annotation_file.txt",
        "surface_temperature_annotation_file.txt",
    ]
    
    # Create dataset instance
    train_data = S2SDataset(
        ROOT_ANNOTATIONS,
        ANNOTATIONS_NAME,
        {'input':[150,151,152,153], 'output':[151,152,153,154]},
        transform_name="ToScaledTensor", target_transform_name="ToScaledTensor",
        transform=ToScaledTensor(), target_transform=ToScaledTensor(),
        input_sequence_length=30, averaging_sequence_length=3
    )

    # Open one sample (returns dict {'input', 'output'})
    print("input: ", type(train_data[0]['input']), "; shape = ", train_data[0]['input'].size())
    print("target: ", type(train_data[0]['target']), "; shape = ", train_data[0]['target'].size())

    # Use of DataLoader
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    batch = next(iter(train_dataloader))
    input_batch, target_batch = batch['input'], batch['target'] 
    
    # Shape: (N,F,H,W)
    print(f"Input tensor shape {input_batch.size()}")
    print(f"Target tensor shape {target_batch.size()}")

    import matplotlib.pyplot as plt

    # Note: H = 0 is South Pole, H = 180 is North Pole
    # Maybe rewrite data at halves of latitudes to exclude poles
    plt.title("Geopotencial")
    plt.imshow(input_batch[0,0,:,:])
    plt.xlabel("degrees east")
    plt.ylabel("degrees north of SP")
    plt.show()


    # Compose transforms
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # composed = transforms.Compose([Rescale(256),
    #                            RandomCrop(224)])

# %% ====================================================== 
#                        TRAINING DATA
#    ======================================================

class Training():
    """
    Class that stores method train.   
    ----------------------------------------------------
    Args
    ----------------------------------------------------
    root_annotations: (str) - main path to directory with annotation_file.
    ANNOTATIONS_NAME: (str)  - list of names of annotation files
    delta_t [days] (float): - delta_t e.g 0.5 for delta_t = 12 hr   
    ----------------------------------------------------
    Methods
    ----------------------------------------------------
    ---> train_test_indexes
    ---> train_test_datasets
    ---> fit_and_save_sklearn_scaler
    ---> load_sklearn_scaler
    ---> load_model
    ---> train_loop
    ---> test_loop
    ---> train
    """

    def __init__(self, root_annotations, ANNOTATIONS_NAME):
        self.root = root_annotations
        self.NAME = ANNOTATIONS_NAME

    def __repr__(self):
        print(f"Training({self.root}, {self.NAME})")

    def train_test_indexes(self, start_date, end_date):
        """
        Args: choose time interval for training dataset
        start_date: e.g. (1, 1, 1979)
        end_date: e.g. (31, 12, 2020)
        Returns: indexes = (train_input_indexes, test_input_indexes, 
                            train_target_indexes, test_target_indexes)
        """
        date_to_day = data_preparation.DateToConsecutiveDay()

        # --> Get start and end indexes for annotation_file 
        #     (asumming dataset starts at 1950)
        start_index = date_to_day.num_of_samples(
            start_date[2], start_date[1], start_date[0], 1)

        end_index = date_to_day.num_of_samples(
            end_date[2], end_date[1], end_date[0], 1)
        
        # --> Create lists containing train and test input and target indices. 
        input_fields_indexes = [i for i in range(start_index, end_index)]
        target_fields_indexes = [i for i in range(start_index+1, end_index+1)]
        
        # Split in train and test indices and shuffle them
        import sklearn.model_selection
        train_test_split_indexes = sklearn.model_selection.train_test_split(
            input_fields_indexes, target_fields_indexes, test_size=0.2, 
            random_state=42, shuffle=True
        )
           
        train_input_indexes = train_test_split_indexes[0]
        test_input_indexes = train_test_split_indexes[1]
        train_target_indexes = train_test_split_indexes[2]
        test_target_indexes = train_test_split_indexes[3]

        indexes = (train_input_indexes, test_input_indexes, 
                   train_target_indexes, test_target_indexes)

        return indexes

    def train_test_datasets(self, SCALERS, transform_name="ToScaledTensor", 
            target_transform_name="ToScaledTensor", transform=ToScaledTensor(), 
            target_transform=ToScaledTensor(), EastWest_pad=0, NorthSouth_pad=0,
            start_date=(1,1,1950), end_date=(31,12,2020), 
            input_sequence_length=1, averaging_sequence_length=1,
            TARGET_VARIABLES=None): 
        """
        Returns train and test dataset instances.
        """

        indexes = self.train_test_indexes(start_date, end_date)        
        
        train_input_indexes = indexes[0]
        test_input_indexes = indexes[1]
        train_target_indexes = indexes[2]
        test_target_indexes = indexes[3]

        # --> Create train and test dataset instances
        train_dataset = S2SDataset(
            self.root, self.NAME, 
            {'input':train_input_indexes, 'output':train_target_indexes},
            SCALERS, transform_name, target_transform_name,
            transform, target_transform, EastWest_pad, NorthSouth_pad,
            input_sequence_length, averaging_sequence_length,
            TARGET_VARIABLES)

        test_dataset = S2SDataset(
            self.root, self.NAME,
            {'input':test_input_indexes, 'output':test_target_indexes},
            SCALERS, transform_name, target_transform_name,
            transform, target_transform, EastWest_pad, NorthSouth_pad,
            input_sequence_length, averaging_sequence_length,
            TARGET_VARIABLES)

        return train_dataset, test_dataset

    def fit_and_save_sklearn_scaler(self, 
            root_annotations, annotations_name,
            scaler, main_root_scaler, 
            scaler_name, EastWest_pad=0, NorthSouth_pad=0,
            start_date=(1,1,1950), end_date=(31,12,2020),
            averaging_sequence_length = 1):
        
        """
        Fit scaler on training dataset data
        and save it in root folder next to annotation_file.
        ----------------------------
        Args:
        ----------------------------
        scaler (str): {"StandardScaler", "MinMaxScaler", "StandardScaler_ReplaceZeroStd", "StandardScaler_GeneralStd"} (MinMax not yet implemented)
        scaler_name (str): name of scaler for saving / loading
        EastWest_pad (optional int): how much to pad east and west (default=0) 
        NorthSouth_pad (optional int): how much to pad north and south (default=0) 
        """

        indexes = self.train_test_indexes(start_date, end_date)        

        # Only training input indexes need to be scaled 
        # (in next day prediction I only scale input field).
        train_input_indexes = np.sort(indexes[0])
        
        # Eliminate negative indexes (because of averaging)
        stop = False
        i = 0
        while stop == False:
            if train_input_indexes[i] - averaging_sequence_length + 1 >= 0:
                train_input_indexes = train_input_indexes[i:]
                stop = True
            else:
                i += 1

        print(f"Train dataset consists of {len(train_input_indexes)} files")

        # Load fields annotations (path to dataset fields)
        annotation_file_geopotencital = np.loadtxt(
            os.path.join(root_annotations, annotations_name),
            dtype=np.str_)

        intermediate_list = []

        for i, idx in enumerate(train_input_indexes):
            if i % 500 == 0:
                if i != 0:
                    print(f"Appending {i} / {len(train_input_indexes)}")
                    intermediate_list.append(samples)
                    del samples
                
                # # Create ndarray which accumulates all samples
                # idx = train_input_indexes[i]
                # path = annotation_file_geopotencital[idx]
                # samples = np.load(path)

                # Create ndarray which accumulates all samples
                SAMPLES = []
                for j in range(i-averaging_sequence_length+1,i+1):
                    idx = train_input_indexes[j]
                    path = annotation_file_geopotencital[idx]
                    samples = np.load(path)
                    SAMPLES.append(samples)
                samples = np.mean(np.array(SAMPLES), axis=0)
                del SAMPLES

                # PAD
                if EastWest_pad or NorthSouth_pad:                
                    samples = Pad(samples, EastWest_pad, NorthSouth_pad)
                H = samples.shape[0]
                W = samples.shape[1]
                samples = samples.reshape(1,H,W)                    
            else:
                # # Accumulate samples
                # idx = train_input_indexes[i]
                # path = annotation_file_geopotencital[idx]
                # sample = np.load(path)
                SAMPLE = []
                for j in range(i-averaging_sequence_length+1,i+1):
                    idx = train_input_indexes[j]
                    path = annotation_file_geopotencital[idx]
                    sample = np.load(path)
                    SAMPLE.append(sample)
                sample = np.mean(np.array(SAMPLE), axis=0)
                del SAMPLE

                if EastWest_pad or NorthSouth_pad:
                    sample = Pad(sample, EastWest_pad, NorthSouth_pad)
                sample = sample.reshape(1,H,W)
                samples = np.append(samples, sample, axis=0)

        intermediate_list.append(samples)
        del samples

        samples = intermediate_list[0]
        for i in range(1, len(intermediate_list)):
            samples = np.append(samples, intermediate_list[i], axis=0)
            print(f"Appended {i} / {len(intermediate_list)} batch of samples")

        print(f"Samples shape: {samples.shape}")                
        
        samples = torch.from_numpy(samples)

        print(f"Tensor shape: {samples.shape}")                

        # Fit scaler
        if scaler == "StandardScaler":
            scaler = torch_scalers.TorchStandardScaler(
                main_root_scaler, scaler_name)
        elif scaler == "MinMaxScaler":
            scaler = torch_scalers.TorchMinMaxScaler(
                main_root_scaler, scaler_name)
        elif scaler == "StandardScaler_ReplaceZeroStd":
            scaler = torch_scalers.TorchStandardScaler_ReplaceZeroStd(
                main_root_scaler, scaler_name)
        elif scaler == "StandardScaler_GeneralStd":
            scaler = torch_scalers.TorchStandardScaler_GeneralStd(
                main_root_scaler, scaler_name)
        else:
            raise ValueError("Not yet implemented")

        scaler.fit(samples)

        print("Scaler is now fitted.")
        
    def load_sklearn_scaler(self, scaler_type, main_root_scaler, scaler_name):
        """
        scaler_type \in {"StandardScaler", "MinMaxScaler"}
        """
        if scaler_type == "StandardScaler": 
            scaler = torch_scalers.TorchStandardScaler(
                main_root_scaler, scaler_name)
        elif scaler_type == "MinMaxScaler": 
            scaler = torch_scalers.TorchMinMaxScaler(
                main_root_scaler, scaler_name)
        else:
            raise ValueError("Not yet implemented")

        return scaler

    @staticmethod
    def load_model(model, root_model, 
                model_name="trained_model_weights.pth"):
        # Check if path exists
        if os.path.exists(os.path.join(root_model, model_name)):
            
            TRAIN_LOSS = np.load(
                os.path.join(root_model,"TRAIN_LOSS.npz"))['loss']
            
            TEST_LOSS = np.load(
                os.path.join(root_model,"TEST_LOSS.npz"))['loss']
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model.load_state_dict(torch.load(os.path.join(
                root_model, model_name), map_location=device))
            
            return TRAIN_LOSS, TEST_LOSS, model
        
        else:
            
            raise ValueError(f"No model is saved in {root_model}")

    @staticmethod
    def DropoutScheduleR(epoch, p0=0.5, tau1=6., tau2=0.7):
        """
        Excepts current epoch and returns probability for dropout
        p for hidden layers.
        """
        p = p0 * 1/(tau1*epoch + 1)**tau2
        return p

    @staticmethod
    def DropoutScheduleE(epoch, p0=0.5, tau=0.5):
        """
        Excepts current epoch and returns probability for dropout
        p for hidden layers.
        """
        p = p0 * np.exp(-epoch/tau)
        return p



    def train_loop(self, model, train_dataloader, criterion,
                   optimizer, scheduler, device, num_batches,
                   p):#, averaging_sequence_length):

        print("\nTraining \n")

        train_loss = 0
        
        model.train()

        # print("1. Using {0} MB".format(
        #     psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) )           

        for num_batch, data_batch in enumerate(train_dataloader): 
            # # Keep track of batches
            # if num_batch % 20 == 0:
                
            #     print("Batch {0} / {1}".format(num_batch, num_batches))
            #     if num_batch != 0:
            #         epoch_time = num_batches * t_batch
            #         print(f"Estimated time per train_loop: {epoch_time/60} min")

            t1 = time.perf_counter()
            # Unpack input and target tensors and move them to device
            input_tensor, target_tensor = data_batch['input'], data_batch['target']
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            # Compute prediction error
            pred = model(input_tensor, p)

            # print(pred.shape, target_tensor.shape)
            loss = 0.1*torch.pow(criterion(pred, target_tensor), 1./256)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss change
            train_loss += loss.item() # .item() converts 1 element loss tensor to number
            
            t7 = time.perf_counter()
            # t_batch = t7 - t1 #s

        # Update lr_rate for next epoch
        if scheduler == None: pass
        else: scheduler.step(loss)

        train_loss /= num_batches

        # print(f"Training max {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")

        return train_loss

    def test_loop(self, model, test_dataloader, criterion, device, num_batches):

        print("\nTesting \n")

        test_loss = 0

        model.eval()

        # No need to keep track of tensor gradients
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                                
                # Unpack input and target tensors and move them to device
                input_tensor, target_tensor = data['input'], data['target']
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)

                # Compute prediction error
                pred = model(input_tensor)
                test_loss += criterion(pred, target_tensor).item()
                # test_loss += criterion(pred[:,0,:,:], target_tensor[:,0,:,:]).item()

        test_loss /= num_batches                                

        # print(f"Testing max: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")

        return test_loss

    def train(self, model, hyperparameters, root_model,
              scaler_type, root_scaler, SCALER_NAMES,
              transform_name, target_transform_name,
              transform, target_transform,
              EastWest_pad, NorthSouth_pad,
              start_date, end_date, 
              input_sequence_length, averaging_sequence_length,
              TARGET_VARIABLES):
        """
        It creates a training dataset, loads hyperparameters, creates criterion,
        optimizer and scheduler functions and starts training model using data
        from dataloader generator. 
        ----------------------------------------------------
        Args:
            model (nn.Module):          - architecture of the model with
                                          loaded initial weights
            hyperparameters (dict):     - dictionary of hyperparameters
                                          (learning rate, weight decay,
                                          multiplicator, batch_size)
            root_model (str):           - path to model directory
            scaler_type (str):          - \in {StandardScaler, MinMaxScaler}
            root_scaler (str):          - path to scaler directory 
            SCALER_NAMES (str):         - list of names of scalers e.g. "StandardScaler_3xReduced"
            transform_name (str):       
            target_transform_name (str):
            transform (cls)             - transform class e.g. ToTensor()
            target_transform (cls)      - target transform class e.g. ToTensor()
                                          or DecreaseDensity()
            EastWest_pad (int)
            NorthSouth_pad (int)
            start_date:                 - e.g. (1,1,1950)
            end_date:                   - e.g. (31,12,2020)
            TARGET_VARIABLES:           - list of True, False - True if variable should be contained in
                                          target fields
        Returns:
            LOSS (arr): array of LOSS-es during training (one test loss per epoch)
        """

        # 1. Check which device is available (cpu or cuda) and move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device={device}")
        model.to(device)

        # 2. Create folder to save the model weights and losses.
        #    If folder already exists, add current date and time to avoid duplication
        try:
            os.mkdir(os.path.join(root_model))
        except:
            t = time.localtime()
            current_time = time.strftime("%Y_%m_%d_%Hh_%Mm_%Ss", t)
            root_model = os.path.join(root_model) + "_" + str(current_time)
            os.mkdir(root_model)

        # 3. Load hyperparameters
        learning_rate = hyperparameters['learning_rate']         # Beginning lr_rate
        # learning_rate_SGD = hyperparameters['learning_rate_SGD'] # SGD lr_rate
        total_epoch = hyperparameters['total_epoch']             # Num of epochs
        factor = hyperparameters['factor']                       # lr_new = factor * lr_old
        patience = hyperparameters['patience']                   # Number of epochs with no improvement after which learning rate will be reduced
        threshold = hyperparameters['threshold']                 # Measure for when to reduce lr
        threshold_mode = hyperparameters['threshold_model']      # {rel, abs}
        batch_size = hyperparameters['batch_size']               # Size of batch 

        dropout_hyperparameters = hyperparameters['dropout_hyperparameters']
        if dropout_hyperparameters['SchedulingFunc'] == "R":
            p0_dropout = dropout_hyperparameters['p0']
            tau1_dropout = dropout_hyperparameters['tau1']
            tau2_dropout = dropout_hyperparameters['tau2']
        elif dropout_hyperparameters['SchedulingFunc'] == "E":
            p0_dropout = dropout_hyperparameters['p0']
            tau_dropout = dropout_hyperparameters['tau']

        # 4. Initialize scaler
        # --> Call scaler
        SCALERS = []
        for i in range(len(self.NAME)):
            scaler = self.load_sklearn_scaler(scaler_type, root_scaler, SCALER_NAMES[i])
            SCALERS.append(scaler)

        # Change start and end date so that sequences fit into dataset
        date_to_day = data_preparation.DateToConsecutiveDay()

        start_index = date_to_day.num_of_samples(
            start_date[2], start_date[1], start_date[0], 1,
            start_year=start_date[2], start_month=start_date[1],
            start_day=start_date[0])

        start_date = date_to_day.num_of_samples_to_date(
            index = start_index \
                    + input_sequence_length * averaging_sequence_length - 1, \
                    delta_t = 1,
            start_year=start_date[2], start_month=start_date[1],
            start_day=start_date[0]
        )        

        end_index = date_to_day.num_of_samples(
            end_date[2], end_date[1], end_date[0], 1,
            start_year=start_date[2], start_month=start_date[1],
            start_day=start_date[0])

        end_date = date_to_day.num_of_samples_to_date(
            index = end_index - averaging_sequence_length + 1, delta_t = 1,
            start_year=start_date[2], start_month=start_date[1],
            start_day=start_date[0]
        )        

        # 5. Create train and test datasets and dataloaders
        # --> Datasets
        train_dataset, test_dataset = self.train_test_datasets(
            SCALERS, transform_name, target_transform_name, 
            transform, target_transform,
            EastWest_pad, NorthSouth_pad,
            start_date, end_date,
            input_sequence_length, averaging_sequence_length,
            TARGET_VARIABLES)

        # --> Dataloaders (shuffle data at each epoch == True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,\
                                     shuffle=True, pin_memory=True, num_workers=2)

        test_dataloader = DataLoader(test_dataset, batch_size=1,\
                                     shuffle=False, pin_memory=True, num_workers=2) 

        num_batches_train = len(train_dataloader)
        num_batches_test = len(test_dataloader)

        # 6. Define criterion function, optimizer and scheduler
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 
                        factor=factor, 
                        patience=patience, 
                        threshold=threshold, # Razlaga thresholda: https://stackoverflow.com/questions/63108131/pytorch-schedule-learning-rate
                        threshold_mode=threshold_mode) 

        # 7. Create lists for tracking Losses during training
        TRAIN_LOSS = []
        TEST_LOSS = []
    
        best_loss = np.inf

        t_start = time.perf_counter()

        #total_epoch_adam = total_epoch - 3

        # 8. Run training
        #for epoch in range(total_epoch_adam):
        for epoch in range(total_epoch):
            # Start time
            t0 = time.perf_counter()
            print(f'\nEpoch: {epoch+1}/{total_epoch} ---------------------------------')

            # Dropout probability
            if dropout_hyperparameters['SchedulingFunc'] == "R":
                p = self.DropoutScheduleR(epoch, p0_dropout, tau1_dropout, tau2_dropout)
            elif dropout_hyperparameters['SchedulingFunc'] == "E":
                p = self.DropoutScheduleE(epoch, p0_dropout, tau_dropout)
            else:
                p = 0.

            # A) Train model
            train_loss = self.train_loop(model, train_dataloader,
                    criterion, optimizer, scheduler, device,
                    num_batches_train, p)#, averaging_sequence_length)

            # B) Validate model            
            test_loss = self.test_loop(model, test_dataloader,
                        criterion, device, num_batches_test)

            # C) Track changes
            print(f"Train loss: {train_loss},\nTest loss: {test_loss},\
                                    \nTime: {time.perf_counter()-t0} s")
            
            # D) Save best model and save model every 5 epochs
            if test_loss < best_loss:
                torch.save(model.state_dict(), os.path.join(root_model,
                                               'trained_model_weights.pth'))
                best_loss = test_loss         

            # if (epoch+1) % 5 == 0:
            #     torch.save(model.state_dict(), os.path.join(root_model,
            #                     'trained_model_weights'+f'_{epoch+1}'+'.pth'))

            TRAIN_LOSS.append(train_loss)
            TEST_LOSS.append(test_loss)

            t1 = time.perf_counter()
            print(f"Time per epoch: {(t1-t0)/60:.1f} min")
            print(f"Estimated time left: {(t1-t0)/60 * (total_epoch-(epoch+1)):.1f} min")

        np.savez(os.path.join(root_model, 'TRAIN_LOSS.npz'),
                 loss=np.array(TRAIN_LOSS))
        np.savez(os.path.join(root_model, 'TEST_LOSS.npz'),
                 loss=np.array(TEST_LOSS))

        t_end = time.perf_counter()
        print(f"Training time: {t_end-t_start}")
        print(f"Training max {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")

        # checkpoint = {
        #     'epoch': epoch,
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'scheduler': scheduler.state_dict()
        # }
        
        # torch.save(checkpoint, os.path.join(root_model, 'checkpoint.pth'))





        # checkpoint = torch.load(os.path.join(root_model, 'checkpoint.pth'))

        # model.load_state_dict(checkpoint['model'])

        # optimizer_SGD = torch.optim.SGD(model.parameters(), lr = learning_rate_SGD)

        # # 8. Run training
        # for epoch in range(total_epoch_adam , total_epoch):
        #     # Start time
        #     t0 = time.perf_counter()
        #     print(f'\nEpoch: {epoch+1}/{total_epoch} ---------------------------------')

        #     # A) Train model
        #     train_loss = self.train_loop(model, train_dataloader,
        #             criterion, optimizer_SGD, None, device,
        #             num_batches_train, p)

        #     # B) Validate model            
        #     test_loss = self.test_loop(model, test_dataloader,
        #                 criterion, device, num_batches_test)

        #     # C) Track changes
        #     print(f"Train loss: {train_loss},\nTest loss: {test_loss},\
        #                             \nTime: {time.perf_counter()-t0} s")
            
        #     # D) Save best model
        #     if test_loss < best_loss:
        #         torch.save(model.state_dict(), os.path.join(root_model,
        #                                        'trained_model_weights.pth'))
        #         best_loss = test_loss         

        #     # Save every 5 epochs
        #     # if (epoch+1) % 5 == 0:
        #     #     torch.save(model.state_dict(), os.path.join(root_model,
        #     #                     'trained_model_weights'+f'_{epoch+1}'+'.pth'))

        #     TRAIN_LOSS.append(train_loss)
        #     TEST_LOSS.append(test_loss)

        #     t1 = time.perf_counter()
        #     print(f"Time per epoch: {(t1-t0)/60:.1f} min")
        #     print(f"Estimated time left: {(t1-t0)/60 * (total_epoch-(epoch+1)):.1f} min")

        # np.savez(os.path.join(root_model, 'TRAIN_LOSS.npz'),
        #          loss=np.array(TRAIN_LOSS))
        # np.savez(os.path.join(root_model, 'TEST_LOSS.npz'),
        #          loss=np.array(TEST_LOSS))

        # t_end = time.perf_counter()
        # print(f"Training time: {t_end-t_start}")
        # print(f"Training max {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024} MB")

        # checkpoint = {
        #     'epoch': epoch,
        #     'model': model.state_dict(),
        #     'optimizer': optimizer_Adam.state_dict(),
        #     'scheduler': scheduler.state_dict()
        # }
        # torch.save(checkpoint, os.path.join(root_model, 'checkpoint.pth'))

        return TRAIN_LOSS, TEST_LOSS, root_model

#%%
if __name__ == "__main__":

    #------------------------
    # Training class instance
    #------------------------
    training = Training(
        "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density",
        "geopotential_annotation_file.txt")

    # --> Choose time interval for training dataset
    start_date = (1, 1, 1950)    
    end_date = (31, 12, 1968)
    # --> Get start and end indexes for annotation_file
    date_to_day = data_preparation.DateToConsecutiveDay()

    start_index = date_to_day.num_of_samples(
        start_date[2], start_date[1], start_date[0], delta_t=1)
    end_index = date_to_day.num_of_samples(
        end_date[2], end_date[1], end_date[0], delta_t=1)
    print(start_index, end_index)        

#%%
if __name__ == "__main__":

    #------------------------
    #       Scaling
    #------------------------
    # Training class instance
    training = Training(
        "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density",
        "geopotential_annotation_file.txt")

    # --> Call scaler
    training.fit_and_save_sklearn_scaler(
        "StandardScaler", 
        "/home/perkanu/Magistrska_naloga/U-Net/Experiment_1_1950-2020/Scalers",
        "StandardScaler_3xReduced_Geopotential_1950-2020",
        EastWest_pad = 0, NorthSouth_pad=0,
        start_date=(1,1,1950), end_date=(31,12,2021),
        input_sequence_length=7
    )

    scaler = training.load_sklearn_scaler(
        "StandardScaler",
        "/home/perkanu/Magistrska_naloga/U-Net/Experiment_1_1950-2020/Scalers",
        "StandardScaler_3xReduced_Geopotential")

    # Check it works
    torch_tensor = torch.rand(1,60,120)+50000
    print(torch_tensor[0,0,0])
    ndarray_scaled = scaler.transform(torch_tensor)
    print(ndarray_scaled[0,0,0])
    ndarray_inverse = scaler.inverse_transform(torch_tensor)
    print(ndarray_inverse[0,0,0])

# %%
