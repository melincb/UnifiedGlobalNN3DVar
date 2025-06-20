# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Script with methods for model testing.

@author: Uroš Perkan
"""

# LIBRARIES
import os
import torch
import copy
import numpy as np

import torch_scalers
import data_preparation_1hr
from train import Pad
import itertools


# PREDICTING CLASS

class Predicting():
    """
    Class that stores methods for predicting.   
    ----------------------------------------------------
    Args
    ----------------------------------------------------
    root_model (str):  - main path to directory with model
    model_name (str):  - name of model e.g. "trained_model_weights.pth"
    root_scaler (str): - main path to directory with scaler
    scaler_name (str): - name of scaler 
                         e.g. "StandardScaler_3xReduced_Geopotential_1950-2020.pt"
    scaler_type (str): - \in {StandardScaler, MinMaxScaler}
    ----------------------------------------------------
    Methods
    ---> load_sklearn_scaler
    ---> load_model
    ---> predict_next
    ---> pred_time_series
    ---> truth_time_series
    ----------------------------------------------------
    """

    def __init__(self, ROOTS_MODEL, model_name):
        self.ROOTS_MODEL = ROOTS_MODEL
        self.model_name = model_name
    
    @staticmethod
    def load_sklearn_scaler(root_scaler, scaler_name, scaler_type):
        
        if scaler_type == "StandardScaler": 
            scaler = torch_scalers.TorchStandardScaler(
                root_scaler, scaler_name)
        elif scaler_type == "MinMaxScaler": 
            scaler = torch_scalers.TorchMinMaxScaler(
                root_scaler, scaler_name)
        else:
            raise ValueError("Not yet implemented")
            
        return scaler

    @staticmethod
    def load_model(model, path_to_model, 
                model_name="trained_model_weights.pth"):
        # Check if path exists
        if os.path.exists(os.path.join(path_to_model, model_name)):
            
            TRAIN_LOSS = np.load(
                os.path.join(path_to_model,"TRAIN_LOSS.npz"))['loss']
            
            TEST_LOSS = np.load(
                os.path.join(path_to_model,"TEST_LOSS.npz"))['loss']
            
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

            model.load_state_dict(torch.load(os.path.join(
                path_to_model, model_name), map_location=device))
            
            return TRAIN_LOSS, TEST_LOSS, model
        
        else:
            
            raise ValueError(f"No model is saved in {path_to_model}")


    def predict_next(
            self, MODELS, INPUT_SEQUENCE, STATIC_TENSOR,
            root_scaler, SCALER_NAMES, SCALER_TYPES):
        """
        Args:
        ---> model: pytorch model
        ---> input_field: ndarray of shape (H, W)
        ---> STATIC_TENSOR: tensor of static fields in correct order
        ---> model: pytorch model
        """

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        # N models for individual variables
        # print('MODELS', MODELS)
        N = len(MODELS)
        
        # 1. Load model and put it in eval mode
        for i in range(N):
            MODELS[i].to(device)
            MODELS[i].load_state_dict(torch.load(
                os.path.join(self.ROOTS_MODEL[i], self.model_name),
                map_location=device))
            MODELS[i].eval()
            # print(i, MODELS[i].training)       

        # 2 Pad field
        INPUT_SEQUENCE = copy.deepcopy(INPUT_SEQUENCE)
        # Fields will not be padded so this doesn't matter
        # for input_field in input_field_sequence:
        #     paded_input_field = Pad(input_field, EastWest_pad, NorthSouth_pad)

        # # 3 Transform field to torch.tensor
        # H = input_field.shape[0]
        # W = input_field.shape[1]

        # input_field = input_field.reshape(1,H,W)
        # input_tensor = torch.from_numpy(input_field)
        # print(INPUT_SEQUENCE.shape)
        INPUT_TENSOR = torch.from_numpy(INPUT_SEQUENCE)
        # 4. Load scaler and scale field
        SCALERS = []

        for i in range(len(SCALER_NAMES)):
            scaler = self.load_sklearn_scaler(
                root_scaler, SCALER_NAMES[i], SCALER_TYPES[i]) 
            SCALERS.append(scaler)
        
        

        # SCALED_TENSOR = torch.clone(INPUT_TENSOR)
        # for i in range(len(SCALER_NAMES)):
        #     # scaled_tensor = SCALERS[i].transform(INPUT_TENSOR[i])
        #     print(torch.amax(INPUT_TENSOR[:,i]), torch.amin(INPUT_TENSOR[:,i]))
        #     scaled_tensor = SCALERS[i].transform(INPUT_TENSOR[:,i])
        #     print(torch.amax(scaled_tensor), torch.amin(scaled_tensor))
        #     # scaled_tensor = torch.reshape(scaled_tensor, (1,1,H,W))
        #     # scaled_tensor = torch.unsqueeze(scaled_tensor, 0)
        #     SCALED_TENSOR[:,i] = scaled_tensor

        # SCALED_TENSOR = SCALED_TENSOR.to(device)
        # print(torch.amax(INPUT_TENSOR[:,i]), torch.amin(INPUT_TENSOR[:,i]))

        num_variables = len(SCALER_NAMES)
        input_sequence_len = int(len(INPUT_SEQUENCE[0]) / num_variables)

        SCALED_TENSOR = torch.clone(INPUT_TENSOR)
        input_sequence_index = 0
        for i in range(num_variables):

            for j in range(input_sequence_len):
                # scaled_tensor = SCALERS[i].transform(INPUT_TENSOR[i])
                scaled_tensor = SCALERS[i].transform(INPUT_TENSOR[:,input_sequence_index])
                # scaled_tensor = torch.reshape(scaled_tensor, (1,1,H,W))
                # scaled_tensor = torch.unsqueeze(scaled_tensor, 0)
                SCALED_TENSOR[:,input_sequence_index] = scaled_tensor
                input_sequence_index += 1
                
        SCALED_TENSOR = SCALED_TENSOR.to(device)

        # 5. Calculate next step
        
        ### Označuje komentarje, ki se nanašajo na računanje gradienta
        ### Če računaš gradient, potem zakomentiraj "with torch.no_grad():" in odkomentiraj "if True:"
        with torch.no_grad():
        # if True:

            SCALED_PRED = torch.clone(SCALED_TENSOR)[:,::input_sequence_len]

            # Napovedi vseh modelov je treba spraviti v PRED, nato pa inverzno skalirati od zunaj.
            # N je v našem primeru 1
            for i in range(N):
                if type(STATIC_TENSOR) == type(1):
                    # To se izvede, če nimamo statičnih polj v modelu
                    DIRECT_PRED = MODELS[i](SCALED_TENSOR)
                else:
                    # To se izvede, če imamo statična polja
                    ### Če računaš gradient zakomentiraj spodnjo vrstico "DIRECT_PRED" in odkomentiraj drugo vrstico (kjer je # INPUT_FIELD = torch.cat((SCALED_TENSOR, STATIC_TENSOR), dim=1).requires_grad_(True))
                    DIRECT_PRED = MODELS[i](torch.cat((SCALED_TENSOR, STATIC_TENSOR), dim=1))
                    ### Če računaš gradiente, odkomentiraj spodnjih enajst vrstic
                    # INPUT_FIELD = torch.cat((SCALED_TENSOR, STATIC_TENSOR), dim=1).requires_grad_(True)
                    # DIRECT_PRED = MODELS[i](INPUT_FIELD)
                    # DIRECT_PRED.sum().backward()
                    # INPUT_FIELD_GRADIENT = INPUT_FIELD.grad # (vsebuje tako gradient aktivnih spremenljivk, kot statičnih polj itd.)
                    # # Oblika polja gradientov: [1, 27, 180, 360]
                    # # Vsebuje gradiente vseh 27 spremenljivk
                    # # prvih 24 spremenljivk je v enakem vrstnem redu, 
                    # # kot v zagon.ipynb v tabeli VARIABLES.
                    # # Zadnje 3 spremenljivke so statična polja, v vrstnem redu
                    # # 24=latitudes, 25=land_sea_mask, 26=surface_topography  
                    # assert False

                ### ČE RAČUNAŠ GRAFIENTE KODA OD TU NAPREJ MORDA NE BO DELOVALA (OPTIMALNO)
                ### -> če so težave, piši Urošu

                if SCALED_PRED.shape == DIRECT_PRED.shape:
                    # Modelska napoved je že v pravi obliki
                    SCALED_PRED = DIRECT_PRED
                else:
                    # Modelska napoved je v obliki (1,1,60,120).
                    # Moramo jo inkorporirati v SCALED_PRED
                    SCALED_PRED[:,i] = DIRECT_PRED[:,0]

            PRED = SCALED_PRED
            for i in range(len(SCALER_NAMES)):
                # Inverzno skaliranje
                # for i in range(len(SCALER_NAMES)):
                pred = SCALERS[i].inverse_transform(SCALED_PRED[:,i])
                # pred = pred[0,0,:,:]
                # pred = pred[0,:,:,:]
                PRED[:,i] = pred

            # if EastWest_pad > 0:
            #     pred = pred[:,EastWest_pad:-EastWest_pad]
            # if NorthSouth_pad > 0:
            #     pred = pred[NorthSouth_pad:-NorthSouth_pad,:]

            return PRED.cpu().detach().numpy()

    def pred_time_series(self, root_annotations, ANNOTATIONS_NAMES,
                    MODELS, STATIC_FIELDS_ROOTS, 
                    root_scaler, SCALER_NAMES, SCALER_TYPES, 
                    start_date, num_of_prediction_steps,
                    input_sequence_len=1, averaging_sequence_len=1,
                    delta_t = 1./24, T = 0,
                    use_ERA5=True, list_of_input_fields=None):
        """
        Args:
        ---> root_annotations (str): main path to directory with annotations_file.
        ---> annotations_name (str): name of annotations file 
                                     e.g. geopotential_annotation_file
        ---> MODELS: list of pytorch model architectures
        ---> start_date: (day, mont, year)
        ---> num_of_prediction_steps: how many steps do you wish to predict
        ---> EastWest_pad
        ---> NorthSouth_pad
        ---> averaging_sequence_len: how many days are averaged in the input of the model
        ---> delta_t = 1./24 <- delta t in fraction of the day
        Returns:
        ---> fields, preds - lists of truths and predictions, first element
                             of preds is the truth. 
        """

        date_to_day = data_preparation_1hr.DateToConsecutiveDay()

        # --> Get index for annotation_file
        index = date_to_day.num_of_samples(
            start_date[2], start_date[1], start_date[0], delta_t=delta_t)

        index = index + T

        # --> Load fields annotations (path to dataset fields)
        ANNOTATION_FILES = []
        for i in range(len(ANNOTATIONS_NAMES)):
            annotation_file = np.loadtxt(
                os.path.join(root_annotations, ANNOTATIONS_NAMES[i]), dtype=np.str_)
            ANNOTATION_FILES.append(annotation_file)
        
        # Truth fields
        FIELDS = []
        for variable_index in range(len(ANNOTATION_FILES)):
            fields = []
            k = 0
            for i in range(num_of_prediction_steps+1):
                intermediate_fields = []
                for j in range(averaging_sequence_len):
                    path = ANNOTATION_FILES[variable_index][index+k]
                    # if variable_index == 0:
                    #     print(path)
                    field = np.load(path)
                    intermediate_fields.append(field)
                    k += 1
                fields.append(np.average(np.array(intermediate_fields), axis=0))
            FIELDS.append(fields)

        # Static fields
        if len(STATIC_FIELDS_ROOTS) > 0:
            STATIC_FIELDS = [] # [st_var1, st_var2, ..., st_varN]
            for static_field_path in STATIC_FIELDS_ROOTS: 
                static_field = np.load(static_field_path)
                # static_field = torch.from_numpy(static_field)
                STATIC_FIELDS.append(static_field)
                # Scalerjev za statična polja ne dodamo, ker jih ne skaliramo
            # for ndarray_field in STATIC_FIELDS:
            #     tensor_field = torch.from
            STATIC_FIELDS = np.array(STATIC_FIELDS)
        else:
            STATIC_FIELDS = []

        if len(STATIC_FIELDS) > 0:
            STATIC_TENSOR = torch.reshape(torch.from_numpy(STATIC_FIELDS), (1,STATIC_FIELDS.shape[0],STATIC_FIELDS.shape[1],STATIC_FIELDS.shape[2]))
        else:
            STATIC_TENSOR = 1

        # # Input sequence
        # INPUT_SEQUENCE = []
        # for variable_index in range(len(ANNOTATION_FILES)):
        #     input_sequence = [fields[0]]
        #     k = 0
        #     for i in range(input_sequence_len-1):
        #         intermediate_fields = []
        #         for j in range(averaging_sequence_len):
        #             path = ANNOTATION_FILES[variable_index][index-k-1]
        #             field = np.load(path)
        #             intermediate_fields.append(field)
        #             k += 1
        #         input_sequence.append(np.average(np.array(intermediate_fields), axis=0))
        #     # input_sequence = list(np.flip(np.array(input_sequence), axis=0))
        #     input_sequence = list(np.array(input_sequence)) # Verjetno bo treba apendati na začetek
        #     INPUT_SEQUENCE.append(input_sequence)


        if use_ERA5:
            # Input sequence
            INPUT_SEQUENCE = [[] for _ in range(len(ANNOTATION_FILES))]

            for variable_index in range(len(ANNOTATION_FILES)):
                # input_sequence = [fields[0]]
                input_sequence = [FIELDS[variable_index][0]]
                k = 0
                for i in range(input_sequence_len-1):
                    intermediate_fields = []
                    for j in range(averaging_sequence_len):
                        path = ANNOTATION_FILES[variable_index][index-k-1]
                        field = np.load(path)
                        intermediate_fields.append(field)
                        k += 1
                    input_sequence.append(np.average(np.array(intermediate_fields), axis=0))
                # input_sequence = list(np.flip(np.array(input_sequence), axis=0))
                input_sequence = list(np.array(input_sequence))
                INPUT_SEQUENCE[variable_index] = input_sequence
            # print('aaaaaaaaaaa', np.array(INPUT_SEQUENCE).dtype, np.shape(INPUT_SEQUENCE))
        else:
            INPUT_SEQUENCE = list_of_input_fields
            if np.array(INPUT_SEQUENCE).dtype != np.float32:
                IS1 = []
                for iseq in INPUT_SEQUENCE:
                    IS1.append(list(np.array(iseq).astype(np.float32)))
                INPUT_SEQUENCE = IS1.copy()
            # print('aaaaaaaaaaa', np.array(INPUT_SEQUENCE).dtype, np.shape(INPUT_SEQUENCE))

        # print(index)
        # print(INPUT_SEQUENCE)
        # INPUT_SEQUENCE = [list(itertools.chain(*INPUT_SEQUENCE))]
        
        # INPUTE_SEQUENCE[i] = [fields[-3], fields[-2], fields[-1], fields[0]] |_var_i
        
        num_pred = 0
        # preds = [fields[0]]
        PREDS = []
        # for i in range(len(FIELDS)):
        for i in range(len(ANNOTATION_FILES)):
            if use_ERA5:
                preds = [FIELDS[i][0]]
            else:
                preds = [INPUT_SEQUENCE[i][0]]
            PREDS.append(preds)


        # RREDS = [ [var_1_0, ..., var_1_p], ..., [var_k_0, ..., var_k_p] ]
        # var_i_j shape = (60,120)
        # for i in range(num_of_prediction_steps):
        while num_pred < num_of_prediction_steps:
            
            # predi = self.predict_next(
            #     model=model,
            #     input_field_sequence=np.array(input_sequence[-input_sequence_len:]),
            #     # input_field_sequence=np.expand_dims(preds[-1],0),
            # )

            predi = self.predict_next(
                MODELS=MODELS,
                INPUT_SEQUENCE=np.array([list(itertools.chain(*INPUT_SEQUENCE[:]))])[:,:input_sequence_len*len(ANNOTATION_FILES)],
                STATIC_TENSOR = STATIC_TENSOR,
                root_scaler=root_scaler, SCALER_NAMES=SCALER_NAMES, SCALER_TYPES=SCALER_TYPES
                # input_field_sequence=np.expand_dims(preds[-1],0),           
            )
            
            # preds.append(predi)
            # input_sequence.append(predi)
            for i in range(len(PREDS)):
                # PREDS[i].append(predi[i])
                # INPUT_SEQUENCE[i].append(predi[i])
                PREDS[i].append(predi[0,i])
                # INPUT_SEQUENCE[i].append(predi[0,i])
                INPUT_SEQUENCE[i].insert(0, predi[0,i])
                del INPUT_SEQUENCE[i][-1]
            
            num_pred += 1
        
        return FIELDS, PREDS
        # return fields, preds

    @staticmethod
    def truth_time_series(root_annotations, annotation_name,
                             start_date, num_of_prediction_steps,
                             averaging_sequence_len,
                             delta_t = 1./24):
        """
        Args:
        ---> root_annotations (str): main path to directory with annotations_file.
        ---> annotations_name (str): name of annotations_file.
        ---> start_date: (day, mont, year)
        ---> num_of_prediction_steps: how many steps do you wish to predict
        ---> delta_t = 1./24 <- delta t in fraction of the day
        Returns:
        ---> fields - lists of truths
        """

        date_to_day = data_preparation_1hr.DateToConsecutiveDay()
        
        # --> Get index for annotation_file
        index = date_to_day.num_of_samples(
            start_date[2], start_date[1], start_date[0], delta_t=delta_t)

        # --> Load fields annotations (path to dataset fields)
        # annotation_file_geopotencital = np.loadtxt(
        #     os.path.join(root_annotations, annotations_name), dtype=np.str_)

        annotation_file = np.loadtxt(
            os.path.join(root_annotations, annotation_name), dtype=np.str_)

        # fields = []

        # for i in range(num_of_prediction_steps+1):
        #     path = annotation_file_geopotencital[index+i]
        #     field = np.load(path)
        #     fields.append(field)

        fields = []

        k = 0
        for i in range(num_of_prediction_steps+1):
            intermediate_fields = []
            for j in range(averaging_sequence_len):
                path = annotation_file[index+k]
                field = np.load(path)
                intermediate_fields.append(field)
                k += 1
            fields.append(np.average(np.array(intermediate_fields), axis=0))

        # # Truth fields
        # FIELDS = []
        # for variable_index in range(len(ANNOTATION_FILES)):
        #     fields = []
        #     k = 0
        #     for i in range(num_of_prediction_steps+1):
        #         intermediate_fields = []
        #         for j in range(averaging_sequence_len):
        #             path = ANNOTATION_FILES[variable_index][index+k]
        #             field = np.load(path)
        #             intermediate_fields.append(field)
        #             k += 1
        #         fields.append(np.average(np.array(intermediate_fields), axis=0))
        #     FIELDS.append(fields)

        return fields # FIELDS
    