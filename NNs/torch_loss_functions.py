import torch
import torch.nn as nn
import numpy as np
import copy

# Constants
torch.pi = torch.acos(torch.zeros(1)).item() * 2

# Helper functions
def Pad(fourdtensor, EastWest_pad, NorthSouth_pad):
    
    # NORTH AND SOUTH PADDING
    if NorthSouth_pad > 0:
        # # Top of matrix transformation
        # top = torch.flip(fourdtensor[:,:,0:NorthSouth_pad,:], dims=(-2,))
        # top = torch.roll(top, shifts=int(top.shape[-1]/2), dims=-1)
        # # Bottom of matrix transformation
        # bottom = torch.flip(fourdtensor[:,:,-NorthSouth_pad:,:], dims=(-2,))
        # bottom = torch.roll(bottom, shifts=int(bottom.shape[-1]/2), dims=-1)
        # # Stack together
        # arr = torch.concat((top, fourdtensor, bottom), dim=-2)
        top = fourdtensor[:,:,0:NorthSouth_pad,:]
        bottom = fourdtensor[:,:,-NorthSouth_pad:,:]
        arr = torch.concat((top, fourdtensor, bottom), dim=-2)
    else:
        arr = fourdtensor

    # EAST AND WEST PADDING
    if EastWest_pad == 0:
        return arr
    else:
        left = arr[:,:,:,0:EastWest_pad]
        right = arr[:,:,:,-EastWest_pad:]
        arr = torch.concat((right, arr, left),dim=-1)
        return arr

# LOSS FUNKCIJE

class ACCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, clima):
        pred = torch.flatten(pred)
        target = torch.flatten(target)
        clima = torch.flatten(clima)
        stevec = torch.mean((pred - clima) * (target - clima))
        imenovalec = torch.sqrt(
            torch.mean(torch.square(pred - clima)) * 
            torch.mean(torch.square(target - clima)) )
        acc_loss = - (stevec / imenovalec - 1)
        return acc_loss 

class MSE_ACCLoss(nn.Module):
    def __init__(self, power_MSE, power_ACC, constant):
        super().__init__()
        self.powMSE = power_MSE
        self.powACC = power_ACC
        self.const = constant
        self.MSE_criterion = torch.nn.MSELoss()
        
    def forward(self, pred, target, clima):
        pred = torch.flatten(pred)
        target = torch.flatten(target)
        clima = torch.flatten(clima)
        stevec = torch.mean((pred - clima) * (target - clima))
        imenovalec = torch.sqrt(
            torch.mean(torch.square(pred - clima)) * 
            torch.mean(torch.square(target - clima)) )
        acc_loss = - (stevec / imenovalec - 1)
        mse_loss = self.MSE_criterion(pred, target)
        return self.const*torch.sqrt(torch.pow(mse_loss, self.powMSE)*torch.pow(acc_loss, self.powACC))

class MSE_plus_ACC_Loss(nn.Module):
    def __init__(self, power_MSE, power_ACC, constantMSE, constantACC):
        super().__init__()
        self.powMSE = power_MSE
        self.powACC = power_ACC
        self.constMSE = constantMSE
        self.constACC = constantACC
        self.MSE_criterion = torch.nn.MSELoss()
        
    def forward(self, scaled_pred, scaled_target, inversescaled_pred, inversescaled_target, inversescaled_clima):
        # MSE
        MSE_pred = torch.flatten(scaled_pred)
        MSE_target = torch.flatten(scaled_target)
        mse_loss = self.MSE_criterion(MSE_pred, MSE_target)
        # ACC
        ACC_pred = torch.flatten(inversescaled_pred)
        ACC_target = torch.flatten(inversescaled_target)
        ACC_clima = torch.flatten(inversescaled_clima)
        stevec = torch.mean((ACC_pred - ACC_clima) * (ACC_target - ACC_clima))
        imenovalec = torch.sqrt(
            torch.mean(torch.square(ACC_pred - ACC_clima)) * 
            torch.mean(torch.square(ACC_target - ACC_clima)) )
        acc_loss = - (stevec / imenovalec - 1)
        # print(f"mse_loss: {mse_loss}")
        # print(f"acc_loss: {acc_loss}")
        # print(f"MSE term {self.constMSE*torch.pow(mse_loss, self.powMSE)}")
        # print(f"ACC term {self.constACC*torch.pow(acc_loss, self.powACC)}")
        # print(f"Skupaj: {self.constMSE*torch.pow(mse_loss, self.powMSE) + self.constACC*torch.pow(acc_loss, self.powACC)}\n")
        return self.constMSE*torch.pow(mse_loss, self.powMSE) + self.constACC*torch.pow(acc_loss, self.powACC), stevec/imenovalec

class MSELoss_pad(nn.Module):
    def __init__(self, pad_NS, pad_EW):
        super().__init__()
        self.pad_NS = pad_NS
        self.pad_EW = pad_EW

    def forward(self, pred, target):

        mse_loss_center = torch.mean(
            (pred[:,:,2*self.pad_NS+1:-2*self.pad_NS-1, 2*self.pad_EW+1:-2*self.pad_EW-1] -\
             target[:,:,2*self.pad_NS+1:-2*self.pad_NS-1, 2*self.pad_EW+1:-2*self.pad_EW-1])**2
        )
        mse_loss_left = torch.mean(
            (pred[:,:,0:2*self.pad_NS+1,:] -\
             target[:,:,0:2*self.pad_NS+1,:])**2
        )
        mse_loss_right = torch.mean(
            (pred[:,:,-2*self.pad_NS-1:,:] -\
             target[:,:,-2*self.pad_NS-1:,:])**2
        )
        mse_loss_bottom = torch.mean(
            (pred[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,0:2*self.pad_EW+1] -\
             target[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,0:2*self.pad_EW+1])**2
        )
        mse_loss_top = torch.mean(
            (pred[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,-2*self.pad_EW-1:] -\
             target[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,-2*self.pad_EW-1:])**2
        )

        mse_loss = mse_loss_center +\
                 0.5*(mse_loss_left+mse_loss_right+mse_loss_bottom+mse_loss_top)
        
        return mse_loss 

class MSELoss_pad_4x(nn.Module):
    def __init__(self, pad_NS, pad_EW):
        super().__init__()
        self.pad_NS = pad_NS
        self.pad_EW = pad_EW

    def forward(self, pred, target):

        mse_loss_center = torch.mean(
            (pred[:,:,2*self.pad_NS+1:-2*self.pad_NS-1, 2*self.pad_EW+1:-2*self.pad_EW-1] -\
             target[:,:,2*self.pad_NS+1:-2*self.pad_NS-1, 2*self.pad_EW+1:-2*self.pad_EW-1])**2
        )
        mse_loss_left = torch.mean(
            (pred[:,:,0:2*self.pad_NS+1,:] -\
             target[:,:,0:2*self.pad_NS+1,:])**2
        )
        mse_loss_right = torch.mean(
            (pred[:,:,-2*self.pad_NS-1:,:] -\
             target[:,:,-2*self.pad_NS-1:,:])**2
        )
        mse_loss_bottom = torch.mean(
            (pred[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,0:2*self.pad_EW+1] -\
             target[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,0:2*self.pad_EW+1])**2
        )
        mse_loss_top = torch.mean(
            (pred[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,-2*self.pad_EW-1:] -\
             target[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,-2*self.pad_EW-1:])**2
        )

        mse_loss = mse_loss_center +\
                 2*(mse_loss_left+mse_loss_right+mse_loss_bottom+mse_loss_top)
        
        return mse_loss 


class MSELoss_surface(nn.Module):
    def __init__(self, pad_NS, pad_EW, batch_shape_without_padding):
        super().__init__()
        self.pad_NS = pad_NS
        self.pad_EW = pad_EW

        # batch_shape_without_padding = (N,C,H,W)         
        # Tensor of the same shape as batch without padding
        phi_tensor = torch.full(batch_shape_without_padding, 0)
        
        # Fill tensor with cos(phi)
        N = batch_shape_without_padding[0]
        C = batch_shape_without_padding[1]
        H = batch_shape_without_padding[2]
        W = batch_shape_without_padding[3]
        d_phi = torch.pi/H
        if H == 60:
            phi_0 = -88.5*torch.pi/180
        elif H == 180:
            phi_0 = -89.5*torch.pi/180
        else:
            raise ValueError(
                "This resolution has not jet been implemented.\
                 Check MSE_Loss_surface - torch_loss_functions.py")

        for i in range(H):
            phi_tensor[:,:,i,:] = torch.full((N,C,W), phi_0 + i*d_phi)

        # Cosine phi
        cos = torch.cos(phi_tensor)

        # Pad tensor
        self.cos = Pad(cos, self.pad_EW, self.pad_NS)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cos = self.cos.to(device)

    def forward(self, pred, target):

        N = pred.shape[0]
        print(N)

        # Multiply with cos(phi) tensor (pred-tar)*cos(phi) = pred*cos(phi)-tar*cos(phi)
        pred = pred*self.cos[:N,:,:,:]
        target = target*self.cos[:N,:,:,:]

        mse_loss_center = torch.mean(
            (pred[:,:,2*self.pad_NS+1:-2*self.pad_NS-1, 2*self.pad_EW+1:-2*self.pad_EW-1] -\
             target[:,:,2*self.pad_NS+1:-2*self.pad_NS-1, 2*self.pad_EW+1:-2*self.pad_EW-1])**2
        )
        mse_loss_left = torch.mean(
            (pred[:,:,0:2*self.pad_NS+1,:] -\
             target[:,:,0:2*self.pad_NS+1,:])**2
        )
        mse_loss_right = torch.mean(
            (pred[:,:,-2*self.pad_NS-1:,:] -\
             target[:,:,-2*self.pad_NS-1:,:])**2
        )
        mse_loss_bottom = torch.mean(
            (pred[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,0:2*self.pad_EW+1] -\
             target[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,0:2*self.pad_EW+1])**2
        )
        mse_loss_top = torch.mean(
            (pred[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,-2*self.pad_EW-1:] -\
             target[:,:,2*self.pad_NS+1:-2*self.pad_NS-1,-2*self.pad_EW-1:])**2
        )

        mse_loss = mse_loss_center +\
                 0.5*(mse_loss_left+mse_loss_right+mse_loss_bottom+mse_loss_top)
        
        return mse_loss 


# LOSS FUNKCIJE

class MSELoss_latitude_weighted(nn.Module):
    def __init__(self, power_MSE, constant, batch_shape_without_padding): # batch_shape_without_padding):
        super().__init__()
        """
        batch_shape_without_padding = (C,H,W)
        """
        self.powMSE = power_MSE
        self.const = constant
        # self.MSE_criterion = torch.nn.MSELoss()
        
        # # batch_shape_without_padding = (N,C,H,W)         
        # # Tensor of the same shape as batch without padding
        # # phi_tensor = torch.full(batch_shape_without_padding, 0)
        # phi_tensor = torch.full(batch_shape_without_padding, 0.0, dtype=torch.float32)

        # # Fill tensor with cos(phi)
        # N = batch_shape_without_padding[0]
        # C = batch_shape_without_padding[1]
        # H = batch_shape_without_padding[2]
        # W = batch_shape_without_padding[3]
        # d_phi = torch.pi/H
        # if H == 60:
        #     phi_0 = -88.5*torch.pi/180
        # elif H == 180:
        #     phi_0 = -89.5*torch.pi/180
        # else:
        #     raise ValueError(
        #         "This resolution has not jet been implemented.\
        #          Check MSE_Loss_surface - torch_loss_functions.py")
        # for i in range(H):
        #     phi_tensor[:,:,i,:] = torch.full((N,C,W), phi_0 + i*d_phi)

        # # Cosine phi
        # cos = torch.cos(phi_tensor)
        # S_avg = sum(cos[0,0,:,0])
        # cos = H*cos/S_avg
        # self.cos = torch.flatten(cos)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.cos = self.cos.to(device)

        phi_tensor = torch.full(batch_shape_without_padding, 0.0, dtype=torch.float32)
        # Fill tensor with cos(phi)
        # N = batch_shape_without_padding[0]
        C = batch_shape_without_padding[0]
        H = batch_shape_without_padding[1]
        W = batch_shape_without_padding[2]
        d_phi = torch.pi/H
        if H == 60:
            phi_0 = -88.5*torch.pi/180
        elif H == 180:
            phi_0 = -89.5*torch.pi/180
        else:
            raise ValueError(
                "This resolution has not jet been implemented.\
                 Check MSE_Loss_surface - torch_loss_functions.py")
        for i in range(H):
            phi_tensor[:,i,:] = torch.full((C,W), phi_0 + i*d_phi)

        # Cosine phi
        cos = torch.cos(phi_tensor)
        S_avg = sum(cos[0,:,0])
        cos = H*cos/S_avg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cos = cos.to(device)

    def forward(self, pred, target):
        # if len(torch.flatten(pred)) == len(self.cos):
        #     pred = torch.flatten(pred)
        #     target = torch.flatten(target)
        #     # mse_loss = self.MSE_criterion(pred, target)
        #     mse_loss = torch.mean( self.cos * (pred - target)**2)
        #     return self.const*torch.sqrt(torch.pow(mse_loss, self.powMSE))
        # else:
        #     # Fill tensor with cos(phi)
        #     N = pred[0]
        #     C = pred[1]
        #     H = pred[2]
        #     W = pred[3]

        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     phi_tensor = torch.full((N,C,H,W), 0.0, dtype=torch.float32, device=device)

        #     d_phi = torch.pi/H
        #     if H == 60:
        #         phi_0 = -88.5*torch.pi/180
        #     elif H == 180:
        #         phi_0 = -89.5*torch.pi/180
        #     else:
        #         raise ValueError(
        #             "This resolution has not jet been implemented.\
        #             Check MSE_Loss_surface - torch_loss_functions.py")
        #     for i in range(H):
        #         phi_tensor[:,:,i,:] = torch.full((N,C,W), phi_0 + i*d_phi)

        #     # Cosine phi
        #     cos = torch.cos(phi_tensor)
        #     S_avg = sum(cos[0,0,:,0])
        #     cos = torch.flatten(H*cos/S_avg) # .to(device)

        #     pred = torch.flatten(pred)
        #     target = torch.flatten(target)
        #     mse_loss = torch.mean( cos * (pred - target)**2)
        #     return self.const*torch.sqrt(torch.pow(mse_loss, self.powMSE))

        N = pred.shape[0]
        cos = torch.flatten(self.cos.repeat(N, 1, 1, 1))

        pred = torch.flatten(pred)
        target = torch.flatten(target)
        # mse_loss = self.MSE_criterion(pred, target)
        mse_loss = torch.mean( cos * (pred - target)**2)
        return self.const*torch.sqrt(torch.pow(mse_loss, self.powMSE))




class ACCLoss_latitude_weighted(nn.Module):
    def __init__(self, batch_shape_without_padding):
        super().__init__()
        """
        Če je climatology = target, potem je rezultat nan.
        Takšne spremenljivke je treba zanemariti 
        (npr. dan v letu = day_of_the_year)
        """

        # # batch_shape_without_padding = (N,C,H,W)         
        # # Tensor of the same shape as batch without padding
        # # phi_tensor = torch.full(batch_shape_without_padding, 0)
        # phi_tensor = torch.full(batch_shape_without_padding, 0.0, dtype=torch.float32)

        # # Fill tensor with cos(phi)
        # N = batch_shape_without_padding[0]
        # C = batch_shape_without_padding[1]
        # H = batch_shape_without_padding[2]
        # W = batch_shape_without_padding[3]
        # d_phi = torch.pi/H
        # if H == 60:
        #     phi_0 = -88.5*torch.pi/180
        # elif H == 180:
        #     phi_0 = -89.5*torch.pi/180
        # else:
        #     raise ValueError(
        #         "This resolution has not jet been implemented.\
        #          Check MSE_Loss_surface - torch_loss_functions.py")
        # for i in range(H):
        #     phi_tensor[:,:,i,:] = torch.full((N,C,W), phi_0 + i*d_phi)

        # # Cosine phi
        # cos = torch.cos(phi_tensor)
        # S_avg = sum(cos[0,0,:,0])
        # cos = H*cos/S_avg
        # self.cos = torch.flatten(cos)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.cos = self.cos.to(device)

        phi_tensor = torch.full(batch_shape_without_padding, 0.0, dtype=torch.float32)
        # Fill tensor with cos(phi)
        # N = batch_shape_without_padding[0]
        C = batch_shape_without_padding[0]
        H = batch_shape_without_padding[1]
        W = batch_shape_without_padding[2]
        d_phi = torch.pi/H
        if H == 60:
            phi_0 = -88.5*torch.pi/180
        elif H == 180:
            phi_0 = -89.5*torch.pi/180
        else:
            raise ValueError(
                "This resolution has not jet been implemented.\
                 Check MSE_Loss_surface - torch_loss_functions.py")
        for i in range(H):
            phi_tensor[:,i,:] = torch.full((C,W), phi_0 + i*d_phi)

        # Cosine phi
        cos = torch.cos(phi_tensor)
        S_avg = sum(cos[0,:,0])
        cos = H*cos/S_avg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cos = cos.to(device)

    def forward(self, pred, target, clima):
        # if len(torch.flatten(pred)) == len(self.cos):
        #     pred = torch.flatten(pred)
        #     target = torch.flatten(target)
        #     clima = torch.flatten(clima)
        #     stevec = torch.mean(self.cos * (pred - clima) * (target - clima))
        #     imenovalec = torch.sqrt(
        #         torch.mean(self.cos * torch.square(pred - clima)) * 
        #         torch.mean(self.cos * torch.square(target - clima)) )
        #     acc_loss = - (stevec / imenovalec - 1)
        #     return acc_loss 
        # else:
        #     # Fill tensor with cos(phi)
        #     N = pred[0]
        #     C = pred[1]
        #     H = pred[2]
        #     W = pred[3]

        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     phi_tensor = torch.full((N,C,H,W), 0.0, dtype=torch.float32, device=device)

        #     d_phi = torch.pi/H
        #     if H == 60:
        #         phi_0 = -88.5*torch.pi/180
        #     elif H == 180:
        #         phi_0 = -89.5*torch.pi/180
        #     else:
        #         raise ValueError(
        #             "This resolution has not jet been implemented.\
        #             Check MSE_Loss_surface - torch_loss_functions.py")
        #     for i in range(H):
        #         phi_tensor[:,:,i,:] = torch.full((N,C,W), phi_0 + i*d_phi)

        #     # Cosine phi
        #     cos = torch.cos(phi_tensor)
        #     S_avg = sum(cos[0,0,:,0])
        #     cos = torch.flatten(H*cos/S_avg) # .to(device)

        #     pred = torch.flatten(pred)
        #     target = torch.flatten(target)
        #     clima = torch.flatten(clima)
        #     stevec = torch.mean(cos * (pred - clima) * (target - clima))
        #     imenovalec = torch.sqrt(
        #         torch.mean(cos * torch.square(pred - clima)) * 
        #         torch.mean(cos * torch.square(target - clima)) ) 
        #     acc_loss = - (stevec / imenovalec - 1)
        #     return acc_loss 

        N = pred.shape[0]
        cos = torch.flatten(self.cos.repeat(N, 1, 1, 1))

        pred = torch.flatten(pred)
        target = torch.flatten(target)
        clima = torch.flatten(clima)
        stevec = torch.mean(cos * (pred - clima) * (target - clima))
        imenovalec = torch.sqrt(
            torch.mean(cos * torch.square(pred - clima)) * 
            torch.mean(cos * torch.square(target - clima)) )
        acc_loss = - (stevec / imenovalec - 1)
        
        return acc_loss 