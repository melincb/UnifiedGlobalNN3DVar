# %% --------------- LIBRARIES -----------------------
import os
import torch

# %% -----------------------------------------------------
#               TORCH STANDARD SCALER
#    -----------------------------------------------------

class TorchStandardScaler:
    """
    Args:
    --> main_scaler_path: path for saving or loading scaler
    --> scaler_name: name for saving or loading scaler
    """

    def __init__(self, main_scaler_path, scaler_name):
        self.main_scaler_path = main_scaler_path
        self.scaler_name = scaler_name
        try:
            scaling_dict = torch.load(
                os.path.join(self.main_scaler_path, self.scaler_name)
            )
            self.mean = scaling_dict['mean']
            self.std = scaling_dict['std']
        except:
            print(
            "This scaler is not fitted. Use TorchStandardScaler.fit.")

    def fit(self, x):
        mean = x.mean(0, keepdim=True)
        std = x.std(0, unbiased=False, keepdim=True)
        dict = {'mean': mean, 'std': std}        
        torch.save(
            dict, 
            os.path.join(self.main_scaler_path, self.scaler_name + ".pt"))

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x

    def inverse_transform(self, x):
        x *= (self.std + 1e-7)
        x += self.mean
        return x


class TorchMinMaxScaler:
    """
    Args:
    --> main_scaler_path: path for saving or loading scaler
    --> scaler_name: name for saving or loading scaler
    """

    def __init__(self, main_scaler_path, scaler_name):
        self.main_scaler_path = main_scaler_path
        self.scaler_name = scaler_name
        try:
            scaling_dict = torch.load(
                os.path.join(self.main_scaler_path, self.scaler_name)
            )
            self.min = scaling_dict['min']
            self.max = scaling_dict['max']
        except:
            print("This scaler is not fitted. Use TorchStandardScaler.fit.")

    def fit(self, x):
        min = x.amin(0, keepdim=True)
        max = x.amax(0, keepdim=True)
        dict = {'min': min, 'max': max}
        torch.save(
            dict, 
            os.path.join(self.main_scaler_path, self.scaler_name + ".pt"))

    def transform(self, x):
        x -= self.min
        x /= (self.max - self.min)
        return x

    def inverse_transform(self, x):
        x *= (self.max - self.min)
        x += self.min
        return x

class TorchStandardScaler_ReplaceZeroStd:
    """
    Args:
    --> main_scaler_path: path for saving or loading scaler
    --> scaler_name: name for saving or loading scaler
    """

    def __init__(self, main_scaler_path, scaler_name):
        self.main_scaler_path = main_scaler_path
        self.scaler_name = scaler_name
        try:
            scaling_dict = torch.load(
                os.path.join(self.main_scaler_path, self.scaler_name)
            )
            self.mean = scaling_dict['mean']
            self.std = scaling_dict['std']
        except:
            print(
            "This scaler is not fitted (or you have the wrong name). Use TorchStandardScaler.fit.")

    def fit(self, x):
        mean = x.mean(0, keepdim=True)
        std = x.std(0, unbiased=False, keepdim=True)
        # print(mean, mean.amin(), mean.amax())
        # print(std, std.amin(), std.amax())
        # mean_mean = torch.mean(mean).item()
        mean_std = torch.mean(std[std>=1e-4]).item()
        # print(mean_std, mean_std)
        # mean[mean==0.] = mean_mean
        # std[std<=1e-4] = mean_std
        std += mean_std
        dict = {'mean': mean, 'std': std}        
        torch.save(
            dict, 
            os.path.join(self.main_scaler_path, self.scaler_name + ".pt"))

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x

    def inverse_transform(self, x):
        x *= (self.std + 1e-7)
        x += self.mean
        return x


class TorchStandardScaler_GeneralStd:
    """
    Args:
    --> main_scaler_path: path for saving or loading scaler
    --> scaler_name: name for saving or loading scaler
    """

    def __init__(self, main_scaler_path, scaler_name):
        self.main_scaler_path = main_scaler_path
        self.scaler_name = scaler_name
        try:
            scaling_dict = torch.load(
                os.path.join(self.main_scaler_path, self.scaler_name)
            )
            self.mean = scaling_dict['mean']
            self.std = scaling_dict['std']
        except:
            print(
            "This scaler is not fitted. Use TorchStandardScaler.fit.")

    def fit(self, x):
        mean = x.mean(0, keepdim=True)
        # std = x.std(0, unbiased=False, keepdim=True)
        std = torch.std(x.flatten(), unbiased=False)
        print("std", std)
        std = torch.zeros(mean.shape) + std 
        # print(mean, mean.amin(), mean.amax())
        # print(std, std.amin(), std.amax())
        # mean_mean = torch.mean(mean).item()
        # mean_std = torch.mean(std[std>=1e-4]).item()
        # print(mean_std, mean_std)
        # mean[mean==0.] = mean_mean
        # std[std<=1e-4] = mean_std
        # std += mean_std
        dict = {'mean': mean, 'std': std}        
        torch.save(
            dict, 
            os.path.join(self.main_scaler_path, self.scaler_name + ".pt"))

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x

    def inverse_transform(self, x):
        x *= (self.std + 1e-7)
        x += self.mean
        return x
