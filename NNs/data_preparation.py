# %% ------------------------------------
#       IMPORT LIBRARIES
# ---------------------------------------

import os
import netCDF4
import numpy as np

#    ===================================== 
#           SAVE DATA INDIVIDUALY
#    =====================================

class DataWriter():
    """
    Get netCDF from Copernicus Climate Data Store and save each
    X(t) individually as it's own file.
    
    Creates a list with samples X(t) and corespodning
    labels y(t)=X(t+k \Delta t).
    
    ----------------------------------------------------
    Args
    ----------------------------------------------------
    root_load: (str)        - main path to directory of netCDF file
                              we wish to load
    root_save: (str)        - main path to directory for saving files
    root_annotations: (str) - main path to directory for saving annotations
    filename: (str)         - name of file we wish to load and save
                              e.g. "geopotential"
    ----------------------------------------------------
    Methods
    ----------------------------------------------------
    --> print_netCDF_file_specs
    --> save_individual_files
    --> save_annotations_file
    """
    
    def __init__(self, root_load, root_save, 
                 root_annotations, fileName):
        self.root_load = root_load               # "/.../main_load_folder"
        self.root_save = root_save               # "/.../main_save_folder"
        self.root_annotations = root_annotations # "/.../folder with annotations file"
        self.fileName = fileName                 # e.g. "geopotential" 

    def __repr__(self):
        return "DataWriter('{}', '{}', '{}', '{}')".format(
            self.root_load, self.root_save, self.root_annotations, self.fileName)

    def print_netCDF_file_specs(self, year, fieldName=False):
        filename = self.fileName + '_' + str(year) + '.nc'
        path = os.path.join(self.root_load, filename)
        file = netCDF4.Dataset(path)
        print(f"{filename} specs:")
        print("-----------------------------------------")
        print(f"{file}")
        print("-----------------------------------------\n")
        print(f"Variables:\n{file.variables}")
        if fieldName:
            print(f"{filename} field specs:")
            print("-----------------------------------------")
            print(f"{file[fieldName]}")
            print("-----------------------------------------\n")


    def save_individual_files(self, year, fieldName, fileName,
                        resolution_degrees=1, save=True,
                        flip_and_roll = False,
                        daily_timesteps = True):
        """
        Method which opens netCDF file of specific year
        and saves each time as .npy file in root_save/fieldName/
        directory.
        It reduces resolution by keepoing every resolution_degrees.
        daily_timesteps = True --> currently only implemented for either daily timestapes of 12 hr timestapes
        """

        # Import interpolation algorithm
        from scipy.interpolate import RectBivariateSpline

        # Define current coordinates
        ln_c = np.arange(-180,180,1)
        lt_c = np.arange(-90,90.1,1)

        # Define final coordinates (for interpolation evaluation)
        ln_f = np.arange(-180,180, resolution_degrees)

        if resolution_degrees == 1:
            lt_f = np.arange(-89.5,89.6,resolution_degrees)
        elif resolution_degrees == 3:
            lt_f = np.arange(-88.5,88.6,resolution_degrees)
        elif resolution_degrees == 6:
            lt_f = np.arange(-87.,87.1,resolution_degrees)
        else:
            raise ValueError("Check what range your resolution_degrees needs")

        # Open netCDF file and field
        filename = self.fileName + '_' + str(year) + '.nc'
        path = os.path.join(self.root_load, filename)
        # print(netCDF4.Dataset(path))
        field = netCDF4.Dataset(path)[fieldName] # (time, lat=181, lon=36)

        assert np.isnan(np.array(field)).sum() == 0, "Contains 'nan', consider using method 'save_individual_files_nan_values'."

        # Open folder for saving the fields
        path_to_year = os.path.join(self.root_save,
            self.fileName + '_' + str(year))
        try:
            os.mkdir(path_to_year)
        except:
            print(f"Directory \"{path_to_year}\" already exists")    

        list_of_paths = []

        # Import datetime
        from datetime import datetime
    
        # Save fields
        day_num = 1
        hour_num = 0
        main_path_to_field = os.path.join(path_to_year, fileName)
        for t in range(field.shape[0]):
            # # initializing day number
            # day_num = str(t+1)
                        
            # # adjusting day num
            # day_num.rjust(3 + len(day_num), '0')
            
            # # converting to date
            # date = datetime.strptime(str(year) + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")

            if daily_timesteps:
                day_num = str(t+1)
                # adjusting day num
                day_num.rjust(3 + len(day_num), '0')
                # converting to date
                date = datetime.strptime(str(year) + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")
            else:
                if hour_num == 24:
                    day_num += 1
                    hour_num = 0
                day_num_str = str(day_num)
                hour_num_str = str(hour_num)
                # adjusting day num
                # day_num_str=day_num_str.rjust(3 + len(day_num_str), '0')
                hour_num_str = hour_num_str.rjust(2, '0')
                # converting to date
                date = datetime.strptime(str(year) + "-" + day_num_str, "%Y-%j").strftime("%Y-%m-%d")
                date = str(date) + "-" + hour_num_str          

            whole_path = main_path_to_field + "_" + str(date)
            
            if save:
                # Interpolate field at lat [-89.5,-88.5,...,+89.5] 
                # to get rid of two whole lines of useles data at poles.
                # Use bicubic spline because of stability and best
                # overall interpolation
                # (https://mmas.github.io/interpolation-scipy)
                if flip_and_roll:
                    print("First define what are the necesary coordinates")
                    assert False, "First define what are the necesary coordinates"
                    # interp_fn = RectBivariateSpline(lt_c, ln_c, np.roll(np.flip(field[t],0),720,1))
                else:
                    interp_fn = RectBivariateSpline(lt_c, ln_c, field[t])
                interp_field = interp_fn.__call__(lt_f, ln_f)
                # Change dtype to float32 because model params
                # are float32 and they have to mathc.
                interp_field = interp_field.astype('float32')
                # Save field
                np.save(whole_path, np.array(interp_field))
            list_of_paths.append(str(whole_path+".npy"))
            # if day_num % 30 == 0 and hour_num == 0:
            #     print(f"year {year}, new month")
            if daily_timesteps == False:
                hour_num += 12

        print(f"Files saved to {path_to_year}.")

        if save == False:
            return list_of_paths

    def save_individual_files_static_fields(
            self, fieldName, fileName,
            resolution_degrees=1, save=True,
            print_variables = True,
            flip_and_roll = True):
        """
        Method which opens netCDF file of specific year
        and saves each time as .npy file in root_save/fieldName/
        directory.
        It reduces resolution by keepoing every resolution_degrees.
        print_variables (bool): print variables inside netCDF file (ends with assertion error)
        """

        # Import interpolation algorithm
        from scipy.interpolate import RectBivariateSpline

        # Define current coordinates
        ln_c = np.arange(-180,180,0.25)
        lt_c = np.arange(-90,90.1,0.25)

        # Define final coordinates (for interpolation evaluation)
        ln_f = np.arange(-180,180, resolution_degrees)

        if resolution_degrees == 1:
            lt_f = np.arange(-89.5,89.6,resolution_degrees)
        elif resolution_degrees == 3:
            lt_f = np.arange(-88.5,88.6,resolution_degrees)
        elif resolution_degrees == 6:
            lt_f = np.arange(-87.,87.1,resolution_degrees)
        else:
            raise ValueError("Check what range your resolution_degrees needs")

        # Open netCDF file and field
        filename = self.fileName + '.nc'
        path = os.path.join(self.root_load, filename)
        ncfile = netCDF4.Dataset(path)
        if print_variables:
            print(f"Variables:\n{netCDF4.Dataset(path).variables}")
        field = ncfile[fieldName] #(time, lat=181, lon=360)

        # Open folder for saving the fields
        path_to_folder = os.path.join(self.root_save,self.fileName)
        try:
            os.mkdir(path_to_folder)
        except:
            print(f"Directory \"{path_to_folder}1\" already exists")    
    
        # Save fields
        main_path_to_field = os.path.join(path_to_folder, fileName)                        

        whole_path = main_path_to_field
        
        list_of_paths = []

        if save:
            # Interpolate field at lat [-89.5,-88.5,...,+89.5] 
            # to get rid of two whole lines of useles data at poles.
            # Use bicubic spline because of stability and best
            # overall interpolation
            # (https://mmas.github.io/interpolation-scipy)
            if flip_and_roll:
                interp_fn = RectBivariateSpline(lt_c, ln_c, np.roll(np.flip(field[0],0),720,1))
            else:
                interp_fn = RectBivariateSpline(lt_c, ln_c, field[0])
            interp_field = interp_fn.__call__(lt_f, ln_f)
            # Change dtype to float32 because model params
            # are float32 and they have to mathc.
            interp_field = interp_field.astype('float32')

            # Save field
            np.save(whole_path, np.array(interp_field))
        list_of_paths.append(str(whole_path+".npy"))

        print(f"Files saved to {path_to_folder}.")

        if save == False:
            return list_of_paths

    def save_individual_files_tau_300_700(
            self, year, root_load_Z700, root_load_Z300,
            fileName_Z700, fileName_Z300, fileName_tau_300_700,
            fieldName, resolution_degrees=1, save=True, 
            print_variables=False,
            daily_timesteps = True):
        """
        Method which opens netCDF file of specific year
        and saves each time as .npy file in root_save/fieldName/
        directory.
        It reduces resolution by keepoing every resolution_degrees.
        daily_timesteps = True --> currently only implemented for either daily timestapes of 12 hr timestapes
        """

        # Import interpolation algorithm
        from scipy.interpolate import RectBivariateSpline

        # Define current coordinates
        ln_c = np.arange(-180,180,1)
        lt_c = np.arange(-90,90.1,1)

        # Define final coordinates (for interpolation evaluation)
        ln_f = np.arange(-180,180, resolution_degrees)

        if resolution_degrees == 1:
            lt_f = np.arange(-89.5,89.6,resolution_degrees)
        elif resolution_degrees == 3:
            lt_f = np.arange(-88.5,88.6,resolution_degrees)
        elif resolution_degrees == 6:
            lt_f = np.arange(-87.,87.1,resolution_degrees)
        else:
            raise ValueError("Check what range your resolution_degrees needs")

        # Open netCDF file and field
        filename_Z700 = fileName_Z700 + '_' + str(year) + '.nc'
        filename_Z300 = fileName_Z300 + '_' + str(year) + '.nc'
        path_Z700 = os.path.join(root_load_Z700, filename_Z700)
        path_Z300 = os.path.join(root_load_Z300, filename_Z300)

        if print_variables:
            print(netCDF4.Dataset(path_Z700).variables)
            print(netCDF4.Dataset(path_Z300).variables)

        field_Z700 = netCDF4.Dataset(path_Z700)[fieldName] #(time, lat=181, lon=36)
        field_Z300 = netCDF4.Dataset(path_Z300)[fieldName] #(time, lat=181, lon=36)
        

        # Open folder for saving the fields
        path_to_year = os.path.join(self.root_save,
            fileName_tau_300_700 + '_' + str(year))
        try:
            os.mkdir(path_to_year)
        except:
            print(f"Directory \"{path_to_year}\" already exists")    

        list_of_paths = []

        # Import datetime
        from datetime import datetime
    
        # Save fields
        day_num = 1
        hour_num = 0
        main_path_to_field = os.path.join(
            path_to_year, fileName_tau_300_700)
            
        for t in range(field_Z300.shape[0]):
            # # initializing day number
            # day_num = str(t+1)
                        
            # # adjusting day num
            # day_num.rjust(3 + len(day_num), '0')
            
            # # converting to date
            # date = datetime.strptime(str(year) + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")

            if daily_timesteps:
                day_num = str(t+1)
                # adjusting day num
                day_num.rjust(3 + len(day_num), '0')
                # converting to date
                date = datetime.strptime(str(year) + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")
            else:
                if hour_num == 24:
                    day_num += 1
                    hour_num = 0
                day_num_str = str(day_num)
                hour_num_str = str(hour_num)
                # adjusting day num
                # day_num_str=day_num_str.rjust(3 + len(day_num_str), '0')
                hour_num_str = hour_num_str.rjust(2, '0')
                # converting to date
                date = datetime.strptime(str(year) + "-" + day_num_str, "%Y-%j").strftime("%Y-%m-%d")
                date = str(date) + "-" + hour_num_str          

            whole_path = main_path_to_field + "_" + str(date)
            
            if save:
                # Interpolate field at lat [-89.5,-88.5,...,+89.5] 
                # to get rid of two whole lines of useles data at poles.
                # Use bicubic spline because of stability and best
                # overall interpolation
                # (https://mmas.github.io/interpolation-scipy)
                interp_fn_Z700 = RectBivariateSpline(lt_c, ln_c, field_Z700[t])
                interp_fn_Z300 = RectBivariateSpline(lt_c, ln_c, field_Z300[t])
                interp_field_Z700 = interp_fn_Z700.__call__(lt_f, ln_f)
                interp_field_Z300 = interp_fn_Z300.__call__(lt_f, ln_f)
                # Change dtype to float32 because model params
                # are float32 and they have to mathc.
                interp_field_Z700 = interp_field_Z700.astype('float32')
                interp_field_Z300 = interp_field_Z300.astype('float32')
                # Save field
                np.save(whole_path, np.array(interp_field_Z300)-np.array(interp_field_Z700))
            list_of_paths.append(str(whole_path+".npy"))
            if daily_timesteps == False:
                hour_num += 12

        print(f"Files saved to {path_to_year}.")

        if save == False:
            return list_of_paths

    def save_individual_files_nan_values(self, year, fieldName, fileName,
                        resolution_degrees=1, save=True,
                        flip_and_roll = False, replace_nan_with=-1,
                        daily_timesteps = True):
        """
        Method which opens netCDF file of specific year
        and saves each time as .npy file in root_save/fieldName/
        directory.
        It reduces resolution by keepoing every resolution_degrees.
        # Example: Sea_Ice_Cover
        daily_timesteps = True --> currently only implemented for either daily timestapes of 12 hr timestapes
        """

        # Import interpolation algorithm
        from scipy.interpolate import RectBivariateSpline

        # Define current coordinates
        ln_c = np.arange(-180,180,1)
        lt_c = np.arange(-90,90.1,1)

        # Define final coordinates (for interpolation evaluation)
        ln_f = np.arange(-180,180, resolution_degrees)

        if resolution_degrees == 1:
            lt_f = np.arange(-89.5,89.6,resolution_degrees)
        elif resolution_degrees == 3:
            lt_f = np.arange(-88.5,88.6,resolution_degrees)
        elif resolution_degrees == 6:
            lt_f = np.arange(-87.,87.1,resolution_degrees)
        else:
            raise ValueError("Check what range your resolution_degrees needs")

        # Open netCDF file and field
        filename = self.fileName + '_' + str(year) + '.nc'
        path = os.path.join(self.root_load, filename)
        # print(netCDF4.Dataset(path))
        field = netCDF4.Dataset(path)[fieldName] #(time, lat=181, lon=36)

        field = np.nan_to_num(np.array(field), copy=False, nan=replace_nan_with)

        # Open folder for saving the fields
        path_to_year = os.path.join(self.root_save,
            self.fileName + '_' + str(year))
        try:
            os.mkdir(path_to_year)
        except:
            print(f"Directory \"{path_to_year}\" already exists")    

        list_of_paths = []

        # Import datetime
        from datetime import datetime
    
        # Save fields
        main_path_to_field = os.path.join(path_to_year, fileName)
        for t in range(field.shape[0]):
            # # initializing day number
            # day_num = str(t+1)
                        
            # # adjusting day num
            # day_num.rjust(3 + len(day_num), '0')
            
            # # converting to date
            # date = datetime.strptime(str(year) + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")

            if daily_timesteps:
                day_num = str(t+1)
                # adjusting day num
                day_num.rjust(3 + len(day_num), '0')
                # converting to date
                date = datetime.strptime(str(year) + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")
            else:
                if hour_num == 24:
                    day_num += 1
                    hour_num = 0
                day_num_str = str(day_num)
                hour_num_str = str(hour_num)
                # adjusting day num
                # day_num_str=day_num_str.rjust(3 + len(day_num_str), '0')
                hour_num_str = hour_num_str.rjust(2, '0')
                # converting to date
                date = datetime.strptime(str(year) + "-" + day_num_str, "%Y-%j").strftime("%Y-%m-%d")
                date = str(date) + "-" + hour_num_str          

            whole_path = main_path_to_field + "_" + str(date)
            
            if save:
                # Interpolate field at lat [-89.5,-88.5,...,+89.5] 
                # to get rid of two whole lines of useles data at poles.
                # Use bicubic spline because of stability and best
                # overall interpolation
                # (https://mmas.github.io/interpolation-scipy)
                if flip_and_roll:
                    print("First define what are the necesary coordinates")
                    assert False, "First define what are the necesary coordinates"
                    # interp_fn = RectBivariateSpline(lt_c, ln_c, np.roll(np.flip(field[t],0),720,1))
                else:
                    interp_fn = RectBivariateSpline(lt_c, ln_c, field[t])
                interp_field = interp_fn.__call__(lt_f, ln_f)
                # Change dtype to float32 because model params
                # are float32 and they have to mathc.
                interp_field = interp_field.astype('float32')
                # Save field
                np.save(whole_path, np.array(interp_field))
            list_of_paths.append(str(whole_path+".npy"))
            if daily_timesteps == False:
                hour_num += 12

        print(f"Files saved to {path_to_year}.")

        if save == False:
            return list_of_paths


    def save_individual_files_day_of_the_year(self, year,
                        fieldName, folderName, fileNameForSaving,
                        resolution_degrees=1, save=True, flip_and_roll=False,
                        daily_timesteps = True):
        """
        Method which opens netCDF file of specific year
        and saves each time as .npy file in root_save/fieldName/
        directory.
        It reduces resolution by keepoing every resolution_degrees.
        daily_timesteps = True --> currently only implemented for either daily timestapes of 12 hr timestapes
        """

        # Import interpolation algorithm
        from scipy.interpolate import RectBivariateSpline

        # Define current coordinates
        ln_c = np.arange(-180,180,1)
        lt_c = np.arange(-90,90.1,1)

        # Define final coordinates (for interpolation evaluation)
        ln_f = np.arange(-180,180, resolution_degrees)

        if resolution_degrees == 1:
            lt_f = np.arange(-89.5,89.6,resolution_degrees)
        elif resolution_degrees == 3:
            lt_f = np.arange(-88.5,88.6,resolution_degrees)
        elif resolution_degrees == 6:
            lt_f = np.arange(-87.,87.1,resolution_degrees)
        else:
            raise ValueError("Check what range your resolution_degrees needs")

        # Open netCDF file and field
        filename = self.fileName + '_' + str(year) + '.nc'
        path = os.path.join(self.root_load, filename)
        # print(netCDF4.Dataset(path))
        field = netCDF4.Dataset(path)[fieldName] #(time, lat=181, lon=36)

        assert np.isnan(np.array(field)).sum() == 0, "Contains 'nan', consider using method 'save_individual_files_nan_values'."

        # Open folder for saving the fields
        path_to_year = os.path.join(self.root_save,
            folderName + '_' + str(year))
        try:
            os.mkdir(path_to_year)
        except:
            print(f"Directory \"{path_to_year}\" already exists")    

        list_of_paths = []

        # Import datetime
        from datetime import datetime
    
        # Save fields
        day_num = 1
        hour_num = 0
        main_path_to_field = os.path.join(path_to_year, fileNameForSaving)
        for t in range(field.shape[0]):
            # initializing day number
            if daily_timesteps:
                day_num = str(t+1)
                # adjusting day num
                day_num.rjust(3 + len(day_num), '0')
                # converting to date
                date = datetime.strptime(str(year) + "-" + day_num, "%Y-%j").strftime("%Y-%m-%d")
            else:
                if hour_num == 24:
                    day_num += 1
                    hour_num = 0
                day_num_str = str(day_num)
                hour_num_str = str(hour_num)
                # adjusting day num
                # day_num_str=day_num_str.rjust(3 + len(day_num_str), '0')
                hour_num_str = hour_num_str.rjust(2, '0')
                # converting to date
                date = datetime.strptime(str(year) + "-" + day_num_str, "%Y-%j").strftime("%Y-%m-%d")
                date = str(date) + "-" + hour_num_str          

            whole_path = main_path_to_field + "_" + str(date)
            
            if save:
                # Interpolate field at lat [-89.5,-88.5,...,+89.5] 
                # to get rid of two whole lines of useles data at poles.
                # Use bicubic spline because of stability and best
                # overall interpolation
                # (https://mmas.github.io/interpolation-scipy)
                if flip_and_roll:
                    print("First define what are the necesary coordinates")
                    assert False, "First define what are the necesary coordinates"
                    # interp_fn = RectBivariateSpline(lt_c, ln_c, np.roll(np.flip(field[t],0),720,1))
                else:
                    interp_fn = RectBivariateSpline(lt_c, ln_c, field[t])

                # Change dtype to float32 because model params
                # are float32 and they have to mathc.
                interp_field = np.full((lt_f.shape[0], ln_f.shape[0]), fill_value=t+1).astype('float32')                
                if daily_timesteps == False:
                    interp_field[::2, ::2] = 0 if (t+1) % 2 == 0 else 1
                # Save field
                np.save(whole_path, np.array(interp_field))
            list_of_paths.append(str(whole_path+".npy"))
            # if day_num % 30 == 0 and hour_num == 0:
            #     print(f"year {year}, new month")
            if daily_timesteps == False:
                hour_num += 12

        print(f"Files saved to {path_to_year}.")

        if save == False:
            return list_of_paths


    def save_annotations_file(self, year, fieldName, fileName,
                              append_global, save_local, 
                              contains_nan=False,
                              daily_timesteps = True):
        """
        Saves two annotation_files:
        ---> if append_global == True:
             fieldName_annotation_file.txt in root_save directory   
        ---> if save_local == True:
             fieldName_year_annotation_file.txt in 
             root_save/fieldName_year/ directory  
        """

        if contains_nan == False:
            list_of_paths = self.save_individual_files(
                year, fieldName, fileName, save=False, daily_timesteps = daily_timesteps
                )
        else:
            list_of_paths = self.save_individual_files_nan_values(
                year, fieldName, fileName, save=False, daily_timesteps = daily_timesteps
                )

        if append_global:
            # Append annotations to the (fields) general annotations_file
            # for all years 
            annotations_file_path_1 = os.path.join(
            self.root_annotations, self.fileName + "_annotation_file.txt"
            )
            annotations_file_path_2 = os.path.join(
            self.root_save, self.fileName + "_annotation_file.txt"
            )

            with open(annotations_file_path_1, "a") as file_handle:
                np.savetxt(file_handle, np.array(list_of_paths), fmt="%s")           
            print(f"Annotations APPENDED to {annotations_file_path_1}.")

            with open(annotations_file_path_2, "a") as file_handle:
                np.savetxt(file_handle, np.array(list_of_paths), fmt="%s")           
            print(f"Annotations APPENDED to {annotations_file_path_2}.")

        if save_local:
            # Create annotations file for files in specific year
            path_to_year = os.path.join(self.root_save,
            self.fileName + '_' + str(year)
            )
            annotations_file_path = os.path.join(
                path_to_year, self.fileName + f"_{year}_annotation_file.txt"
                )
            with open(annotations_file_path, "w") as file_handle:
                np.savetxt(file_handle, np.array(list_of_paths), fmt="%s")
        
            print(f"Annotations saved in {annotations_file_path}.")

    def save_annotations_file_day_of_the_year(
            self, year, fieldName, folderName, fileNameForSaving,
            append_global, save_local, daily_timesteps=True):
        """
        Saves two annotation_files:
        ---> if append_global == True:
             fieldName_annotation_file.txt in root_save directory   
        ---> if save_local == True:
             fieldName_year_annotation_file.txt in 
             root_save/fieldName_year/ directory  
        """

        list_of_paths = self.save_individual_files_day_of_the_year(
            year, fieldName, folderName, 
            fileNameForSaving, save=False, 
            daily_timesteps=daily_timesteps)

        if append_global:
            # Append annotations to the (fields) general annotations_file
            # for all years 
            annotations_file_path_1 = os.path.join(
            self.root_annotations, fileNameForSaving + "_annotation_file.txt"
            )
            annotations_file_path_2 = os.path.join(
            self.root_save, fileNameForSaving + "_annotation_file.txt"
            )

            with open(annotations_file_path_1, "a") as file_handle:
                np.savetxt(file_handle, np.array(list_of_paths), fmt="%s")           
            print(f"Annotations APPENDED to {annotations_file_path_1}.")

            with open(annotations_file_path_2, "a") as file_handle:
                np.savetxt(file_handle, np.array(list_of_paths), fmt="%s")           
            print(f"Annotations APPENDED to {annotations_file_path_2}.")

        if save_local:
            # Create annotations file for files in specific year
            path_to_year = os.path.join(self.root_save,
                fileNameForSaving + '_' + str(year)
                )
            annotations_file_path = os.path.join(
                path_to_year, fileNameForSaving + f"_{year}_annotation_file.txt"
                )
            with open(annotations_file_path, "w") as file_handle:
                np.savetxt(file_handle, np.array(list_of_paths), fmt="%s")
        
            print(f"Annotations saved in {annotations_file_path}.")

    def save_annotations_file_tau_300_700(
            self, year, root_load_Z700, root_load_Z300,
            fileName_Z700, fileName_Z300, fileName_tau_300_700,
            fieldName, append_global, save_local, daily_timesteps=True):
        """
        Saves two annotation_files:
        ---> if append_global == True:
             fieldName_annotation_file.txt in root_save directory   
        ---> if save_local == True:
             fieldName_year_annotation_file.txt in 
             root_save/fieldName_year/ directory  
        """

        list_of_paths = self.save_individual_files_tau_300_700(
            year, root_load_Z700, root_load_Z300,
            fileName_Z700, fileName_Z300, fileName_tau_300_700,
            fieldName, save=False, print_variables=False,
            daily_timesteps=daily_timesteps) #, resolution_degrees=3

        # list_of_paths = self.save_individual_files(
        #     year, fieldName, fileName, save=False
        #     )

        if append_global:
            # Append annotations to the (fields) general annotations_file
            # for all years 
            annotations_file_path_1 = os.path.join(
            self.root_annotations, self.fileName + "_annotation_file.txt"
            )
            annotations_file_path_2 = os.path.join(
            self.root_save, self.fileName + "_annotation_file.txt"
            )

            with open(annotations_file_path_1, "a") as file_handle:
                np.savetxt(file_handle, np.array(list_of_paths), fmt="%s")           
            print(f"Annotations APPENDED to {annotations_file_path_1}.")

            with open(annotations_file_path_2, "a") as file_handle:
                np.savetxt(file_handle, np.array(list_of_paths), fmt="%s")           
            print(f"Annotations APPENDED to {annotations_file_path_2}.")

        if save_local:
            # Create annotations file for files in specific year
            path_to_year = os.path.join(self.root_save,
            self.fileName + '_' + str(year)
            )
            annotations_file_path = os.path.join(
                path_to_year, self.fileName + f"_{year}_annotation_file.txt"
                )
            with open(annotations_file_path, "w") as file_handle:
                np.savetxt(file_handle, np.array(list_of_paths), fmt="%s")
        
            print(f"Annotations saved in {annotations_file_path}.")

    def change_annotation_file_paths(
            self, server_name = "Camelot", server_path = "/mnt/data/perkanu"):

        path = os.path.join(self.root_annotations, self.fileName + "_annotation_file.txt")

        # Load fields annotations (path to dataset fields)
        annotation_file = np.loadtxt(path, dtype=np.str_)

        list_of_new_paths = []

        for i in range(len(annotation_file)):
            line = annotation_file[i]
            new_line = server_path + line[13:]
            list_of_new_paths.append(new_line)

        new_path = os.path.join(
            self.root_annotations, 
            self.fileName + "_annotation_file" + "_" + server_name + ".txt")

        with open(new_path, "w") as file_handle:
            np.savetxt(file_handle, np.array(list_of_new_paths), fmt="%s")

    def change_annotation_file_paths_custom(
            self, server_name = "Camelot2", 
            replace_string = "/home/perkanu/Magistrska_naloga/U-Net/Data_12hr/3x_Reduced_Density/", 
            replace_with = "/scratch/perkanu/Data/Data_3_degrees/Data_12hr/"):
        """
        replace_string and replace_with must have all characters, including the last "/"
        """

        path = os.path.join(self.root_annotations, self.fileName + "_annotation_file.txt")

        # Load fields annotations (path to dataset fields)
        annotation_file = np.loadtxt(path, dtype=np.str_)

        list_of_new_paths = []

        for i in range(len(annotation_file)):
            line = annotation_file[i]
            # new_line = server_path + line[13:]
            new_line = replace_with + line[len(replace_string):]
            list_of_new_paths.append(new_line)

        new_path = os.path.join(
            self.root_annotations, 
            self.fileName + "_annotation_file" + "_" + server_name + ".txt")

        with open(new_path, "w") as file_handle:
            np.savetxt(file_handle, np.array(list_of_new_paths), fmt="%s")


    def change_annotation_file_paths_day_of_the_year(
            self, folderName, server_name = "Camelot", 
            server_path = "/mnt/data/perkanu"):

        path = os.path.join(self.root_annotations, folderName + "_annotation_file.txt")

        # Load fields annotations (path to dataset fields)
        annotation_file = np.loadtxt(path, dtype=np.str_)

        list_of_new_paths = []

        for i in range(len(annotation_file)):
            line = annotation_file[i]
            new_line = server_path + line[13:]
            list_of_new_paths.append(new_line)

        new_path = os.path.join(
            self.root_annotations, 
            folderName + "_annotation_file" + "_" + server_name + ".txt")

        with open(new_path, "w") as file_handle:
            np.savetxt(file_handle, np.array(list_of_new_paths), fmt="%s")

    def save_latitude_field(
            self, save_fileName, resolution_degrees=1):
        """
        Method which saves latitudes 2D ndarray.
        scaler_type in {"None", "StandardScaler", "MinMaxScaler}
        """

        # Import interpolation algorithm
        from scipy.interpolate import RectBivariateSpline

        # Define final coordinates (for interpolation evaluation)
        ln_f = np.arange(-180,180, resolution_degrees)

        if resolution_degrees == 1:
            lt_f = np.arange(-89.5,89.6,resolution_degrees)
        elif resolution_degrees == 3:
            lt_f = np.arange(-88.5,88.6,resolution_degrees)
        elif resolution_degrees == 6:
            lt_f = np.arange(-87.,87.1,resolution_degrees)
        else:
            raise ValueError("Check what range your resolution_degrees needs")

        # Create latitude field
        lat = np.zeros((len(lt_f), len(ln_f)))

        for i in range(lat.shape[0]):
            for j in range(lat.shape[1]):
                lat[i,j] = lt_f[i]
        
        # Open folder for saving the field
        path_to_folder = os.path.join(self.root_save,
            save_fileName)
        
        np.save(path_to_folder, lat.astype('float32')/60.)

        print(f"Files saved to {path_to_folder}.")

    @staticmethod
    def save_static_field(
            path_load, path_save, fieldName,
            resolution_degrees=1, print_variables = True,
            shift=0):
        """
        Method which opens netCDF file of specific year
        and saves each time as .npy file in root_save/fieldName/
        directory.
        It reduces resolution by keepoing every resolution_degrees.
        print_variables (bool): print variables inside netCDF file (ends with assertion error)
        shift: when saving I add shift value to all matrix elements (in orography I shift field by -2.5)
        """

        # Import interpolation algorithm
        from scipy.interpolate import RectBivariateSpline

        # Define current coordinates
        ln_c = np.arange(-180,180,1)
        lt_c = np.arange(-90,90.1,1)

        # Define final coordinates (for interpolation evaluation)
        ln_f = np.arange(-180,180, resolution_degrees)

        if resolution_degrees == 1:
            lt_f = np.arange(-89.5,89.6,resolution_degrees)
        elif resolution_degrees == 3:
            lt_f = np.arange(-88.5,88.6,resolution_degrees)
        elif resolution_degrees == 6:
            lt_f = np.arange(-87.,87.1,resolution_degrees)
        else:
            raise ValueError("Check what range your resolution_degrees needs")

        # Open netCDF file and field
        ncfile = netCDF4.Dataset(path_load)
        if print_variables:
            print(f"Variables:\n{netCDF4.Dataset(path_load).variables}")
        field = ncfile[fieldName] #(time, lat=181, lon=36)

        field = np.array(field[0])
        average = np.average(field)
        std = np.std(field)
        print(f"average = {average}, std = {std}")

        whole_path = path_save
        
        # Interpolate field at lat [-89.5,-88.5,...,+89.5] 
        # to get rid of two whole lines of useles data at poles.
        # Use bicubic spline because of stability and best
        # overall interpolation
        # (https://mmas.github.io/interpolation-scipy)
        interp_fn = RectBivariateSpline(lt_c, ln_c, (field-average)/std)
        interp_field = interp_fn.__call__(lt_f, ln_f)
        # Change dtype to float32 because model params
        # are float32 and they have to mathc.
        interp_field = interp_field.astype('float32')

        # Save field
        np.save(whole_path, np.array(interp_field)+shift)


#--------------------------------------------------------------
# Example
if __name__ == "__main__":
    # GEOPOTENTIAL Z500
    if False:
        # Create instance for geopotential_500
        geopotentialWrite = DataWriter(
        "/home/perkanu/Magistrska_naloga/U-Net/Data/Geopotential_500_netCDF",
        "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density/Geopotential_500",
        "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density",
        "geopotential"
        )

        # Print specifications of netCDF files
        geopotentialWrite.print_netCDF_file_specs(2021, "lat")
        geopotentialWrite.print_netCDF_file_specs(2021, "geopotential")

        # Convert and save data in correct form for dataloader.
        year = 2021
        # geopotentialWrite.save_individual_files(year, 'geopotential', density_reduction_number=3)
        # geopotentialWrite.save_annotations_file(year, 'geopotential', True, True)
    
    # LATITUDES
    if True:
        # Create instance for geopotential_500
        latitudesWrite = DataWriter(
            "/home/perkanu/Magistrska_naloga/U-Net/Data/Geopotential_500_netCDF - nima veze",
            "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density/Static_Fields",
            "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density - nima veze",
            "geopotential - nima veze"
        )

        # Convert and save data in correct form for dataloader.
        
        latitudesWrite.save_latitude_field(
            "latitudes", 
            resolution_degrees=3)

    # STATIC FIELDS
    if True:
       # Create instance for geopotential_500
        StaticWrite = DataWriter(
            "/home/perkanu/Magistrska_naloga/U-Net/Data/Geopotential_500_netCDF - nima veze",
            "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density/Static_Fields - nima veze",
            "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density - nima veze",
            "geopotential - nima veze"
        )

        StaticWrite.save_static_field(
            path_load="/home/perkanu/Magistrska_naloga/Data_netCDF/Geopotential_Surface_netCDF/geopotential_surface_2022.nc",
            path_save="/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density/Static_Fields/surface_topography",
            fieldName="z",
            resolution_degrees=3,
            print_variables = True,
            shift=-2.5
        )

        StaticWrite.save_static_field(
            path_load="/home/perkanu/Magistrska_naloga/Data_netCDF/LandSeaMask_netCDF/land_sea_mask_2022.nc",
            path_save="/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density/Static_Fields/land_sea_mask",
            fieldName="lsm",
            resolution_degrees=3,
            print_variables = True,
            shift=0.
        )


#--------------------------------------------------------------

#%% =================================================
#           DATE TO INDEX CONVERSION
# ===================================================

from datetime import date, timedelta


#class DateToConsecutiveDay():
class DateToConsecutiveDay():
        """
        Class for converting date to consecutive day or consecutive day
        to date.
        ----------------------------------------------------
        Methods
        ----------------------------------------------------
        --> convert_consecutive_day_to_date
        --> convert_date_to_consecutive_day
        --> num_of_samples
        --> index_to_date
        """

        def __init__(self):
            pass

        @staticmethod
        def convert_consecutive_day_to_date(consecutive_day):
            """
            Args:
            --> consecutive_day (int): Consecutive day in year
                                    [1,2,...,365] 
            """
            # Year is chosen not to have 29.2
            year=2022

            # Transform consecutive_day and year to strings
            consecutive_day = str(consecutive_day)
            year = str(year)

            # adjusting day num
            consecutive_day.rjust(3 + len(consecutive_day), '0')
            
            from datetime import datetime

            # converting to date
            res = datetime.strptime(year + "-" + consecutive_day,
                                    "%Y-%j").strftime("%Y-%m-%d")

            # Get date
            month = int(res.split("-")[1])
            day = int(res.split("-")[2])

            return day, month

        @staticmethod
        def convert_date_to_consecutive_day(date, delta_t = 1):
            """
            Args:
            --> date (tuple): (day, month)
            --> delta_t (float): if you don't have days but e.g.
                                 dt = 12 h => delta_t = 0.5
            Returns:
            --> consecutive_day (1, 2, ..., 365) 
            """
            day, month = date

            from datetime import date

            # year is chosen not to have 29.2.
            day_beginning = date(2022, 1, 1)
            day_index = date(2022, month, day)

            index = (day_index - day_beginning).days * int(1/delta_t)

            jan_1 = 0

            # +1 because jan_1 has consecutive_day = 1
            consecutive_day = index - jan_1 + 1 

            return consecutive_day

        @staticmethod
        def num_of_samples(year=2021, month=12, day=31, delta_t=1,
                            start_year=1950, start_month=1, start_day=1):
            """
            Returns index of field at day.month.year at 0 hr in 
            annotation_file. It assumes that you are using dataset
            with start date 1950 so 01.01.1950 at 0 hr is index 0.
            Indexes are correct (end index is included).
            """
            from datetime import date

            day_beginning = date(start_year, start_month, start_day)
            day_index = date(year, month, day)

            index = (day_index - day_beginning).days * int(1/delta_t)

            return index

        @staticmethod
        def num_of_samples_to_date(
            index, delta_t=1,
            start_year=1950, 
            start_month=1, 
            start_day=1
            ):
            """
            Returns day.month.year from index in 
            annotation_file. It assumes that you are using dataset
            with start date 1950 so 01.01.1950 at 0 hr is index 0.
            Indexes are correct (end index is included).
            """
            from datetime import date, timedelta

            day_beginning = date(start_year, start_month, start_day)

            day_index = day_beginning + timedelta(int(index*delta_t))

            year = int(day_index.strftime("%Y-%m-%d").split("-")[0])
            month = int(day_index.strftime("%Y-%m-%d").split("-")[1])
            day = int(day_index.strftime("%Y-%m-%d").split("-")[2])

            day = (day, month, year)

            return day

        @staticmethod
        def index_to_date(index, delta_t=1, start_year=1950, start_month=1, start_day=1):
            """
            Returns day.month.year at 0 hr in annotation_file
            from index. It assumes that you are using dataset
            with start date 1950 so 01.01.1950 at 0 hr is index 0.
            Indexes are correct (end index is included).
            """
            day_beginning = date(start_year, start_month, start_day)

            day_index = (timedelta(index*int(delta_t)) + day_beginning)

            return day_index

# Show it works
if __name__ == "__main__":
    date_to_day = DateToConsecutiveDay()
    date1 = date_to_day.convert_consecutive_day_to_date(consecutive_day=1)
    date2 = date_to_day.convert_consecutive_day_to_date(consecutive_day=365)
    print(date1, date2)
    cons_day1 = date_to_day.convert_date_to_consecutive_day(date1)
    cons_day2 = date_to_day.convert_date_to_consecutive_day(date2)
    print(cons_day1, cons_day2)

#%% =================================================
#               CALCULATE CLIMATOLOGY
# ===================================================

class CalculateClimatology():
    """
    Class that calculates climatology.   
    ----------------------------------------------------
    Args
    ----------------------------------------------------
    root_annotations: (str) - main path to directory for saving annotations
    annotations_name: (str) - name of annotations file
    root_save_clima: (str)  - maint path to directory for saving
                              and loading daily climatology fields
    ----------------------------------------------------
    Methods
    ----------------------------------------------------
    --> calculate_and_save
    --> load_day_climatology_from_consecutive_day
    --> load_day_climatology_from_date
    --> clima_time_series_from_date
    --> clima_time_series_from_consecutive_day
    """

    def __init__(self, root_annotations, annotations_name, root_save_clima):
        self.root = root_annotations
        self.name = annotations_name
        self.root_save_load = root_save_clima

    def calculate_and_save(self, start_year, end_year):
        """
        Args:
            start_year (int): - start year for calculating climatology 
                                (not first posible - 1950)
            end_year (int):   - end year (included) for calculating climatology
                                (not last posible - 2021)
        """
        assert start_year != 1950, "Can't choose first year"
        assert end_year != 2022, "Can't choose last year"

        # --> Load fields annotations (path to dataset fields)
        annotation_file_geopotencital = np.loadtxt(
            os.path.join(self.root, self.name), dtype=np.str_)            

        date_to_day = DateToConsecutiveDay()

        for consecutive_day in range(1, 366):
            print(f"Day {consecutive_day} / 365")
            # Averages for all years for given consecutive day
            for hr in [0, 12]:
                year_averages = []
                for year in range(start_year, end_year+1):
                    
                    day, month = date_to_day.convert_consecutive_day_to_date(
                        consecutive_day=consecutive_day)
                    
                    # I don't subtract index, because no data is missing.
                    index0 = self.date_to_day.num_of_samples(year, month, day, 1) 
                    path0 = annotation_file_geopotencital[index0]
                    field0 = np.load(path0)
                    
                    # For smoothing use 5 days before and after as well
                    # in climatology calculation.
                    Fields = [field0]
                    #for t in range(1, 2):
                    for t in range(1, 6):
                        # Days before
                        index_minus_t = index0 - t
                        path_minus_t = annotation_file_geopotencital[
                            index_minus_t]
                        field_minus_t = np.load(path_minus_t)
                        Fields.append(field_minus_t)
                        # Days after
                        index_plus_t = index0 + t
                        path_plus_t = annotation_file_geopotencital[
                            index_plus_t]
                        field_plus_t = np.load(path_plus_t)
                        Fields.append(field_plus_t)

                    # Average for specific year
                    one_year_avg = np.mean(Fields, axis=0)
                    year_averages.append(one_year_avg)

                consecutive_day_climatology = np.mean(year_averages, axis=0)

                np.save(
                    os.path.join(
                    self.root_save_load, "climatology" + "_" + str(consecutive_day)),
                    consecutive_day_climatology
                    )

    def load_day_climatology_from_consecutive_day(
        self, consecutive_day):
        """
        Args:
        ---> consecutive_day (int): 1,2,...,365
        """
        path = os.path.join(
            self.root_save_load, "climatology" + "_"
            + str(consecutive_day) + ".npy")

        day_clima = np.load(path)
        return day_clima

    def load_day_climatology_from_date(
        self, date):
        """
        Args:
        ---> date (tuple): (day, month)
        """

        date_to_day = DateToConsecutiveDay()

        consecutive_day = date_to_day.convert_date_to_consecutive_day(
            date, delta_t=1)

        path = os.path.join(
            self.root_save_load, "climatology" + "_"
            + str(consecutive_day) + ".npy")

        day_clima = np.load(path)
        
        return day_clima

    def clima_time_series_from_date(self, start_date,
                num_of_prediction_steps, averaging_sequence_len):
        """
        Args:
        --> start_date (tuple): (day, month)
        --> num_of_prediction_steps (int):
                number of consecutive climatologyies
        Returns:
        --> climas = [clima1, clima2, ..., clima_num_of_pred_steps]
        """

        date_to_day = DateToConsecutiveDay()

        consecutive_day = date_to_day.convert_date_to_consecutive_day(
            start_date, delta_t=1)

        # # List with climatologies
        # climas = [] 
        
        # for i in range(num_of_prediction_steps+1):
        #     day_clima = self.load_day_climatology_from_consecutive_day(
        #         consecutive_day)
        #     climas.append(day_clima)
        #     consecutive_day += 1
        #     # Će pridemo čez 1. januar
        #     if consecutive_day > 365:
        #         consecutive_day = 1

        # List with climatologies
        climas = [] 

        for i in range(num_of_prediction_steps+1):
            intermediate_day_clima = []
            for j in range(averaging_sequence_len):
                day_clima = self.load_day_climatology_from_consecutive_day(
                    consecutive_day)
                intermediate_day_clima.append(day_clima)
                consecutive_day += 1
                # Će pridemo čez 1. januar
                if consecutive_day > 365:
                    consecutive_day = 1
            climas.append(np.average(np.array(intermediate_day_clima), axis=0))

        return climas

    def clima_time_series_from_consecutive_day(
            self, start_day, num_of_prediction_steps,
            averaging_sequence_len):
        """
        Args:
        --> start_day: start consecutive day in year
        --> num_of_prediction_steps (int):
                number of consecutive climatologyies
        Returns:
        --> climas = [clima1, clima2, ..., clima_num_of_pred_steps]
        """
        # # List with climatologies
        # climas = [] 
        
        # consecutive_day = start_day

        # for i in range(num_of_prediction_steps+1):
        #     day_clima = self.load_day_climatology_from_consecutive_day(
        #         consecutive_day)
        #     climas.append(day_clima)
        #     consecutive_day += 1
        #     # Će pridemo čez 1. januar
        #     if consecutive_day > 365:
        #         consecutive_day = 1

        # List with climatologies
        
        climas = [] 
        
        consecutive_day = start_day

        for i in range(num_of_prediction_steps+1):
            intermediate_day_clima = []
            for j in range(averaging_sequence_len):
                day_clima = self.load_day_climatology_from_consecutive_day(
                    consecutive_day)
                intermediate_day_clima.append(day_clima)
                consecutive_day += 1
                # Će pridemo čez 1. januar
                if consecutive_day > 365:
                    consecutive_day = 1
            climas.append(np.average(np.array(intermediate_day_clima), axis=0))

        return climas

# Example how to run it
if __name__ == "__main__":
    clima = CalculateClimatology(
        "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density",
        "geopotential_annotation_file.txt",
        "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density/Geopotential_500")

    # clima.calculate_and_save(1951, 2020)

    day1 = clima.load_day_climatology_from_consecutive_day(5)

    import matplotlib.pyplot as plt
    plt.imshow(day1)
    plt.show()

    #clima.calculate_and_save()
    day2 = clima.load_day_climatology_from_date((5, 1))

    plt.imshow(day2)
    plt.show()

    plt.imshow(day2-day1)
    plt.show()


#%% =================================================
#           CALCULATE NATURAL VARIABILITY
# ===================================================

class CalculateNaturalVariability():
    """
    Class that calculates natural variability.   
    ----------------------------------------------------
    Args
    ----------------------------------------------------
    root_annotations: (str) - main path to directory for saving annotations
    annotations_name: (str) - name of annotations file
    root_save_clima: (str)  - maint path to directory for saving
                              and loading daily natural variability fields
    ----------------------------------------------------
    Methods
    ----------------------------------------------------
    --> calculate_and_save
    --> load_day_natural_variability_from_consecutive_day
    --> load_day_natural_variability_from_date
    --> natural_variability_time_series_from_date
    --> natural_variability_time_series_from_consecutive_day
    """

    def __init__(self, root_annotations, annotations_name, root_save_natural_variability):
        self.root = root_annotations
        self.name = annotations_name
        self.root_save_load = root_save_natural_variability

    def calculate_and_save(self, start_year, end_year):
        """
        Args:
            start_year (int): - start year for calculating climatology 
                                (not first posible - 1950)
            end_year (int):   - end year (included) for calculating climatology
                                (not last posible - 2021)
        """
        assert start_year != 1950, "Can't choose first year"
        assert end_year != 2022, "Can't choose last year"

        # --> Load fields annotations (path to dataset fields)
        annotation_file_geopotencital = np.loadtxt(
            os.path.join(self.root, self.name), dtype=np.str_)            

        date_to_day = DateToConsecutiveDay()

        for consecutive_day in range(1, 366):
            print(f"Day {consecutive_day} / 365")
            # Averages for all years for given consecutive day
            #year_averages = [] 

            for year in range(start_year, end_year+1):
                
                day, month = date_to_day.convert_consecutive_day_to_date(
                    consecutive_day=consecutive_day)
                
                # I don't subtract index, because no data is missing.
                index0 = date_to_day.num_of_samples(year, month, day, 1) 
                path0 = annotation_file_geopotencital[index0]
                field0 = np.expand_dims(np.load(path0), axis=0)
                
                # For smoothing use 5 days before and after as well
                # in climatology calculation.
                Fields = field0 # Fields = [field0]
                
                for t in range(1, 6):
                    # Days before
                    index_minus_t = index0 - t
                    path_minus_t = annotation_file_geopotencital[
                        index_minus_t]
                    field_minus_t = np.expand_dims(np.load(path_minus_t), axis=0)
                    Fields = np.append(Fields, field_minus_t, axis=0) # Fields.append(field_minus_t)
                    # Days after
                    index_plus_t = index0 + t
                    path_plus_t = annotation_file_geopotencital[
                        index_plus_t]
                    field_plus_t = np.expand_dims(np.load(path_plus_t), axis=0)
                    Fields = np.append(Fields, field_plus_t, axis=0) # Fields.append(field_plus_t)

                if year == start_year:
                    year_values = Fields
                else:
                    year_values = np.append(year_values, Fields, axis=0)

            consecutive_day_std = np.std(year_values, axis=0)

            # consecutive_day_climatology = np.mean(year_averages, axis=0)

            np.save(
                os.path.join(
                self.root_save_load, "natural_variability" + "_" + str(consecutive_day)),
                consecutive_day_std
                )

    def load_day_natural_variability_from_consecutive_day(
        self, consecutive_day):
        """
        Args:
        ---> consecutive_day (int): 1,2,...,365
        """
        path = os.path.join(
            self.root_save_load, "natural_variability" + "_"
            + str(consecutive_day) + ".npy")

        day_natural_variability = np.load(path)
        return day_natural_variability

    def load_day_natural_variability_from_date(
        self, date):
        """
        Args:
        ---> date (tuple): (day, month)
        """

        date_to_day = DateToConsecutiveDay()

        consecutive_day = date_to_day.convert_date_to_consecutive_day(
            date, delta_t=1)

        path = os.path.join(
            self.root_save_load, "natural_variability" + "_"
            + str(consecutive_day) + ".npy")

        day_natural_variability = np.load(path)
        
        return day_natural_variability

    def natural_variability_time_series_from_date(self, start_date,
                num_of_preiction_steps):
        """
        Args:
        --> start_date (tuple): (day, month)
        --> num_of_preiction_steps (int):
                number of consecutive climatologyies
        Returns:
        --> climas = [clima1, clima2, ..., clima_num_of_pred_steps]
        """

        date_to_day = DateToConsecutiveDay()

        consecutive_day = date_to_day.convert_date_to_consecutive_day(
            start_date, delta_t=1)

        # List with climatologies
        natural_variabilities = [] 
        
        for i in range(num_of_preiction_steps+1):
            day_natural_variability = self.load_day_natural_variability_from_consecutive_day(
                consecutive_day)
            natural_variabilities.append(day_natural_variability)
            consecutive_day += 1
            # Će pridemo čez 1. januar
            if consecutive_day > 365:
                consecutive_day = 1

        return natural_variabilities

    def natural_variability_time_series_from_consecutive_day(
            self, start_day, num_of_preiction_steps):
        """
        Args:
        --> start_day: start consecutive day in year
        --> num_of_preiction_steps (int):
                number of consecutive natural_variabilities
        Returns:
        --> natural_variabilities = [natural_variability1, 
                                     natural_variability2, 
                                            ..., 
                                     natural_variability_num_of_pred_steps]
        """
        # List with natural_variabilities
        natural_variabilities = [] 
        
        consecutive_day = start_day

        for i in range(num_of_preiction_steps+1):
            day_natural_variability = self.load_day_natural_variability_from_consecutive_day(
                consecutive_day)
            natural_variabilities.append(day_natural_variability)
            consecutive_day += 1
            # Će pridemo čez 1. januar
            if consecutive_day > 365:
                consecutive_day = 1

        return natural_variabilities

# Example how to run it
if __name__ == "__main__":
    natural_variability = CalculateNaturalVariability(
        "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density",
        "geopotential_annotation_file.txt",
        "/home/perkanu/Magistrska_naloga/U-Net/Data/3x_Reduced_Density/Geopotential_500")

    natural_variability.calculate_and_save(1951, 1955)

    # day1 = clima.load_day_climatology_from_consecutive_day(5)

    # import matplotlib.pyplot as plt
    # plt.imshow(day1)
    # plt.show()

    # #clima.calculate_and_save()
    # day2 = clima.load_day_climatology_from_date((5, 1))

    # plt.imshow(day2)
    # plt.show()

    # plt.imshow(day2-day1)
    # plt.show()
# %%
