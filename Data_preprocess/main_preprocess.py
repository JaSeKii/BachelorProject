# import os
# import vtk
# import numpy as np
# import matplotlib.pyplot as plt
# #import SimpleITK as sitk
from os import listdir
from os.path import isfile, join
from pathlib import Path
import seaborn as sns
import tempfile
import matplotlib.pyplot as plt
from preprocess_tools import *

#input paths
'''
Set inputpath for the source dataset
'''
input_path_Dataset = '/scratch/s214596/Preprocess_dataset/CovidDataset/raw_CT_data'

input_path_CT = str(Path(input_path_Dataset).resolve()) + "/"
input_path_segmentations = str(Path(input_path_Dataset).parent.resolve()) + "/Segmentations/"

#outout paths
output_path_segmentations = str(Path(input_path_Dataset).parent.resolve()) + "/Segmentations/"
output_path_lung_wo_vessel = str(Path(input_path_Dataset).parent.resolve()) + '/processed_data/lung_wo_vessels/'
output_path_lung = str(Path(input_path_Dataset).parent.resolve()) + '/processed_data/lungs/'
output_path_lung_wov_attenuation = str(Path(input_path_Dataset).parent.resolve()) + "/Attenuation/"

'''
*****FLAGS*****
'''
Segmentate = False
Fast = False
Total = True
Vessel = False
Effusion = False
Attenuation = False

if __name__ == "__main__":
    maybe_mkdir_p(input_path_Dataset)
    maybe_mkdir_p(input_path_segmentations)
    
    dataset = [f for f in listdir(input_path_Dataset) if isfile(join(input_path_Dataset, f))]
    
    for patient in dataset:
        if Segmentate and Total:
            get_segmentations(input_file_path=input_path_CT + patient,
                                output_path=output_path_segmentations + f'LungSEG_{patient}',
                                task='total', fast=Fast)
        if Segmentate and Vessel:
            get_segmentations(input_file_path=input_path_CT + patient,
                                output_path=output_path_segmentations + f'vessel_seg_{patient}',
                                task='lung_vessels', fast=Fast)
        if Segmentate and Effusion:
            get_segmentations(input_file_path=input_path_CT + patient,
                                output_path=output_path_segmentations + f'pleural_effusion_seg_{patient}',
                                task='pleural_pericard_effusion', fast=Fast)
        print(patient)

        # Get lung segmentation without lung vessels:

        # convert nifti files to numpy arrays in order to process them.
        ct_as_np = load_nifti_convert_to_numpy(input_path=input_path_CT+patient)
        if Total:
            lung_seg_as_np = load_nifti_convert_to_numpy(input_path=input_path_segmentations+f'LungSEG_{patient}')
        if Vessel:
            vessel_seg_as_np = load_nifti_convert_to_numpy(input_path=input_path_segmentations+f'vessel_seg_{patient}')
        
        
        # extract CT of the lungs with lung vessels and resample.
        lung_w_vessels = segment_lungs_with_vessels(ct_as_np, lung_seg_as_np)

        if Vessel and Attenuation:
            # extract CT of the lungs without the lung vessels, and the attenuation of the lungs (w.o. vessels)
            lungs_wo_vessels, attenuation_of_lungs = segment_lungs_without_vessels(ct_as_np, lung_seg_as_np, vessel_seg_as_np)
        elif Vessel and not Attenuation:
            lungs_wo_vessels, _ = segment_lungs_without_vessels(ct_as_np, lung_seg_as_np, vessel_seg_as_np, False)
        
        #convert the processed arrays back to nifti and save to scratch directory. 
        convert_numpy_to_nifti_and_save(lung_w_vessels,output_path_lung+ f'Lung_{patient}',input_path_CT+patient)
        
        if Vessel:
            convert_numpy_to_nifti_and_save(lungs_wo_vessels,output_path=output_path_lung_wo_vessel+f'Lung_wo_vessels_{patient}',original_nifti_path=input_path_CT+patient)
        if Attenuation and Vessel:
            np.save(output_path_lung_wov_attenuation+f'attenuation_{patient}.npy', attenuation_of_lungs)
    # dataset directory : /scratch/s214596/Dataset


