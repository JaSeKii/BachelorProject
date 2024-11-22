# import os
# import vtk
# import numpy as np
# import matplotlib.pyplot as plt
# #import SimpleITK as sitk
from os import listdir
from os.path import isfile, join
import json
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess_tools import *

'''
*****FLAGS*****
'''
Segmentate = True       # Do segmentations with totalsegmentator
Fast = False            # Use totalsegmentators fast flag (3mm resolution instead of 1.5mm)
Total = True            # Use totalsegmentators total segmentation (main task)
Vessel = False          # Use totalsegmentators lung vessel segmentation task
Effusion = False        # Use totalsegmentators pleural effusion segmentation task
Attenuation = False     # Create npy file with 1d array of attenuation values 
cluster = 'Titans'      # Options : 'Rigshospitalet', 'local' , 'Titans'
Resample = True         # Resample data into having a standardized physical spacing (0.5,0.5,0.5)
dataset_type = 'Covid'
Verbose = True

if cluster == 'Titans':
    #input paths
    input_path_Dataset = '/scratch/s214596/Preprocess_dataset/CovidDataset/raw_CT_data'
    input_path_GT_segmentations = str(Path(input_path_Dataset).resolve()) + "/_GT_raw/"
    input_path_total_segmentations = str(Path(input_path_Dataset).parent.resolve()) + "/Segmentations/"
    input_path_Resampled_data = str(Path(input_path_Dataset).parent.resolve()) + "/Resampled_data/"

#outout paths
output_path_Resampled_data = input_path_Resampled_data
output_path_total_segmentations = input_path_total_segmentations
output_path_lung_wo_vessel = str(Path(input_path_Dataset).parent.resolve()) + '/processed_data/lung_wo_vessels/'
output_path_lung = str(Path(input_path_Dataset).parent.resolve()) + '/processed_data/lungs/'
output_path_lung_wov_attenuation = str(Path(input_path_Dataset).parent.resolve()) + "/Attenuation/"



if __name__ == "__main__":
    maybe_mkdir_p(input_path_Dataset)
    maybe_mkdir_p(input_path_total_segmentations)
    maybe_mkdir_p(output_path_lung)
   
    if dataset_type == 'Covid':
        if Verbose: print('Resampling and distrbuting raw data from Covid dataset')
        dataset = covidDatasetResampler(input_path_Dataset, input_path_GT_segmentations, output_path_Resampled_data)
        if Verbose: print('Resampling complete')
    #dataset = extract_dataset_from_collection(input_path_json, input_path_Dataset)

    if Resample and dataset_type!='Covid': 
        for patient in dataset:
            print('this should not be active!')
            patient_ct_resampled = resample_image(input_path_Dataset + patient) 
            sitk.WriteImage(patient_ct_resampled,input_path_Resampled_data + patient)
    
    for patient in tqdm(dataset):
        p_id = patient.split('/')[-1]
        
        if Segmentate and not Resample:
            print('this shouldnt happen either')
            input_path_Resampled_data = input_path_Dataset
            
        if Segmentate and Total:
            get_segmentations(input_file_path=input_path_Resampled_data + p_id,
                                output_path=output_path_total_segmentations + f'LungSEG_{p_id}',
                                task='total', fast=Fast)
        if Segmentate and Vessel:
            get_segmentations(input_file_path=input_path_Resampled_data + p_id,
                                output_path=output_path_total_segmentations + f'vessel_seg_{p_id}',
                                task='lung_vessels', fast=Fast)
        if Segmentate and Effusion:
            get_segmentations(input_file_path=input_path_Resampled_data + p_id,
                                output_path=output_path_total_segmentations + f'pleural_effusion_seg_{p_id}',
                                task='pleural_pericard_effusion', fast=Fast)
        
        # Get lung segmentation without lung vessels:

        # convert nifti files to numpy arrays in order to process them.
        ct_as_np = load_nifti_convert_to_numpy(input_path=input_path_Resampled_data+p_id)
        
        if Total:
            lung_seg_as_np = load_nifti_convert_to_numpy(input_path=output_path_total_segmentations+f'LungSEG_{p_id}')
        if Vessel:
            vessel_seg_as_np = load_nifti_convert_to_numpy(input_path=input_path_total_segmentations+f'vessel_seg_{p_id}')
        
        
        # extract CT of the lungs with lung vessels and resample.
        lung_w_vessels, attenuation = segment_lungs_with_vessels(ct_as_np, lung_seg_as_np)
        #convert the processed arrays back to nifti and save to scratch directory. 
        convert_numpy_to_nifti_and_save(lung_w_vessels,output_path_lung+ f'Lung_{p_id}',input_path_Resampled_data+p_id)
        
        
        
        if Vessel and Attenuation:
            # extract CT of the lungs without the lung vessels, and the attenuation of the lungs (w.o. vessels)
            lungs_wo_vessels, attenuation_of_lungs = segment_lungs_without_vessels(ct_as_np, lung_seg_as_np, vessel_seg_as_np)
        elif Vessel and not Attenuation:
            lungs_wo_vessels, _ = segment_lungs_without_vessels(ct_as_np, lung_seg_as_np, vessel_seg_as_np, False)
        
        if Vessel:
            convert_numpy_to_nifti_and_save(lungs_wo_vessels,output_path=output_path_lung_wo_vessel+f'Lung_wo_vessels_{p_id}',original_nifti_path=input_path_Resampled_data+p_id)
        if Attenuation and Vessel:
            np.save(output_path_lung_wov_attenuation+f'attenuation_{p_id}.npy', attenuation_of_lungs)
    


