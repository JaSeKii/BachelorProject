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

cluster = 'Rigshospitalet' # 'Rigshospitalet', 'local' , 'Titans'



if cluster == 'Titans':
    #input paths
    input_path_Dataset = '/scratch/s214596/Dataset/raw_data'
    input_path_segmentations = str(Path(__file__).parent.resolve()) + "/Segmentations/"

    #outout paths
    output_path_segmentations = str(Path(__file__).parent.resolve()) + "/Segmentations/"
    output_path_lung_wov_attenuation = str(Path(__file__).parent.resolve()) + "/Attenuation/"

    output_path_lung_wo_vessel = '/scratch/s214596/Dataset/processed_data/lungs_wo_vessels/'
    output_path_lung = '/scratch/s214596/Dataset/processed_data/lungs/'
    
elif cluster == 'Rigshospitalet':
    #input paths
    input_path_whitelist_patients = 'E:/Lungeholdet2024/BachelorProject/Patient_Info_Lung_Pilot_1.json'
    input_path_Dataset_raw = 'I:/DTU-Lung-Pilot-1/NIFTI/'
    input_path_Dataset_resampled = '"E:/Lungeholdet2024/DTU-Lung-Pilot-1/Dataset/raw_resampled_data"'
    input_path_segmentations = str(Path(__file__).parent.resolve()) + "/Segmentations/"

    #outout paths
    output_path_segmentations = str(Path(__file__).parent.resolve()) + "/Segmentations/"
    output_path_lung_wov_attenuation = str(Path(__file__).parent.resolve()) + "/Attenuation/"

    output_path_lung_wo_vessel = 'E:/Lungeholdet2024/DTU-Lung-Pilot-1/Dataset/processed_data/lungs_wo_vessels/'
    output_path_lung = 'E:/Lungeholdet2024/DTU-Lung-Pilot-1/Dataset/processed_data/lungs/'


if __name__ == "__main__":
    
    dataset = extract_dataset_from_collection(input_path_whitelist_patients, input_path_Dataset_raw)

    #dataset = [f for f in listdir(input_path_Dataset) if isfile(join(input_path_Dataset, f)) and f in whitelist.values()]
    #print(dataset)
    #flags
    segmentate = True
    fast = False
    resample = True
    if resample: 
        for patient in dataset:
           patient_ct_resampled = resample_image(input_path_Dataset_raw + patient) 
           sitk.WriteImage(patient_ct_resampled,input_path_Dataset_resampled + patient)

    #dataset = ['4_lung_15.nii.gz']
    
    for patient in tqdm(dataset):
        if segmentate:
            get_segmentations(input_file_path=input_path_Dataset_resampled + patient,
                                output_path=output_path_segmentations + f'total_seg_{patient}',
                                task='total', fast=fast)
            get_segmentations(input_file_path=input_path_Dataset_resampled + patient,
                                output_path=output_path_segmentations + f'vessel_seg_{patient}',
                                task='lung_vessels', fast=fast)
            # get_segmentations(input_file_path=input_path_Dataset + patient,
            #                     output_path=output_path_segmentations + f'pleural_effusion_seg_{patient}',
            #                     task='pleural_pericard_effusion', fast=fast)

        patient_name = patient[:10]+patient[15:17]
        # Get lung segmentation without lung vessels:

        # convert nifti files to numpy arrays in order to process them.
        ct_as_np = load_nifti_convert_to_numpy(input_path=input_path_Dataset+patient)
        lung_seg_as_np = load_nifti_convert_to_numpy(input_path=input_path_segmentations+f'total_seg_{patient}')
        vessel_seg_as_np = load_nifti_convert_to_numpy(input_path=input_path_segmentations+f'vessel_seg_{patient}')
        
        # extract CT of the lungs with lung vessels.
        lung_w_vessels, attenuation_of_lungs = segment_lungs_with_vessels(ct_as_np, lung_seg_as_np)

        # extract CT of the lungs without the lung vessels, and the attenuation of the lungs (w.o. vessels)
        lungs_wo_vessels, attenuation_of_lungs_wo_vessels = segment_lungs_without_vessels(ct_as_np, lung_seg_as_np, vessel_seg_as_np)
        
        #convert the processed arrays back to nifti and save to scratch directory. 
        convert_numpy_to_nifti_and_save(lung_w_vessels,output_path_lung+ f'{patient}',input_path_Dataset+patient)
        convert_numpy_to_nifti_and_save(lungs_wo_vessels,output_path=output_path_lung_wo_vessel+f'wo_vessels_{patient}',original_nifti_path=input_path_Dataset+patient)

        np.save(output_path_lung_wov_attenuation+f'attenuation_lung_{patient_name}.npy', attenuation_of_lungs)
        np.save(output_path_lung_wov_attenuation+f'attenuation_no_ves_lung_{patient_name}.npy', attenuation_of_lungs_wo_vessels)
    # dataset directory : /scratch/s214596/Dataset


