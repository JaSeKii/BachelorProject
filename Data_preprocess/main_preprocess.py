# import os
# import vtk
# import numpy as np
# import matplotlib.pyplot as plt
# #import SimpleITK as sitk
from os import listdir
from os.path import isfile, join
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

    output_path_lung_wo_vessel = '/scratch/s214596/Dataset/processed_data/lung_wo_vessels/'
    output_path_lung = '/scratch/s214596/Dataset/processed_data/lungs/'
    
elif cluster == 'Rigshospitalet':
    #input paths
    input_path_Dataset = str(Path().resolve()) + "/"
    #input_path_Dataset = ''
    input_path_segmentations = str(Path(__file__).parent.resolve()) + "/Segmentations/"

    #outout paths
    output_path_segmentations = str(Path(__file__).parent.resolve()) + "/Segmentations/"
    output_path_lung_wov_attenuation = str(Path(__file__).parent.resolve()) + "/Attenuation/"

    output_path_lung_wo_vessel = 'Dataset/processed_data/lung_wo_vessels/'
    output_path_lung = 'Dataset/processed_data/lungs/'


if __name__ == "__main__":
    dataset = [f for f in listdir(input_path_Dataset) if isfile(join(input_path_Dataset, f))]

    #flags
    segmentate = False
    fast = False

    dataset = ['4_lung_15.nii.gz']
    
    for patient in dataset:
        if segmentate:
            get_segmentations(input_file_path=input_path_Dataset + patient,
                                output_path=output_path_segmentations + f'total_seg_{patient}',
                                task='total', fast=fast)
            get_segmentations(input_file_path=input_path_Dataset + patient,
                                output_path=output_path_segmentations + f'vessel_seg_{patient}',
                                task='lung_vessels', fast=fast)
            get_segmentations(input_file_path=input_path_Dataset + patient,
                                output_path=output_path_segmentations + f'pleural_effusion_seg_{patient}',
                                task='pleural_pericard_effusion', fast=fast)


        # Get lung segmentation without lung vessels:

        # convert nifti files to numpy arrays in order to process them.
        ct_as_np = load_nifti_convert_to_numpy(input_path=input_path_Dataset+patient)
        lung_seg_as_np = load_nifti_convert_to_numpy(input_path=input_path_segmentations+f'total_seg_{patient}')
        vessel_seg_as_np = load_nifti_convert_to_numpy(input_path=input_path_segmentations+f'vessel_seg_{patient}')
        
        # extract CT of the lungs with lung vessels.
        lung_w_vessels = segment_lungs_with_vessels(ct_as_np, lung_seg_as_np)

        # extract CT of the lungs without the lung vessels, and the attenuation of the lungs (w.o. vessels)
        lungs_wo_vessels, attenuation_of_lungs = segment_lungs_without_vessels(ct_as_np, lung_seg_as_np, vessel_seg_as_np)
        
        #convert the processed arrays back to nifti and save to scratch directory. 
        convert_numpy_to_nifti_and_save(lung_w_vessels,output_path_lung+ f'Lung_{patient}',input_path_Dataset+patient)
        convert_numpy_to_nifti_and_save(lungs_wo_vessels,output_path=output_path_lung_wo_vessel+f'Lung_wo_vessels_{patient}',original_nifti_path=input_path_Dataset+patient)
    
        np.save(output_path_lung_wov_attenuation+f'attenuation_{patient}.npy', attenuation_of_lungs)
    # dataset directory : /scratch/s214596/Dataset


