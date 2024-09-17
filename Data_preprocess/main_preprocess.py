# import os
# import vtk
# import numpy as np
# import matplotlib.pyplot as plt
# #import SimpleITK as sitk
from pathlib import Path
from preprocess_tools import get_segmentations, load_nifti_convert_to_numpy, segment_lungs_without_vessels, convert_numpy_to_nifti_and_save


input_path = str(Path().resolve()) + "/"
output_path = str(Path(__file__).parent.resolve()) + "/Segmentations/"
#output_path = "/home/s214596/Bachelor project/BachelorProject/Data_preprocess"

dataset = ['4_lung_15.nii.gz']
patient = dataset[0]



if __name__ == "__main__":
    get_segmentations(input_file_path=input_path + patient,
                        output_path=output_path + f'total_seg_{patient}',
                        task='total', fast=False)
    # get_segmentations(input_file_path=input_path + patient,
    #                     output_path=output_path + f'vessel_seg_{patient}',
    #                     task='lung_vessels')


    # Get lung segmentation without lung vessels:

    # convert nifti files to numpy arrays in order to process them.
    ct_as_np = load_nifti_convert_to_numpy(input_path=input_path+patient)
    lung_seg_as_np = load_nifti_convert_to_numpy(input_path=output_path+f'total_seg_{patient}')
    vessel_seg_as_np = load_nifti_convert_to_numpy(input_path=output_path+f'vessel_seg_{patient}')

    # extract ct of lungs without the lung vessels, and the attenuation of the lungs (w.o. vessels)
    lungs_wo_vessels, attenuation_of_lungs = segment_lungs_without_vessels(ct_as_np, lung_seg_as_np, vessel_seg_as_np)


    convert_numpy_to_nifti_and_save(lungs_wo_vessels,output_path=output_path+f'Lung_wo_vessels_{patient}',original_nifti_path=input_path+patient)
    print(attenuation_of_lungs)



