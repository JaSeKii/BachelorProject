'''
Functions for preprocessing and segmentation nifti images
'''

import os
import vtk
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import seaborn as sns
from totalsegmentator.python_api import totalsegmentator


def get_segmentations(input_file_path, output_path, task="total", fast=True):
    '''
    Get segmentations using TotalSegmentator.

    input: path to input nifti file

    output: path to folder for segmentations

    task: what type of segmentation (total, lung_vessels, ...) see documentation on github
    '''

    input_file = input_file_path
    output_dir = output_path
    
    if not os.path.exists(input_file):
        print(f"Could not find {input_file}")
        return False

    multi_label = True
    # Nr of threads for resampling
    nr_thr_resamp = 1
    # Nr of threads for saving segmentations
    nr_thr_saving = 1
    # Run faster lower resolution model
    fast_model = fast
    if task == 'lung_vessels':
        fast_model = False

    # Look at the TotalSegmentator documentation for more information on the tasks
    task = task

    # Calc volume (in mm3) and mean intensity. Results will be in statistics.json
    calc_statistics = False
    # Calc radiomics features. Requires pyradiomics. Results will be in statistics_radiomics.json
    calc_radiomics = False
    # Do initial rough body segmentation and crop image to body region
    body_seg = False
    # Process image in 3 chunks for less memory consumption
    force_split = False
    run_quit = True
    verbose = False

    totalsegmentator(input_file, output_dir, multi_label, nr_thr_resamp, nr_thr_saving,
                     fast_model, nora_tag="None", preview=False, task=task, roi_subset=None,
                     statistics=calc_statistics, radiomics=calc_radiomics, crop_path=None, body_seg=body_seg,
                     force_split=force_split, output_type="nifti", quiet=run_quit, verbose=verbose, test=False)

    return True

def load_nifti_convert_to_numpy(input_path, state_shape=False):
    '''
    input: path to nifti file

    output: numpy array of image
    '''
    img = sitk.ReadImage(input_path)
    img_t = sitk.GetArrayFromImage(img)
    img_np = img_t.transpose(2, 1, 0)
    if state_shape:
        print(f"Numpy image shape {img_np.shape}")
    return img_np

def convert_numpy_to_nifti_and_save(np_file, output_path, original_nifti_path):
    img = sitk.ReadImage(original_nifti_path)
    img_o_np = np_file.transpose(2, 1, 0)

    img_o = sitk.GetImageFromArray(img_o_np)
    # Copy essential image information from the original ct scan (spacing, origin, directions and so on)
    img_o.CopyInformation(img)

    print(f"saving")
    sitk.WriteImage(img_o, output_path)

def segment_lungs_without_vessels(ct_img, lung_seg, vessel_seg):
    '''
    given the ct image, total segmentation and lung vessel segmentation as arrays:

    output: array of lungs without vessels and 1d-array of lung tissue without lung vessels 
    '''

    # Preprocess the segmentations to binary in order to multiply them with the ct array.
    lung_seg = np.isin(lung_seg,np.array([10,11,12,13,14])).astype(int) #lung segment numbers is  [10:14]
    vessel_seg = np.where(vessel_seg > 0,0,1)

    # multiply the ct image with the lung segmentation, to isolate the lungs
    result_lung = np.multiply(ct_img,lung_seg)

    #multiply the isolated lung ct with the lung vessel segmentation to remove the vessels from the ct.
    result_lung_no_vessels = np.multiply(result_lung,vessel_seg)

    # remove all ct values equal to 0
    attenuation = result_lung_no_vessels.ravel()
    attenuation = attenuation[attenuation != 0]
    return result_lung_no_vessels, attenuation

def density_plot(attenuation):
    sns.set_style('whitegrid')
    sns.displot(attenuation)
    plt.show()

if __name__ == "__main__":
    #compute_totalsegmentator_segmentations()
    output_path = "/home/s214596/Bachelor project/BachelorProject/Data_preprocess/Segmentations/test1.nii.gz"
    get_segmentations(input_file_path='none', output_path=output_path)