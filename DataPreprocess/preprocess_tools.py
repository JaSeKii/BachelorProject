'''
Functions for preprocessing and segmentation nifti images
'''

import os
#import vtk
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import seaborn as sns
import json
from pathlib import Path
from glob import glob
from totalsegmentator.python_api import totalsegmentator

#Function from batchgenerators by Isensee
def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def get_segmentations(input_file_path, output_path, task="total", fast=True):
    '''
    Get segmentations using TotalSegmentator.

    input: path to input nifti file, path to folder for segmentations

    output: 

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

def resample_image(im_path : str, interpolator = sitk.sitkLinear, new_spacing = [0.5]*3) -> sitk.Image:

    """
    The function reads and resamples an image to have a given voxel. 

    Input: 

        im_path:        The path to the image that needs resampling 

        intepolator:    The intepolator for the resampling, default is linear

        new_spacing:    The new voxel, default is (0.5, 0.5, 0.5)

    Output: 

        A resampled image represented as a 3D array

    """

    # load image 
    im = sitk.ReadImage(im_path) 

    # calculate new size of resampled image 
    original_spacing = im.GetSpacing()
    original_size = im.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    # resample 
    return sitk.Resample(im, new_size, 
                         transform = sitk.Transform(), 
                         interpolator = interpolator, 
                         outputOrigin = im.GetOrigin(), 
                         outputSpacing = new_spacing, 
                         outputDirection = im.GetDirection(), 
                         defaultPixelValue = 0, 
                         outputPixelType = im.GetPixelID())
class idxCounter:
    def __init__(self):
        '''
        An object that is simply a 3 digit index that increases by one every time step() is called.
        Used to name data files correctly after nnUNet standards
        '''
        self.fdigit = 0
        self.sdigit = 0
        self.tdigit = 0
        
    def step(self):
        if self.fdigit == 9:
            self.sdigit += 1
            self.fdigit = 0
        elif self.sdigit == 9 and self.fdigit==9:
            self.tdigit += 1
            self.sdigit = 0
            self.fdigit = 0
        else: self.fdigit += 1
    
    def __repr__(self):
        return f'{self.tdigit}{self.sdigit}{self.fdigit}'
        

def covidDatasetResampler(input_path,input_GT_seg_path, output_path,output_path_GT):
    healthy = glob(os.path.join(input_path,"[0-9][0-9][0-9][0-9].nii.gz"))
    sick = glob(os.path.join(input_path,"*_[0-9][0-9][0-9]*.nii.gz"))
    
    maybe_mkdir_p(output_path)
    maybe_mkdir_p(input_GT_seg_path)
    maybe_mkdir_p(os.path.join(Path(output_path).parent,'GT_segmentations'))
    idx = idxCounter()
    
    if (len(sick) + len(healthy)) > len(os.listdir(output_path)):
        for patient in sick:
            p_id = patient.split('/')[-1]
            resampled_patient = resample_image(patient)
            resampled_GT_seg = resample_image(input_GT_seg_path+p_id)
            print(input_GT_seg_path+p_id)
            sitk.WriteImage(resampled_patient, os.path.join(output_path,f'Covid_sick_{idx}_0000.nii.gz'))
            sitk.WriteImage(resampled_GT_seg, output_path_GT+f'Covid_sick_{idx}.nii.gz')
            idx.step()
            
        for patient in healthy:
            resampled_patient = resample_image(patient)
            sitk.WriteImage(resampled_patient, os.path.join(output_path,f'Covid_healthy_{idx}_0000.nii.gz'))
            idx.step()

    new_dataset = glob(os.path.join(output_path,"*.nii.gz"))
    return new_dataset


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
    '''
    input: numpy file, path for nifti file, path to original nifti file (for metadata)
    Convert a numpy array into a nifti file and save 
    '''
    img = sitk.ReadImage(original_nifti_path)
    img_o_np = np_file.transpose(2, 1, 0)

    img_o = sitk.GetImageFromArray(img_o_np)
    # Copy essential image information from the original ct scan (spacing, origin, directions and so on)
    img_o.CopyInformation(img)

    print(f"saving")
    sitk.WriteImage(img_o, output_path)

    


def segment_lungs_with_vessels(ct_img, total_seg, Attenuation = False):
    '''
    given the ct image and total segmentation as arrays:

    output: array of lungs with vessels 
    '''
    # Preprocess the segmentations to binary in order to multiply them with the ct array.
    lung_seg = np.isin(total_seg,np.array([10,11,12,13,14])).astype(int) #lung segment numbers is  [10:14]
    # multiply the ct image with the lung segmentation, to isolate the lungs
    result_lung = np.where(lung_seg==1,ct_img,-10000)
    # remove all ct values equal to 0
    
    attenuation = []
    if Attenuation:
        attenuation = result_lung.ravel()
        attenuation = attenuation[attenuation != -10000]
    return result_lung, attenuation


def segment_lung_with_GT_from_Total_seg(total_seg, GT_seg, Has_GT=True):
    lung_seg = np.isin(total_seg,np.array([10,11,12,13,14])).astype(int)
    if Has_GT:
        GT_seg = np.where(GT_seg>0,2,0)
        lung_with_GT_seg = np.where(GT_seg>0,GT_seg,lung_seg)
        return lung_with_GT_seg
    return lung_seg


def segment_lungs_without_vessels(ct_img, total_seg, vessel_seg, Attenuation = True):
    '''
    given the ct image, total segmentation and lung vessel segmentation as arrays:

    output: array of lungs without vessels and 1d-array of lung tissue without lung vessels 
    '''

    # Preprocess the segmentations to binary in order to multiply them with the ct array.
    lung_seg = np.isin(total_seg,np.array([10,11,12,13,14])).astype(int) #lung segment numbers is  [10:14]
    vessel_seg = np.where(vessel_seg > 0,0,1)

    # multiply the ct image with the lung segmentation, to isolate the lungs
    lung_seg = np.where(lung_seg==0,np.nan,1) 
    result_lung = np.multiply(ct_img,lung_seg)
    result_lung = np.where(np.isnan(result_lung),-10000,result_lung)

    # multiply the isolated lung ct with the lung vessel segmentation to remove the vessels from the ct.
    result_lung_no_vessels = np.where(vessel_seg==1,np.multiply(result_lung,vessel_seg),-10000)

    # remove all ct values equal to 0
    attenuation = result_lung_no_vessels.ravel()
    attenuation = attenuation[attenuation != -10000]
    return result_lung_no_vessels, attenuation



def extract_dataset_from_collection(whitelist_file_path, path_to_dataset):
    '''
    given a path to a json file containing the relevant nifti filenames, and a path to the raw dataset

    extract filenames of the whitelisted nifti files from the dataset
    '''
    filtered_dataset = []
    with open(whitelist_file_path, 'r') as json_file:
        whitelist = json.load(json_file)
    whitelist_filenames = [fn['filepath'] for fn in whitelist.values()]
    for patient_id in os.listdir(path_to_dataset):
        if patient_id[0:28] in whitelist_filenames and os.path.isfile(os.path.join(path_to_dataset, patient_id)):
            filtered_dataset.append(patient_id[0:28] + '.nii.gz')
    return filtered_dataset



if __name__ == "__main__":
    #compute_totalsegmentator_segmentations()
    output_path = "/home/s214596/Bachelor project/BachelorProject/Data_preprocess/Segmentations/test1.nii.gz"
    get_segmentations(input_file_path='none', output_path=output_path)