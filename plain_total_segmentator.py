import os
import dicom2nifti
import nibabel as nib
import pydicom
from totalsegmentator.python_api import totalsegmentator
from pathlib import Path
import time

def compute_totalsegmentator_segmentations():
    """
    Use TotalSegmentator to compute segmentations
    """
    input_file = "C:/Users/jacob/OneDrive/Uni/7. Semester/Bachelor/4_lung_15.nii.gz"

    # Actually just a file name, not a directory (since we pack all segmentations in one file)
    output_dir = "C:/Users/jacob/OneDrive/Uni/7. Semester/Bachelor/input_ct_segmentations.nii.gz"

    if not os.path.exists(input_file):
        print(f"Could not find {input_file}")
        return False

    multi_label = True
    # Nr of threads for resampling
    nr_thr_resamp = 1
    # Nr of threads for saving segmentations
    nr_thr_saving = 1
    # Run faster lower resolution model
    fast_model = True

    # Look at the TotalSegmentator documentation for more information on the tasks
    task = "total"

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




if __name__ == "__main__":
    # dicom_ct = Path("/Users/jacob/OneDrive/Uni/7. Semester/Bachelor/DICOMData/Subject (1)")
    # dicom2nifti.convert_directory(dicom_ct, ".")
    
    nifti = nib.load("input_ct_segmentations.nii.gz")
    compute_totalsegmentator_segmentations()
