from preprocess_tools import *
import pathlib

input_path = str(pathlib.Path().resolve())
output_path = str(pathlib.Path(__file__).parent.resolve()) + "/Segmentations/"

dataset = ['/4_lung_15.nii.gz']
patient = dataset[0]

get_segmentations(input_file_path=input_path + patient,
                    output_path=output_path + f'full_seg_{patient}.nii.gz',
                    task='total')
# get_segmentations(input_file_path=input_path + patient,
#                     output_path=output_path + f'vessel_seg_{patient}',
#                     task='lung_vessels')



# Get lung segmentation without lung vessels:

ct_as_np = load_nifti_convert_to_numpy(input_path=input_path+patient)
lung_seg_as_np = load_nifti_convert_to_numpy(input_path=output_path+f'full_seg_{patient}')
vessel_seg_as_np = load_nifti_convert_to_numpy(input_path=output_path+f'vessel_seg_{patient}')

lungs_wo_vessels, attenuation_of_lungs = segment_lungs_without_vessels(ct_as_np, lung_seg_as_np, vessel_seg_as_np)
print(attenuation_of_lungs)



