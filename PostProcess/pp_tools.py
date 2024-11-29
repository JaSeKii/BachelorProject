import SimpleITK as sitk
import numpy as np

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