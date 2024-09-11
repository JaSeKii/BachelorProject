import vtk
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
import dicom2nifti
#import pyvista as pv

from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

 
def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()

def read_a_surface_and_show_some_information():
    """
    Reads a surface using VTK and show some stats
    """
    name_renal = "C:/Users/jacob/OneDrive/Uni/7. Semester/Bachelor/billeder/lung2.vtk"

    pd = vtk.vtkPolyDataReader()
    pd.SetFileName(name_renal)
    pd.Update()
    cl = pd.GetOutput()
    n_points = cl.GetNumberOfPoints()


    print(f"Surface {name_renal} has {n_points} vertices")
    p_0 = cl.GetPoint(0)

def read_nifti_itk_to_vtk(file_name, img_mask_name=None, flip_for_volume_rendering=None):
    """
    Convenience function to read a NIFTI file using SimpleITK and transforming it into a VTK image
    (The SimpleITK image readers are way better than VTK image readers, but sometimes we need an image in VTK format)
    """
    try:
        img = sitk.ReadImage(file_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {file_name}")
        return None

    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()

    # convert the SimpleITK image to a numpy array
    i2 = sitk.GetArrayFromImage(img)

    vtk_image = vtk.vtkImageData()
    # VTK expects 3-dimensional parameters
    if len(size) == 2:
        size.append(1)
    if len(origin) == 2:
        origin.append(0.0)
    if len(spacing) == 2:
        spacing.append(spacing[0])
    if len(direction) == 4:
        direction = [
            direction[0],
            direction[1],
            0.0,
            direction[2],
            direction[3],
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        vtk_image.SetDirectionMatrix(direction)

    depth_array = numpy_to_vtk(i2.ravel(), deep=1)
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)

    vtk_image.Modified()
    return vtk_image

def convert_label_map_to_surface(label_name, output_file,  segment_id=1):
    vtk_img = read_nifti_itk_to_vtk(label_name)
    if vtk_img is None:
        return False

    print(f"Generating: {output_file}")
    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetNumberOfContours(1)
    mc.SetValue(0, segment_id)
    mc.Update()

    if mc.GetOutput().GetNumberOfPoints() < 10:
        print(f"No isosurface found in {label_name}")
        return False

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputConnection(mc.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(output_file)
    writer.Write()

    return True


def extract_segmentations_as_surfaces():
    label_name = "C:/Users/jacob/OneDrive/Uni/7. Semester/Bachelor/input_ct_segmentations.nii.gz"

    out_name_l1 = "C:/Users/jacob/OneDrive/Uni/7. Semester/Bachelor/billeder/lung1.vtk"
    out_name_l2 = "C:/Users/jacob/OneDrive/Uni/7. Semester/Bachelor/billeder/lung2.vtk"
    out_name_l3 = "C:/Users/jacob/OneDrive/Uni/7. Semester/Bachelor/billeder/lung3.vtk"
    out_name_l4 = "C:/Users/jacob/OneDrive/Uni/7. Semester/Bachelor/billeder/lung4.vtk"
    out_name_l5 = "C:/Users/jacob/OneDrive/Uni/7. Semester/Bachelor/billeder/lung5.vtk"

    convert_label_map_to_surface(label_name, out_name_l1, 10)
    convert_label_map_to_surface(label_name, out_name_l2, 11)
    convert_label_map_to_surface(label_name, out_name_l3, 12)
    convert_label_map_to_surface(label_name, out_name_l4, 13)
    convert_label_map_to_surface(label_name, out_name_l5, 14)

def read_an_image_and_sample_a_pixel_value():
    ct_name = "F:/DTU-Kidney-1/NIFTI/KIDNEY_HEALTHY_0064_SERIES0020.nii.gz"

    # use SimpleITK to read the NIFTI file
    try:
        img = sitk.ReadImage(ct_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {ct_name}")
        return

    # Extract image data in numpy format
    img_t = sitk.GetArrayFromImage(img)

    # Due to the coordinate conventions in SimpleITK and numpy we need to reorder the image
    img_np = img_t.transpose(2, 1, 0)
    print(f"Numpy image shape {img_np.shape}")

    # Position in physcial coordinates (in mm)
    pos = [11.7, 4.2, 1773.2]

    # Get the index coordinates of the point
    p_idx = img.TransformPhysicalPointToIndex(pos)

    # Sample the voxel value from the numpy array (that is why we needed to transpose the numpy array)
    vox_val = img_np[p_idx]

    print(f"Physical point {pos} has indices {p_idx} with voxel value {vox_val}")


def read_an_image_change_in_numpy_and_save_again():
    """
    Here an image is read and converted to numpy.
    The values in the numpy array is changed and put into a new image that are then saved using SimpleITK
    """
    ct_name = "F:/DTU-Kidney-1/NIFTI/KIDNEY_HEALTHY_0064_SERIES0020.nii.gz"
    ct_proc_out = "C:/data/RenalArteries/processing/modified_image.nii.gz"

    # use SimpleITK to read the NIFTI file
    try:
        img = sitk.ReadImage(ct_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {ct_name}")
        return

    # Extract image data in numpy format
    img_t = sitk.GetArrayFromImage(img)

    # Due to the coordinate conventions in SimpleITK and numpy we need to reorder the image
    img_np = img_t.transpose(2, 1, 0)
    print(f"Numpy image shape {img_np.shape}")

    # Position in physcial coordinates (in mm)
    pos = [11.7, 4.2, 1773.2]

    # Get the index coordinates of the point
    p_idx = img.TransformPhysicalPointToIndex(pos)

    # Set the value in the given voxel
    img_np[p_idx] = -1000

    # Now create a new image by transforming numpy array into image. Remember to transpose back again
    img_o_np = img_np.transpose(2, 1, 0)

    img_o = sitk.GetImageFromArray(img_o_np)
    # Copy essential image information from the original ct scan (spacing, origin, directions and so on)
    img_o.CopyInformation(img)

    print(f"saving")
    sitk.WriteImage(img_o, ct_proc_out)

if __name__ == '__main__':
    ct_name  = '/home/s214596/Bachelor project/BachelorProject/4_lung_15.nii.gz'
    Segmentations = '/home/s214596/Bachelor project/BachelorProject/ct_segmentations.nii.gz'
    lung_vessels = '/home/s214596/Bachelor project/BachelorProject/lung_vessels.nii.gz'

    #load ct file
    img = sitk.ReadImage(ct_name)
    img_t = sitk.GetArrayFromImage(img)
    img_np = img_t.transpose(2, 1, 0)
    print(f"Numpy image shape {img_np.shape}")

    #Load Segmentation file
    seg = sitk.ReadImage(Segmentations)
    seg_t = sitk.GetArrayFromImage(seg)
    seg_np = seg_t.transpose(2, 1, 0)
    seg_np = np.isin(seg_np,np.array([10,11,12,13,14])).astype(int)
    
    #load lung vessel segmentation
    lung_seg = sitk.ReadImage(lung_vessels)
    lung_seg_t = sitk.GetArrayFromImage(lung_seg)
    lung_seg_np = lung_seg_t.transpose(2,1,0)


    result_lung = np.multiply(img_np,seg_np)
    result_lung_no_vessels = np.multiply(result_lung,lung_seg_np)

    plt.hist(result_lung_no_vessels.ravel(),bins='auto')
    plt.show()

    img_o_np = result_lung_no_vessels.transpose(2, 1, 0)

    img_o = sitk.GetImageFromArray(img_o_np)
    # Copy essential image information from the original ct scan (spacing, origin, directions and so on)
    img_o.CopyInformation(img)

    print(f"saving")
    sitk.WriteImage(img_o, 'result.nii.gz')