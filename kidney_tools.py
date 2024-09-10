import os
# import time
# from pathlib import Path
import vtk
import SimpleITK as sitk
# from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import vtk_to_numpy
from skimage.morphology import binary_opening, binary_closing, binary_dilation, binary_erosion
from scipy.ndimage import binary_fill_holes
from skimage.morphology import ball
from skimage.measure import label
import numpy as np
from scipy.ndimage import center_of_mass

# VMTK only works with a very specific environment
try:
    from vmtk import vmtkscripts
except ImportError:
    pass


def setup_vtk_error_handling(err_dir):
    """
    Create a text file where potential VTK errors are dumped
    """
    error_out_file = os.path.join(err_dir, "vtk_errors.txt")

    error_out = vtk.vtkFileOutputWindow()
    error_out.SetFileName(error_out_file)
    vtk_std_error_out = vtk.vtkOutputWindow()
    vtk_std_error_out.SetInstance(error_out)


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

    depth_array = numpy_to_vtk(i2.ravel(), deep=True)
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
    label_name = "C:/data/RenalArteries/DTU-Kidney-1/Slicer/KIDNEY_HEALTHY_0064_SERIES0020/" \
                 "KIDNEY_HEALTHY_0064_SERIES0020_abdominal_1.nii.gz-label.nii.gz"

    out_name_aorta = "C:/data/RenalArteries/processing/aorta_surface.vtk"
    out_name_kidney_l = "C:/data/RenalArteries/processing/kidney_left_surface.vtk"
    out_name_kidney_r = "C:/data/RenalArteries/processing/kidney_right_surface.vtk"
    out_name_renal = "C:/data/RenalArteries/processing/renal_artery_surface.vtk"

    convert_label_map_to_surface(label_name, out_name_aorta, 1)
    convert_label_map_to_surface(label_name, out_name_kidney_l, 2)
    convert_label_map_to_surface(label_name, out_name_kidney_r, 3)
    convert_label_map_to_surface(label_name, out_name_renal, 4)


def read_a_surface_and_show_some_information():
    """
    Reads a surface using VTK and show some stats
    """
    name_renal = "C:/data/RenalArteries/processing/renal_artery_surface.vtk"

    pd = vtk.vtkPolyDataReader()
    pd.SetFileName(name_renal)
    pd.Update()
    cl = pd.GetOutput()
    n_points = cl.GetNumberOfPoints()

    print(f"Surface {name_renal} has {n_points} vertices")
    p_0 = cl.GetPoint(0)


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


def preprocess_for_centerline_extraction(vtk_in):
    """
    Generate a single and smoothed surface that is good for centerline extraction
    """
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(vtk_in)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    # print("Filling holes")
    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(conn.GetOutput())
    fill_holes.SetHoleSize(1000.0)
    fill_holes.Update()

    # print("Triangle filter")
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(fill_holes.GetOutput())
    triangle.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(triangle.GetOutput())
    cleaner.Update()

    smooth_filter = vtk.vtkSmoothPolyDataFilter()
    smooth_filter.SetInputData(cleaner.GetOutput())
    smooth_filter.SetNumberOfIterations(100)
    smooth_filter.SetRelaxationFactor(0.1)
    smooth_filter.FeatureEdgeSmoothingOff()
    smooth_filter.BoundarySmoothingOn()
    smooth_filter.Update()

    # decimate = vtk.vtkDecimatePro()
    # decimate.SetInputData(smooth_filter.GetOutput())
    # decimate.SetTargetReduction(0.90)
    # decimate.PreserveTopologyOn()
    # decimate.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(smooth_filter.GetOutput())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()

def find_renal_artery_landmarks():
    label_name = "C:/data/RenalArteries/DTU-Kidney-1/Slicer/KIDNEY_HEALTHY_0064_SERIES0020/" \
                 "KIDNEY_HEALTHY_0064_SERIES0020_abdominal_1.nii.gz-label.nii.gz"
    out_name_landmark = "C:/data/RenalArteries/processing/aorta_renal_artery_landmark.txt"
    out_name_landmark_end = "C:/data/RenalArteries/processing/renal_artery_landmark_end.txt"
    renal_artery_surface_name = "C:/data/RenalArteries/processing/renal_artery_surface.vtk"
    dijkstra_out_name = "C:/data/RenalArteries/processing/renal_artery_surface_dijkstra.vtk"

    try:
        label_img = sitk.ReadImage(label_name)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {label_name}")
        return

    label_img_np = sitk.GetArrayFromImage(label_img)

    # First find intersection between a dilated aorta and the renal artery segmentation
    aorta_id = 1
    renal_artery_id = 4
    mask_np_aorta = label_img_np == aorta_id
    mask_np_ra = label_img_np == renal_artery_id
    footprint = ball(radius=2)

    print(f"Dilate aorta")
    dilated_aorta = binary_dilation(mask_np_aorta, footprint)
    print(f"Dilation over")

    overlap_mask = np.bitwise_and(dilated_aorta, mask_np_ra)

    # Find center of mask of overlap region
    com = center_of_mass(overlap_mask)

    # Do the transpose of the coordinates (SimpleITK vs. numpy)
    com_np = [com[2], com[1], com[0]]

    com_phys = label_img.TransformIndexToPhysicalPoint([int(com_np[0]), int(com_np[1]), int(com_np[2])])

    f_p_out = open(out_name_landmark, "w")
    f_p_out.write(f"{com_phys[0]} {com_phys[1]} {com_phys[2]}")
    f_p_out.close()

    # Now find opposite landmark on surface

    pd = vtk.vtkPolyDataReader()
    pd.SetFileName(renal_artery_surface_name)
    pd.Update()
    surface = pd.GetOutput()
    surface = preprocess_for_centerline_extraction(surface)

    # First find closest point on the surface (from the start point)
    start_p = com_phys

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    idx_min = locator.FindClosestPoint(start_p)
    # print(f"DEBUG: idx_min: {idx_min} start_p {start_p}")

    # Do a Dijkstra on the surface from the start point
    print("Dijkstra")
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(surface)
    dijkstra.SetStartVertex(idx_min)
    dijkstra.Update()
    weights = vtk.vtkDoubleArray()
    dijkstra.GetCumulativeWeights(weights)
    surface.GetPointData().SetScalars(weights)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(surface)
    writer.SetFileName(dijkstra_out_name)
    writer.SetFileTypeToBinary()
    writer.Write()

    # Find the vertex with the largest scaler (weight) meaning the one farthest away from the start point
    w_temp = vtk_to_numpy(weights)
    idx_max = np.argmax(w_temp)
    max_p = surface.GetPoint(idx_max)

    f_p_out = open(out_name_landmark_end, "w")
    f_p_out.write(f"{max_p[0]} {max_p[1]} {max_p[2]}")
    f_p_out.close()

    # print(f"DEBUG: idx_max: {idx_max} max_p {max_p}")


def read_landmarks(filename):
    x, y, z = 0, 0, 0
    with open(filename) as f:
        for line in f:
            if len(line) > 1:
                temp = line.split()  # Remove whitespaces and line endings and so on
                x, y, z = np.double(temp)
    return x, y, z


def read_several_landmarks(filename):
    lms = []
    with open(filename) as f:
        for line in f:
            if len(line) > 1:
                temp = line.split()  # Remove whitespaces and line endings and so on
                lm = np.double(temp)
                lms.extend(list(lm))
    return lms

def add_distances_from_landmark_to_centerline(in_center, lm_in):
    """
    Add scalar values to a center line where each value is the accumulated distance to the start point
    :param in_center: vtk center line
    :param lm_in: start landmark
    :return: centerline with scalar values
    """
    n_points = in_center.GetNumberOfPoints()

    cen_p_start = in_center.GetPoint(0)
    cen_p_end = in_center.GetPoint(n_points - 1)
    dist_start = np.linalg.norm(np.subtract(lm_in, cen_p_start))
    dist_end = np.linalg.norm(np.subtract(lm_in, cen_p_end))
    # print(f"Dist start: {dist_start} end: {dist_end}")

    start_idx = 0
    inc = 1
    # Go reverse
    if dist_start > dist_end:
        start_idx = n_points - 1
        inc = -1

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    scalars = vtk.vtkDoubleArray()
    scalars.SetNumberOfComponents(1)

    idx = start_idx
    p_1 = in_center.GetPoint(idx)
    accumulated_dist = 0
    pid = points.InsertNextPoint(p_1)
    scalars.InsertNextValue(accumulated_dist)

    while 0 < idx <= n_points:
        idx += inc
        p_2 = in_center.GetPoint(idx)
        dist = np.linalg.norm(np.subtract(p_1, p_2))
        # pid = points.InsertNextPoint(p_1)
        # scalars.InsertNextValue(accumulated_dist)
        accumulated_dist += dist
        pid_2 = points.InsertNextPoint(p_2)
        scalars.InsertNextValue(accumulated_dist)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(pid)
        lines.InsertCellPoint(pid_2)
        p_1 = p_2
        pid = pid_2

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    del points
    pd.SetLines(lines)
    del lines
    pd.GetPointData().SetScalars(scalars)
    del scalars

    return pd


def compute_renal_artery_centerline():
    # in_name_landmark = "C:/data/RenalArteries/processing/aorta_renal_artery_landmark.txt"
    # in_name_landmark_end = "C:/data/RenalArteries/processing/renal_artery_landmark_end.txt"
    # renal_artery_surface_name = "C:/data/RenalArteries/processing/renal_artery_surface.vtk"
    # renal_artery_centerline_name = "C:/data/RenalArteries/processing/renal_artery_centerline.vtk"

    in_name_landmark = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/aorta_renal_artery_landmark.txt"
    in_name_landmark_end = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_landmark_end.txt"
    renal_artery_surface_name = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_surface.vtk"
    renal_artery_centerline_name = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_centerline.vtk"

    start_p = read_landmarks(in_name_landmark)
    end_p = read_landmarks(in_name_landmark_end)

    pd = vtk.vtkPolyDataReader()
    pd.SetFileName(renal_artery_surface_name)
    pd.Update()
    surface = pd.GetOutput()
    surface = preprocess_for_centerline_extraction(surface)

    # computes the centerlines using vmtk
    centerlinePolyData = vmtkscripts.vmtkCenterlines()
    centerlinePolyData.Surface = surface
    centerlinePolyData.SeedSelectorName = "pointlist"
    centerlinePolyData.SourcePoints = start_p
    centerlinePolyData.TargetPoints = end_p
    try:
        centerlinePolyData.Execute()
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"When computing cl on {renal_artery_surface_name}")
        return False

    if centerlinePolyData.Centerlines.GetNumberOfPoints() < 10:
        print("Something wrong with centerline")
        return False

    cl_with_distances = add_distances_from_landmark_to_centerline(centerlinePolyData.Centerlines, start_p)

    writer = vtk.vtkPolyDataWriter()
    # writer.SetInputData(centerlinePolyData.Centerlines)
    writer.SetInputData(cl_with_distances)
    writer.SetFileName(renal_artery_centerline_name)
    writer.SetFileTypeToBinary()
    writer.Write()
    return True


def find_endpoint_on_dijkstra_surface():
    in_name_landmark = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/aorta_renal_artery_landmark.txt"
    end_p_out = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_landmark_end_points.txt"
    renal_artery_surface_name = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_surface.vtk"
    dijkstra_name = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_surface_dijkstra.vtk"
    # renal_artery_centerline_name = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_centerline.vtk"

    # dijkstra_name = "C:/data/RenalArteries/processing/renal_artery_surface_dijkstra.vtk"
    # end_p_out  = "C:/data/RenalArteries/processing/renal_artery_landmark_end_points.txt"

    # pd = vtk.vtkPolyDataReader()
    # pd.SetFileName(dijkstra_name)
    # pd.Update()
    pd = vtk.vtkPolyDataReader()
    pd.SetFileName(renal_artery_surface_name)
    pd.Update()
    surface = pd.GetOutput()
    surface = preprocess_for_centerline_extraction(surface)

    start_p = read_landmarks(in_name_landmark)

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    idx_min = locator.FindClosestPoint(start_p)

    print("Dijkstra")
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(surface)
    dijkstra.SetStartVertex(idx_min)
    dijkstra.Update()
    weights = vtk.vtkDoubleArray()
    dijkstra.GetCumulativeWeights(weights)
    surface.GetPointData().SetScalars(weights)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(surface)
    writer.SetFileName(dijkstra_name)
    writer.SetFileTypeToBinary()
    writer.Write()

    # surface = pd.GetOutput()
    distance_array = surface.GetPointData().GetScalars()

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()

    n_points = surface.GetNumberOfPoints()

    f_p_out = open(end_p_out, "w")

    search_radius = 5
    for idx in range(n_points):
        is_max = True
        p = surface.GetPoint(idx)
        p_val = distance_array.GetValue(idx)
        result = vtk.vtkIdList()
        locator.FindPointsWithinRadius(search_radius, p, result)
        if result.GetNumberOfIds() > 0:
            for i in range(result.GetNumberOfIds()):
                id = result.GetId(i)
                id_val = distance_array.GetValue(id)
                if id_val > p_val:
                    is_max = False
                    break
        if is_max:
            print(f"Found max at {p} with value {p_val}")
            f_p_out.write(f"{p[0]} {p[1]} {p[2]}\n")

    f_p_out.close()


def compute_renal_artery_centerlines_and_branching():
    in_name_landmark = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/aorta_renal_artery_landmark.txt"
    in_name_landmark_end = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_landmark_end_points.txt"
    renal_artery_surface_name = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_surface.vtk"
    # dijkstra_name = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_surface_dijkstra.vtk"
    renal_artery_centerline_name = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_centerline_branch_test.vtk"
    renal_artery_centerline_name_2 = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_centerline_branch_test_2.vtk"
    renal_artery_centerline_name_3 = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_centerline_branch_test_3.vtk"
    renal_artery_centerline_name_4 = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_centerline_branch_group_id.vtk"
    renal_artery_centerline_name_5 = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_centerline_branch_sphere_radius.vtk"


    # in_name_landmark = "C:/data/RenalArteries/processing/aorta_renal_artery_landmark.txt"
    # in_name_landmark_end = "C:/data/RenalArteries/processing/renal_artery_landmark_end_points.txt"
    # renal_artery_surface_name = "C:/data/RenalArteries/processing/renal_artery_surface.vtk"
    # renal_artery_centerline_name = "C:/data/RenalArteries/processing/renal_artery_centerline_branch_test.vtk"
    # renal_artery_centerline_name_2 = "C:/data/RenalArteries/processing/renal_artery_centerline_branch_test_2.vtk"
    # renal_artery_centerline_name_3 = "C:/data/RenalArteries/processing/renal_artery_centerline_branch_test_3.vtk"
    # renal_artery_centerline_name_4 = "C:/data/RenalArteries/processing/renal_artery_centerline_branch_group_id.vtk"
    # renal_artery_centerline_name_5 = "C:/data/RenalArteries/processing/renal_artery_centerline_branch_sphere_radius.vtk"

    start_p = read_landmarks(in_name_landmark)
    end_p = read_several_landmarks(in_name_landmark_end)

    pd = vtk.vtkPolyDataReader()
    pd.SetFileName(renal_artery_surface_name)
    pd.Update()
    surface = pd.GetOutput()
    surface = preprocess_for_centerline_extraction(surface)

    # computes the centerlines using vmtk
    centerlinePolyData = vmtkscripts.vmtkCenterlines()
    centerlinePolyData.Surface = surface
    centerlinePolyData.SeedSelectorName = "pointlist"
    centerlinePolyData.SourcePoints = start_p
    centerlinePolyData.TargetPoints = end_p
    try:
        centerlinePolyData.Execute()
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"When computing cl on {renal_artery_surface_name}")
        return False


    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(centerlinePolyData.Centerlines)
    writer.SetFileName(renal_artery_centerline_name)
    writer.SetFileTypeToBinary()
    writer.Write()

    # Extracting branches and merging
    print("\n--extracting branches--")
    branchExtractor = vmtkscripts.vmtkBranchExtractor()
    branchExtractor.Centerlines = centerlinePolyData.Centerlines
    branchExtractor.Execute()
    centerlines = branchExtractor.Centerlines

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(branchExtractor.Centerlines)
    writer.SetFileName(renal_artery_centerline_name_2)
    writer.SetFileTypeToBinary()
    writer.Write()

    print("\n--branch extracting done--")
    mergeCenterlines = vmtkscripts.vmtkCenterlineMerge()
    mergeCenterlines.Centerlines = centerlines
    mergeCenterlines.Execute()
    mergedCenterlines = mergeCenterlines.Centerlines

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(mergeCenterlines.Centerlines)
    writer.SetFileName(renal_artery_centerline_name_3)
    writer.SetFileTypeToBinary()
    writer.Write()

    assignAttribute = vtk.vtkAssignAttribute()
    assignAttribute.SetInputData(mergedCenterlines)
    assignAttribute.Assign(mergeCenterlines.GroupIdsArrayName, vtk.vtkDataSetAttributes.SCALARS,
                               vtk.vtkAssignAttribute.CELL_DATA)
    assignAttribute.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(assignAttribute.GetOutput())
    writer.SetFileName(renal_artery_centerline_name_4)
    writer.SetFileTypeToBinary()
    writer.Write()

    assignAttribute = vtk.vtkAssignAttribute()
    assignAttribute.SetInputData(centerlinePolyData.Centerlines)
    assignAttribute.Assign('MaximumInscribedSphereRadius', vtk.vtkDataSetAttributes.SCALARS,
                           vtk.vtkAssignAttribute.POINT_DATA)
    assignAttribute.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(assignAttribute.GetOutput())
    writer.SetFileName(renal_artery_centerline_name_5)
    writer.SetFileTypeToBinary()
    writer.Write()

    # Extract main branch
    #print("\n--extracting main branch--")
    # cellId = 0
    # assignAttribute = vtk.vtkAssignAttribute()
    # assignAttribute.SetInputData(mergedCenterlines)
    # assignAttribute.Assign(mergeCenterlines.GroupIdsArrayName, vtk.vtkDataSetAttributes.SCALARS,
    #                            vtk.vtkAssignAttribute.CELL_DATA)
    # thresholder = vtk.vtkThreshold()
    # thresholder.SetInputConnection(assignAttribute.GetOutputPort())
    # groupId = mergedCenterlines.GetCellData().GetArray(mergeCenterlines.GroupIdsArrayName).GetValue(cellId)
    # thresholder.ThresholdBetween(groupId - 0.5, groupId + 0.5)
    # thresholder.Update()
    #
    # output = thresholder.GetOutput()


    # if centerlinePolyData.Centerlines.GetNumberOfPoints() < 10:
    #     print("Something wrong with centerline")
    #     return False
    #
    # cl_with_distances = add_distances_from_landmark_to_centerline(centerlinePolyData.Centerlines, start_p)
    #
    # writer = vtk.vtkPolyDataWriter()
    # # writer.SetInputData(centerlinePolyData.Centerlines)
    # writer.SetInputData(cl_with_distances)
    # writer.SetFileName(renal_artery_centerline_name)
    # writer.SetFileTypeToBinary()
    # writer.Write()
    return True


def find_branch_points():
    renal_artery_centerline_name_5 = "C:/data/RenalArteries/processing/renal_artery_centerline_branch_group_id.vtk"
    b_point = "C:/data/RenalArteries/processing/branch_points.txt"
    pd = vtk.vtkPolyDataReader()
    pd.SetFileName(renal_artery_centerline_name_5)
    pd.Update()
    cl = pd.GetOutput()
    n_cells = cl.GetNumberOfCells()
    n_points = cl.GetNumberOfPoints()

    count_list = np.zeros(n_points)
    for i in range(n_cells):
        cell = cl.GetCell(i)
        id_list = cell.GetPointIds()
        for j in range(id_list.GetNumberOfIds()):
            count_list[id_list.GetId(j)] += 1


    f = open(b_point, "w")
    # print(count_list)
    # print(np.where(count_list > 1))
    idx = np.where(count_list > 1)[0]
    print(idx)
    for i in range(len(idx)):
        p = cl.GetPoint(idx[i])
        f.write(f"{p[0]} {p[1]} {p[2]}\n")
    f.close()


def find_branch_points_by_point_overlap():
    # renal_artery_centerline_name_5 = "C:/data/RenalArteries/processing/renal_artery_centerline_branch_group_id.vtk"
    # b_point = "C:/data/RenalArteries/processing/branch_points.txt"
    # renal_artery_centerline_name_5 = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_centerline_branch_sphere_radius.vtk"
    # renal_artery_centerline_name_5 = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_centerline_branch_sphere_radius.vtk"
    renal_artery_centerline_name_5 = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/renal_artery_centerline_branch_test.vtk"
    b_point = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/branch_points.txt"
    b_point_2 = "C:/data/RenalArteries/Victoria-16-11-2023/less_good_case/branch_points_2.txt"
    pd = vtk.vtkPolyDataReader()
    pd.SetFileName(renal_artery_centerline_name_5)
    pd.Update()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(pd.GetOutput())
    cleaner.Update()

    cl = cleaner.GetOutput()
    # cl = pd.GetOutput()
    n_cells = cl.GetNumberOfCells()
    n_points = cl.GetNumberOfPoints()
    n_points_org = pd.GetOutput().GetNumberOfPoints()
    n_cells_org = pd.GetOutput().GetNumberOfCells()

    f = open(b_point, "w")

    stored_points = set()
    for i in range(n_points):
        p = cl.GetPoint(i)
        if (p[0], p[1], p[2]) in stored_points:
            print("duplicate point")
            print(p)
        else:
            stored_points.add((p[0], p[1], p[2]))
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    f.close()

    print("Second method")

    #
    #
    count_list = np.zeros(n_points)
    for i in range(n_cells):
        cell = cl.GetCell(i)
        id_list = cell.GetPointIds()
        for j in range(id_list.GetNumberOfIds()):
            count_list[id_list.GetId(j)] += 1
    #
    #
    f = open(b_point_2, "w")
    # # print(count_list)
    # # print(np.where(count_list > 1))
    idx = np.where(count_list > 1)[0]
    # print(idx)
    for i in range(len(idx)):
        p = cl.GetPoint(idx[i])
        print("duplicate point")
        print(p)
        f.write(f"{p[0]} {p[1]} {p[2]}\n")
    f.close()


def combine_aorta_iliac_arteries():
    segment_file = "C:/data/RenalArteries/aortabranch/DTU_001/segmentations/total/total.nii.gz"
    combined_segment_out = "C:/data/RenalArteries/aortabranch/DTU_001/segmentations/aorta_iliac_combined.nii.gz"
    out_combined_surface = "C:/data/RenalArteries/aortabranch/DTU_001/segmentations/aorta_iliac_combined_surface.vtk"

    out_name_aorta = "C:/data/RenalArteries/processing/aorta_surface.vtk"
    out_name_kidney_l = "C:/data/RenalArteries/processing/kidney_left_surface.vtk"
    out_name_kidney_r = "C:/data/RenalArteries/processing/kidney_right_surface.vtk"
    out_name_renal = "C:/data/RenalArteries/processing/renal_artery_surface.vtk"


    try:
        label_img = sitk.ReadImage(segment_file)
    except RuntimeError as e:
        print(f"Error reading {segment_file})")
        return

    aorta_segm_id = 52
    iliac_left_segm_id = 65
    iliac_right_segm_id = 66
    label_img_np = sitk.GetArrayFromImage(label_img)
    mask_np_aorta = label_img_np == aorta_segm_id
    masp_np_iliac_l = label_img_np == iliac_left_segm_id
    mask_np_ilac_r = label_img_np == iliac_right_segm_id

    mask_np = np.bitwise_or(mask_np_aorta, masp_np_iliac_l)
    mask_np = np.bitwise_or(mask_np, mask_np_ilac_r)
    mask_np = binary_fill_holes(mask_np)
    mask_np = binary_closing(mask_np, ball(radius=2))

    img_o = sitk.GetImageFromArray(mask_np.astype(int))
    img_o.CopyInformation(label_img)

    print(f"Debug: saving {combined_segment_out}")
    sitk.WriteImage(img_o, combined_segment_out)

    print(f"Writing surface {out_combined_surface}")
    convert_label_map_to_surface(combined_segment_out, out_combined_surface, 1)

    # convert_label_map_to_surface(label_name, out_name_aorta, 1)
    # convert_label_map_to_surface(label_name, out_name_kidney_l, 2)
    # convert_label_map_to_surface(label_name, out_name_kidney_r, 3)
    # convert_label_map_to_surface(label_name, out_name_renal, 4)


def find_iliac_endpoints_on_dijkstra_surface():
    in_name_landmark = "C:/data/RenalArteries/aortabranch/DTU_001/landmarks/aorta_end_point.txt"
    end_p_out = "C:/data/RenalArteries/aortabranch/DTU_001/landmarks/iliac_end_points.txt"
    surface_name = "C:/data/RenalArteries/aortabranch/DTU_001/segmentations/aorta_iliac_combined_surface.vtk"
    dijkstra_name = "C:/data/RenalArteries/aortabranch/DTU_001/segmentations/aorta_iliac_combined_surface_dijkstra.vtk"
    centerline_name = "C:/data/RenalArteries/aortabranch/DTU_001/segmentations/aorta_iliac_combined_centerline.vtk"

    pd = vtk.vtkPolyDataReader()
    pd.SetFileName(surface_name)
    pd.Update()
    surface = pd.GetOutput()
    surface = preprocess_for_centerline_extraction(surface)

    start_p = read_landmarks(in_name_landmark)

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    idx_min = locator.FindClosestPoint(start_p)

    print("Dijkstra")
    dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
    dijkstra.SetInputData(surface)
    dijkstra.SetStartVertex(idx_min)
    dijkstra.Update()
    weights = vtk.vtkDoubleArray()
    dijkstra.GetCumulativeWeights(weights)
    surface.GetPointData().SetScalars(weights)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(surface)
    writer.SetFileName(dijkstra_name)
    writer.SetFileTypeToBinary()
    writer.Write()

    # surface = pd.GetOutput()
    distance_array = surface.GetPointData().GetScalars()

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()

    n_points = surface.GetNumberOfPoints()

    f_p_out = open(end_p_out, "w")

    search_radius = 5
    min_dist = 40
    for idx in range(n_points):
        is_max = True
        p = surface.GetPoint(idx)
        p_val = distance_array.GetValue(idx)
        if p_val > min_dist:
            result = vtk.vtkIdList()
            locator.FindPointsWithinRadius(search_radius, p, result)
            if result.GetNumberOfIds() > 0:
                for i in range(result.GetNumberOfIds()):
                    id = result.GetId(i)
                    id_val = distance_array.GetValue(id)
                    if id_val > p_val:
                        is_max = False
                        break
            if is_max:
                print(f"Found max at {p} with value {p_val}")
                f_p_out.write(f"{p[0]} {p[1]} {p[2]}\n")

    f_p_out.close()

    end_p = read_several_landmarks(end_p_out)

    print(f"Computing centerline")
    # computes the centerlines using vmtk
    centerlinePolyData = vmtkscripts.vmtkCenterlines()
    centerlinePolyData.Surface = surface
    centerlinePolyData.SeedSelectorName = "pointlist"
    centerlinePolyData.SourcePoints = start_p
    centerlinePolyData.TargetPoints = end_p
    try:
        centerlinePolyData.Execute()
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        return

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(centerlinePolyData.Centerlines)
    writer.SetFileName(centerline_name)
    writer.SetFileTypeToBinary()
    writer.Write()




def do_kidney_things():
    process_dir = "C:/data/RenalArteries/processing/"
    setup_vtk_error_handling(process_dir)
    # read_an_image_and_sample_a_pixel_value()
    # extract_segmentations_as_surfaces()
    # read_an_image_change_in_numpy_and_save_again()
    # read_a_surface_and_show_some_information()
    # find_renal_artery_landmarks()
    # compute_renal_artery_centerline()
    # find_endpoint_on_dijkstra_surface()
    compute_renal_artery_centerlines_and_branching()
    # find_branch_points()
    # find_branch_points_by_point_overlap()

def do_aorta_things():
    process_dir = "C:/data/RenalArteries/processing/"
    setup_vtk_error_handling(process_dir)
    # combine_aorta_iliac_arteries()
    find_iliac_endpoints_on_dijkstra_surface()

if __name__ == "__main__":
    # do_kidney_things()
    do_aorta_things()
