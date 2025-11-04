import itk
from awkward.index import numpy
from itk import RTK as rtk
import numpy as np
import os
import vtk
from volume import VolumeClass, extract_index
from projections import ProjectionsClass, read_geometry, write_geometry
from surface import VolumeToSurface
import tempfile
from multiprocessing.pool import Pool
import gatetools as gt
import time
from matplotlib import pyplot as plt
from numpy.linalg import multi_dot


import torch
# import torchvision
import torch.nn.functional as F


eps = 1e-8

def conversion_factor_vectorized(volume,kregion, geometry, mu0):
    volume = volume.permute(0,2,1,3) #(1,Ny,Nx,Nz)
    kregion = kregion.permute(0,2,1,3)
    dev = volume.device
    Ny,Nx,Nz = volume.shape[1], volume.shape[2], volume.shape[3]

    spacing = 2.3976

    list_GantryAngles = list(geometry.GetGantryAngles())
    list_OutOfPlaneAngles = list(geometry.GetOutOfPlaneAngles())
    list_InPlaneAngles = list(geometry.GetInPlaneAngles())
    list_ProjectioOffsetX = list(geometry.GetProjectionOffsetsX())
    list_ProjectioOffsetY = list(geometry.GetProjectionOffsetsY())

    tensor_GantryAngles = torch.Tensor(list_GantryAngles).to(dev)
    tensor_OutOfPlaneAngles = torch.Tensor(list_OutOfPlaneAngles).to(dev)
    tensor_InPlaneAngles = torch.Tensor(list_InPlaneAngles).to(dev)
    tensor_ProjectionOffsetX = torch.Tensor(list_ProjectioOffsetX).to(dev)
    tensor_ProjectionOffsetY = torch.Tensor(list_ProjectioOffsetY).to(dev)

    Nangles = len(list_GantryAngles)

    A = torch.zeros((Nangles,3,4)).to(dev)
    for i in range(Nangles):
        gantry_angle = tensor_GantryAngles[i]
        out_of_plane_angle = tensor_OutOfPlaneAngles[i]
        in_plane_angle = tensor_InPlaneAngles[i]
        proj_offset_x,proj_offset_y = tensor_ProjectionOffsetX[i],tensor_ProjectionOffsetY[i]
        def rot_x(theta):  # out-of-plane
            c, s = torch.cos(theta), torch.sin(theta)
            return torch.tensor([[1, 0, 0],
                                 [0, c, -s],
                                 [0, s, c]], device=dev, dtype=volume.dtype)

        def rot_y(theta):  # gantry
            c, s = torch.cos(theta), torch.sin(theta)
            return torch.tensor([[c, 0, s],
                                 [0, 1, 0],
                                 [-s, 0, c]], device=dev, dtype=volume.dtype)

        def rot_z(theta):  # in-plane
            c, s = torch.cos(theta), torch.sin(theta)
            return torch.tensor([[c, -s, 0],
                                 [s, c, 0],
                                 [0, 0, 1]], device=dev, dtype=volume.dtype)

        R = rot_z(torch.Tensor(-gantry_angle)) @ \
            rot_x(torch.Tensor(-out_of_plane_angle)) @ \
            rot_y(torch.Tensor(-in_plane_angle))

        max_dim = max(volume.shape[1], volume.shape[2], volume.shape[3])
        scale_x = volume.shape[1] / max_dim
        scale_y = volume.shape[2] / max_dim
        scale_z = volume.shape[3] / max_dim

        S_aspect = torch.diag(torch.tensor([scale_x, scale_y, scale_z], device=dev, dtype=volume.dtype))
        S_aspect_inv = torch.diag(
            torch.tensor([1 / scale_x, 1 / scale_y, 1 / scale_z], device=dev, dtype=volume.dtype))

        R = S_aspect_inv @ R @ S_aspect

        # Add translation
        T = torch.tensor([[2 * proj_offset_x / spacing / (Nz - 1)],
                          [0.],  # 0 for sure
                          [2 * proj_offset_y / spacing / (Ny - 1)]], device=dev, dtype=volume.dtype)
        T = (R @ T)

        # Build affine 3x4
        A[i,:,:] = torch.cat([R, T], dim=1)

    batch_size = 5

    x_idx = torch.arange(Nx, device=dev).view(1, 1, Nx, 1)
    conversion_factor_tensor = torch.zeros((Nangles,Ny,Nz),device=dev)
    for k in range(Nangles//batch_size):
        volume_batch = volume.expand(batch_size,-1,-1,-1,-1)  # (120,1,Ny,Nx,Nz)
        kregion_batch = kregion.expand(batch_size,-1,-1,-1,-1)

        grid = F.affine_grid(A[batch_size*k:batch_size*k+batch_size,:,:], size=volume_batch.shape, align_corners=True)

        volumes_rotated = F.grid_sample(torch.cat((volume_batch,kregion_batch),dim=1),grid,
                                        mode="bilinear", padding_mode="zeros", align_corners=True)

        volume_batch_rotated = volumes_rotated[:,0,:,:,:]
        kregion_batch_rotated = volumes_rotated[:,1,:,:,:]

        tau_i = torch.argmax((kregion_batch_rotated>0).float(),dim=2)
        tau_mm = -(tau_i-Nx/2)*spacing # (120, Ny, Nz)

        mask_lines_to_project = (x_idx < tau_i.unsqueeze(2)).to(kregion_batch_rotated.dtype)
        conversion_factor_batch = torch.exp(mu0*tau_mm+
                                            (volume_batch_rotated*mask_lines_to_project*spacing).sum(2))

        conversion_factor_batch[tau_i==0]=0

        conversion_factor_tensor[batch_size*k:batch_size*k+batch_size,:,:] = conversion_factor_batch

        del conversion_factor_batch,mask_lines_to_project,tau_mm,tau_i,kregion_batch_rotated,volume_batch_rotated,
        del volumes_rotated,grid,kregion_batch,volume_batch


    return conversion_factor_tensor



class ExponentialProjectionsGeometryClass(object):
    """
    Class that allows to compute and store the data needed to convert the attenuated projection
    to exponential projection.

    :param attenuation_map: Attenuation map.
    :type attenuation_map: Union[itk.itkImagePython, str, VolumeClass]
    :param geometry: Geometry of the acquisition.
    :type geometry: itk.rtkThreeDCircularProjectionGeometryPython.rtkThreeDCircularProjectionGeometry
    :param like: Copy information from this image (origin, size, spacing, direction).
    :type like: Union[str, itk.itkImagePython, VolumeClass], optional
    :param origin_projections: Origin of the projections. Default = centered.
    :type origin_projections: Union[list, numpy.ndarray], optional
    :param spacing_projections: Spacing of the projections. Default = [1, 1, 1].
    :type spacing_projections: Union[list, numpy.ndarray], optional
    :param size_projections: Size of the projections. Default = [128, 128, 128].
    :type size_projections: Union[list, numpy.ndarray], optional
    :param direction_projections: Direction of the projections. Default = identity.
    :type direction_projections: numpy.ndarray, optional
    :param ellipse_center: Center coordinate of the ellipse (mm).
    :type ellipse_center: list, optional
    :param ellipse_axis: Size of the axis of the ellipse (mm).
    :type ellipse_axis: list, optional
    :param voxelized_region: Voxelized K region.
    :type voxelized_region: VolumeClass
    """

    def __init__(self, attenuation_map, geometry, like=None, origin_projections=None, spacing_projections=None,
                 size_projections=None, direction_projections=None, ellipse_center=None, ellipse_axis=None,
                 voxelized_region=None):
        if not isinstance(attenuation_map, VolumeClass):
            self.volume_attenuation_map = VolumeClass(attenuation_map)
            self.itk_attenuation_map = self.volume_attenuation_map.itk_image
        else:
            self.volume_attenuation_map = attenuation_map
            self.itk_attenuation_map = self.volume_attenuation_map.itk_image
        if not isinstance(geometry, itk.rtkThreeDCircularProjectionGeometryPython.rtkThreeDCircularProjectionGeometry):
            raise TypeError("Geometry must be of type "
                            "itk.rtkThreeDCircularProjectionGeometryPython.rtkThreeDCircularProjectionGeometry")
        self.geometry = geometry
        if like is not None:
            if not isinstance(like, VolumeClass):
                like = VolumeClass(like)
            like.get_physical_coordinates()
            self.origin = like.origin
            self.size = like.size
            self.spacing = like.spacing
            self.direction = like.direction
            self.__matrix_index_to_pp_projections = like.get_index_to_physical_point_matrix()
        else:
            self.set_projections_coordinates(like, origin_projections, spacing_projections, size_projections,
                                             direction_projections)
        if ellipse_center is not None and not isinstance(ellipse_center, list):
            raise TypeError("Ellipse center should be a list.")
        if ellipse_axis is not None and not isinstance(ellipse_axis, list):
            raise TypeError("Ellipse axis should be a list.")
        if voxelized_region is not None and not isinstance(voxelized_region, VolumeClass):
            raise TypeError(
                "Wrong type for voxelized_region {}, must be of type VolumeClass".format(type(voxelized_region)))
        self.ellipse_center = ellipse_center
        self.ellipse_axis = ellipse_axis
        self.voxelized_region = voxelized_region
        if self.ellipse_center is not None and self.ellipse_axis is not None:
            self.__k_region_type = "ellipse"
            self.__matrix_pp_to_index_voxelized_region = None
            self.__matrix_index_to_pp_voxelized_region = None
        elif self.voxelized_region is not None:
            self.__k_region_type = "voxelized"
            self.__matrix_pp_to_index_voxelized_region = self.voxelized_region.get_physical_point_to_index_matrix()
            self.__matrix_index_to_pp_voxelized_region = self.voxelized_region.get_index_to_physical_point_matrix()
        else:
            self.__k_region_type = None
            self.__matrix_pp_to_index_voxelized_region = None
            self.__matrix_index_to_pp_voxelized_region = None
        self.conversion_factor = None
        self.mu_zero = None

    def compute_mu_zero(self):
        """
        Compute the mu zero value inside the K region.
        """
        if self.__k_region_type == "ellipse":
            k_region = rtk.constant_image_source(information_from_image=self.itk_attenuation_map,
                                                 ttype=(itk.Image[itk.F, 3]))
            k_region = rtk.draw_ellipsoid_image_filter(k_region, center=self.ellipse_center, axis=self.ellipse_axis,
                                                       density=1)
        elif self.__k_region_type == "voxelized":
            k_region = self.voxelized_region.itk_image
        else:
            raise ValueError("K region should be set before computing the mu zero value.")
        mask = itk.binary_threshold_image_filter(k_region, lower_threshold=1e-10, outside_value=0, inside_value=1)
        mask = itk.cast_image_filter(mask, ttype=(itk.Image[itk.F, 3], itk.Image[itk.UC, 3]))
        stats = itk.LabelStatisticsImageFilter[itk.Image[itk.F, 3], itk.Image[itk.UC, 3]].New()
        stats.SetInput(self.itk_attenuation_map)
        stats.SetLabelInput(mask)
        stats.Update()
        self.mu_zero = stats.GetMean(1)

    def set_k_region_from_ellipse(self, ellipse_center, ellipse_axis):
        """
        Set the K region from ellipse parameters.

        :param ellipse_center: Center coordinate of the ellipse (mm)
        :type ellipse_center: list
        :param ellipse_axis: Size of the axis of the ellipse (mm)
        :type ellipse_axis: list
        """
        if not isinstance(ellipse_center, list):
            raise TypeError("Ellipse center should be a list.")
        if not isinstance(ellipse_axis, list):
            raise TypeError("Ellipse axis should be a list.")
        self.ellipse_center = ellipse_center
        self.ellipse_axis = ellipse_axis
        self.__k_region_type = "ellipse"
        self.mu_zero = None

    def set_k_region_from_voxelized_volume(self, voxelized_region):
        """
        Set the K region from voxelized volume.

        :param voxelized_region: Voxelized K region.
        :type voxelized_region: VolumeClass
        """
        if voxelized_region is not None and not isinstance(voxelized_region, VolumeClass):
            raise TypeError(
                "Wrong type for voxelized_region {}, must be of type VolumeClass".format(type(voxelized_region)))
        self.voxelized_region = voxelized_region
        self.__matrix_pp_to_index_voxelized_region = self.voxelized_region.get_physical_point_to_index_matrix()
        self.__matrix_index_to_pp_voxelized_region = self.voxelized_region.get_index_to_physical_point_matrix()
        self.__k_region_type = "voxelized"
        self.mu_zero = None

    def set_projections_coordinates(self, like=None, origin=None, spacing=None, size=None, direction=None):
        """
        Set the origin, spacing, size and direction of the projections needed to compute the conversion factor.

        :param like: Copy information from this image (origin, size, spacing, direction)
        :type like: Union[str, itk.itkImagePython, VolumeClass], optional
        :param origin: Origin of the projections. Default = centered
        :type origin: Union[list, numpy.ndarray], optional
        :param spacing: Spacing of the projections. Default = [1, 1, 1]
        :type spacing: Union[list, numpy.ndarray], optional
        :param size: Size of the projections. Default = [128, 128, 128]
        :type size: Union[list, numpy.ndarray], optional
        :param direction: Direction of the projections. Default = identity
        :type direction: numpy.ndarray, optional
        """
        if like is not None:
            if not isinstance(like, VolumeClass):
                like = VolumeClass(like)
            like.get_physical_coordinates()
            self.origin = like.origin
            self.size = like.size
            self.spacing = like.spacing
            self.direction = like.direction
            self.__matrix_index_to_pp_projections = like.get_index_to_physical_point_matrix()
        else:
            if size is None:
                self.size = [128, 128, 128]
            else:
                self.size = size
            if spacing is None:
                self.spacing = [1, 1, 1]
            else:
                self.spacing = spacing
            if origin is None:
                self.origin = [-0.5 * (self.size[i] - 1) * self.spacing[i] for i in range(3)]
            else:
                self.origin = origin
            if direction is None:
                self.direction = np.eye(3, 3)
            else:
                self.direction = direction
            kwargs_proj_source_constant_zero = {"ttype": [itk.Image[itk.F, 3]], "size": self.size,
                                                "spacing": self.spacing,
                                                "origin": self.origin,
                                                "direction": itk.GetMatrixFromArray(self.direction)}
            proj = ProjectionsClass(rtk.constant_image_source(**kwargs_proj_source_constant_zero),
                                    self.geometry)
            self.__matrix_index_to_pp_projections = proj.get_index_to_physical_point_matrix()

    def compute_conversion_factor(self):
        """
        Compute the conversion factor needed to obtain the exponential projections from the attenuated projections.
        """
        if self.origin is None or self.spacing is None or self.size is None or self.direction is None:
            raise ValueError(
                "Origin, spacing and size of the projections should be set before computing the conversion factor.")
        if self.mu_zero is None:
            self.compute_mu_zero()
        if self.__k_region_type == "ellipse":
            self.__compute_conversion_factor_from_ellipse()
        elif self.__k_region_type == "voxelized":
            self.__compute_conversion_factor_from_voxelized_region()

    def __compute_conversion_factor_from_ellipse(self):
        """
        Compute the conversion factor needed to obtain the exponential projections from the attenuated projections.
        """
        att_phantom = np.zeros((self.size[2], self.size[1], self.size[0]))
        kwargs_one_proj_source = {"ttype": [itk.Image[itk.F, 3]], "size": [self.size[0], self.size[1], 1],
                                  "spacing": [self.spacing[0], self.spacing[1], 1],
                                  "origin": [self.origin[0], self.origin[1], 0],
                                  "direction": itk.GetMatrixFromArray(self.direction)}
        for i in range(self.size[2]):
            geometry_proj = rtk.ThreeDCircularProjectionGeometry.New()
            geometry_proj.AddProjectionInRadians(self.geometry.GetSourceToIsocenterDistances()[i],
                                                 self.geometry.GetSourceToDetectorDistances()[i],
                                                 self.geometry.GetGantryAngles()[i])
            dir_det = itk.GetArrayFromMatrix(geometry_proj.GetRotationMatrix(0))[2, :3]
            # Compute clip plane parameters v and r for the ellipse.
            # They are determined by finding the two tangential points of the ellipse in
            # the direction orthogonal to the detector.
            if np.abs(dir_det[0]) < 0.00001:
                x1 = self.ellipse_axis[0]
                x2 = -x1
                z1 = 0
                z2 = 0
            else:
                slope = dir_det[2] / dir_det[0]
                x1 = self.ellipse_axis[0] ** 2 * slope / np.sqrt(
                    self.ellipse_axis[0] ** 2 * slope ** 2 + self.ellipse_axis[2] ** 2)
                z1 = -x1 * self.ellipse_axis[2] ** 2 / (slope * self.ellipse_axis[0] ** 2)
                x2 = -1 * x1
                z2 = -x2 * self.ellipse_axis[2] ** 2 / (slope * self.ellipse_axis[0] ** 2)
            x1 += self.ellipse_center[0]
            z1 += self.ellipse_center[2]
            x2 += self.ellipse_center[0]
            z2 += self.ellipse_center[2]
            v = np.array([z2 - z1, 0, x1 - x2])
            v = np.divide(v, np.linalg.norm(v) * np.sign(np.dot(dir_det, v)), out=v, casting='unsafe')
            v = v.astype(float)
            r = np.dot(v, [x1, 0, z1])
            # First compute the distance detector to clip plane
            source = rtk.constant_image_source(**kwargs_one_proj_source)
            det_to_clip = rtk.RayEllipsoidIntersectionImageFilter[type(source), type(source)].New(source,
                                                                                                  center=[0.] * 3,
                                                                                                  axis=[1.e5] * 3,
                                                                                                  geometry=geometry_proj)
            det_to_clip.AddClipPlane(v, r)
            det_to_clip.AddClipPlane(-dir_det, self.geometry.GetSourceToIsocenterDistances()[i])
            det_to_clip.Update()
            det_to_clip = itk.GetArrayFromImage(det_to_clip.GetOutput())
            cent_to_clip = - det_to_clip + self.geometry.GetSourceToIsocenterDistances()[i]
            # Then the clip plane to ellipse exit
            source = rtk.constant_image_source(**kwargs_one_proj_source)
            clip_to_ellipse = rtk.RayEllipsoidIntersectionImageFilter[type(source), type(source)].New(source,
                                                                                                      center=self.ellipse_center,
                                                                                                      axis=self.ellipse_axis,
                                                                                                      geometry=geometry_proj)
            clip_to_ellipse.AddClipPlane(v, r)
            clip_to_ellipse.Update()
            clip_to_ellipse = itk.GetArrayFromImage(clip_to_ellipse.GetOutput())
            cent_to_ellipse = np.add(cent_to_clip, clip_to_ellipse, where=clip_to_ellipse != 0,
                                     out=np.zeros_like(cent_to_clip))
            # And now the attenuation phantom
            source = rtk.constant_image_source(**kwargs_one_proj_source)
            attenuation_clipped = rtk.constant_image_source(information_from_image=self.itk_attenuation_map,
                                                            ttype=(itk.Image[itk.F, 3]))
            attenuation_clipped = rtk.DrawGeometricPhantomImageFilter.New(input=attenuation_clipped,
                                                                          is_forbild_config_file=True,
                                                                          config_file=os.path.join(
                                                                              os.path.dirname(__file__),
                                                                              "data/Rectangle.txt"))
            attenuation_clipped.AddClipPlane(v, r)
            attenuation_clipped.Update()
            ellipse_source = rtk.constant_image_source(information_from_image=self.itk_attenuation_map,
                                                       ttype=(itk.Image[itk.F, 3]))
            ellipse_source = rtk.DrawEllipsoidImageFilter.New(input=ellipse_source, center=self.ellipse_center,
                                                              axis=self.ellipse_axis,
                                                              density=1)

            ellipse_source.AddClipPlane(v, r)
            ellipse_source.Update()
            attenuation_clipped = itk.subtract_image_filter(attenuation_clipped, ellipse_source)
            attenuation_clipped = itk.multiply_image_filter(self.itk_attenuation_map, attenuation_clipped)
            joseph = rtk.JosephForwardProjectionImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
            joseph.SetGeometry(geometry_proj)
            joseph.SetInput(0, source)
            joseph.SetInput(1, attenuation_clipped)
            joseph.Update()
            phantom = itk.GetArrayFromImage(joseph.GetOutput()).squeeze()

            #     # See Natterer page 47
            att_phantom[i, :, :] = phantom + cent_to_ellipse * self.mu_zero
        self.conversion_factor = VolumeClass(np.exp(att_phantom), self.spacing, self.origin, self.direction)

    def __compute_conversion_factor_from_voxelized_region(self):
        volume_to_surface = VolumeToSurface(self.voxelized_region)
        surface = volume_to_surface.extract_surface()
        kwargs_proj_source_constant_zero = {"ttype": [itk.Image[itk.F, 3]], "size": self.size,
                                            "spacing": self.spacing,
                                            "origin": self.origin,
                                            "direction": itk.GetMatrixFromArray(self.direction)}
        kwargs_proj_source_constant_one = {"ttype": [itk.Image[itk.F, 3]], "size": self.size,
                                           "spacing": self.spacing,
                                           "origin": self.origin,
                                           "direction": itk.GetMatrixFromArray(self.direction),
                                           "constant": 1}
        cent_to_ellipse = ProjectionsClass(rtk.constant_image_source(**kwargs_proj_source_constant_zero), self.geometry)
        inferior_clip = ProjectionsClass(rtk.constant_image_source(**kwargs_proj_source_constant_one), self.geometry)
        obb_tree = vtk.vtkOBBTree()
        obb_tree.SetDataSet(surface.surface)
        obb_tree.BuildLocator()
        index_angle = range(self.size[2])
        timeLoop = time.time()
        for i in index_angle:
            current_angle = self.geometry.GetGantryAngles()[i]
            outPlane_angle = self.geometry.GetOutOfPlaneAngles()[i]
            dir_det = itk.GetArrayFromMatrix(self.geometry.GetRotationMatrix(i))[2, :3]
            dir_det *= 2 * self.geometry.GetSourceToIsocenterDistances()[i]
            projection_index_transform_matrix = np.dot(
                np.dot(self.__matrix_pp_to_index_voxelized_region, itk.GetArrayFromMatrix(
                    self.geometry.GetProjectionCoordinatesToFixedSystemMatrix(i))),
                self.__matrix_index_to_pp_projections)
            # Precompute some values outside the loop
            matrix_index_to_pp_voxelized_region = self.__matrix_index_to_pp_voxelized_region

            # Precompute the fixed part of pixel_position
            fixed_pixel_position = projection_index_transform_matrix[:, 3]
            for n in range(self.size[0]):
                for m in range(self.size[1]):
                    pixel_position = np.zeros(3)
                    position_index = [n, m, i]
                    for k in range(3):
                        pixel_position[k] = projection_index_transform_matrix[k][3]
                        for j in range(3):
                            pixel_position[k] += projection_index_transform_matrix[k][j] * position_index[j]
                    pixel_position = np.dot(self.__matrix_index_to_pp_voxelized_region, np.append(pixel_position, 1))

                    p_source = pixel_position[:3]
                    p_target = pixel_position[:3] + dir_det
                    dist_source_target = np.linalg.norm(p_target - p_source)

                    dist_source_target =  2 * self.geometry.GetSourceToIsocenterDistances()[i]
                    points_vtk_intersection = vtk.vtkPoints()
                    obb_tree.IntersectWithLine(p_source, p_target, points_vtk_intersection, None)
                    points_vtk_intersection_data = points_vtk_intersection.GetData()
                    nb_points_vtk_intersection = points_vtk_intersection_data.GetNumberOfTuples()
                    points_intersection = []
                    list_dist_source_intersection = []
                    for idx in range(nb_points_vtk_intersection):
                        _tup = points_vtk_intersection_data.GetTuple3(idx)
                        points_intersection.append(_tup)
                        list_dist_source_intersection.append(np.linalg.norm(np.array(_tup) - p_source))
                    if points_intersection:
                        idx_interect = np.argmin(list_dist_source_intersection)
                        dist_source_intersection = np.min(list_dist_source_intersection)
                        tau = -np.cos(outPlane_angle) * np.sin(current_angle) * points_intersection[idx_interect][
                            0] - np.cos(outPlane_angle) * np.cos(current_angle) * \
                            points_intersection[idx_interect][2] + np.sin(outPlane_angle) * \
                            points_intersection[idx_interect][1]
                        cent_to_ellipse[i, position_index[1], position_index[0]] = tau
                        inferior_clip[i, m, n] = 1 - dist_source_intersection / dist_source_target
        source = rtk.constant_image_source(**kwargs_proj_source_constant_zero)
        itk_inferior_clip_double = itk.GetImageFromArray(inferior_clip.numpy_image.astype(np.double))
        # itk.imwrite(itk_inferior_clip_double, "inferior_clip.mha")
        itk_inferior_clip_double.CopyInformation(inferior_clip.itk_image)
        joseph = rtk.JosephForwardProjectionImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
        joseph.SetGeometry(self.geometry)
        joseph.SetInput(0, source)
        joseph.SetInput(1, self.itk_attenuation_map)
        joseph.SetInferiorClipImage(itk_inferior_clip_double)
        joseph.Update()

        phantom = itk.GetArrayFromImage(joseph.GetOutput())
        att_phantom = phantom + cent_to_ellipse.numpy_image * self.mu_zero
        self.conversion_factor = VolumeClass(np.exp(att_phantom), self.spacing, self.origin, self.direction)

    def get_conversion_factor(self):
        """
        Get the conversion factor as a VolumeClass.

        :return: Conversion factor.
        :rtype: VolumeClass
        """
        if self.conversion_factor is None:
            self.compute_conversion_factor()
        return self.conversion_factor

    def get_mu_zero(self):
        """
        Get the conversion factor as a VolumeClass.

        :return: Conversion factor.
        :rtype: float
        """
        if self.mu_zero is None:
            self.compute_mu_zero()
        return self.mu_zero

    def save_conversion_factor(self, output_path):
        """
        Save the conversion factor as an ITK image.

        :param output_path: Path where to save the conversion factor
        :type output_path: str
        """
        if self.conversion_factor is None:
            self.compute_conversion_factor()
        self.conversion_factor.save(output_path)

    def read_conversion_factor(self, path):
        """
        Read the conversion factor from a file.

        :param path: Path to the conversion factor
        :type path: str
        """
        self.conversion_factor = VolumeClass(path)


class ExponentialProjectionsClass(ProjectionsClass):
    """
    Class to handle exponential projections. This class allows to compute the exponential projections from the
    attenuated projections and to compute the exponential data consistency condition (eDCC).

    :param projections: SPECT projections
    :type projections: ProjectionsClass
    :param exponential_geometry: ExponentialProjectionGeometryClass used to create exponential projections.
    :type exponential_geometry: ExponentialProjectionsGeometryClass
    """

    def __init__(self, projections, exponential_geometry):
        if isinstance(projections, ProjectionsClass):
            self.projections = projections
        else:
            raise TypeError("projections must be a ProjectionsClass")
        if isinstance(exponential_geometry, ExponentialProjectionsGeometryClass):
            self.exponential_geometry = exponential_geometry
        else:
            raise TypeError("exponential_geometry must be a ExponentialProjectionsGeometryClass")
        self.conversion_factor = self.exponential_geometry.get_conversion_factor()
        self.mu_zero = self.exponential_geometry.get_mu_zero()
        numpy_exponential_projections = self.projections.numpy_image * self.conversion_factor.numpy_image
        itk_exponential_projections = itk.GetImageFromArray(numpy_exponential_projections.astype(np.float32))
        itk_exponential_projections.CopyInformation(self.projections.itk_image)
        super(ExponentialProjectionsClass, self).__init__(itk_exponential_projections, self.projections.geometry,
                                                          attenuation_map=self.exponential_geometry.volume_attenuation_map)
        self.edcc = []
        self.variance = []

    def set_exponential_geometry(self, exponential_geometry):
        """
        Set the exponential geometry used to convert attenuated projections to exponential projections.

        :param exponential_geometry: ExponentialProjectionGeometryClass used to create exponential projections
        :type exponential_geometry: ExponentialProjectionsGeometryClass
        """
        if isinstance(exponential_geometry, ExponentialProjectionsGeometryClass):
            self.exponential_geometry = exponential_geometry
        else:
            raise TypeError("exponential_geometry must be a ExponentialProjectionsGeometryClass")
        self.conversion_factor = self.exponential_geometry.get_conversion_factor()
        self.mu_zero = self.exponential_geometry.mu_zero

    def compute_exponential_projection(self):
        """
        Compute the exponential projections based on the exponential geometry.
        """
        if self.exponential_geometry is None:
            raise ValueError("Exponential geometry must be set or created before computing the exponential projections")
        self.numpy_image = self.projections.numpy_image * self.conversion_factor.numpy_image
        self.itk_image = itk.GetImageFromArray(self.numpy_image.astype(np.float32))
        self.itk_image.CopyInformation(self.projections.itk_image)

    def compute_variance_edcc(self, ind_angle, sigma, em_slice):
        """
        Compute the variance of the exponential data consistency conditions eDCC for a given angle and line

        :param ind_angle: Index of the angle of the projection
        :type ind_angle: int
        :param sigma: Point of the Laplace transform where the eDCC are defined
        :type sigma: float
        :param em_slice: Line on which is computed the eDCC
        :type em_slice: int
        :return: Variance of the eDCC for the angle and line
        :rtype: float
        """
        var = (self.s[1] - self.s[0]) ** 2 * np.sum(
            self.conversion_factor.numpy_image[ind_angle, em_slice, :] ** 2 * self.projections.numpy_image[ind_angle,
                                                                              em_slice, :]
            * np.exp(2 * sigma * self.s))

        return var

    def compute_edcc(self, em_slice, list_angle_rad=None, divide_by_variance=True):
        """
        Compute the absolute relative error of the exponential data consistency conditions (eDCC) for the given pair of
        projections.

        :param em_slice: Index of the lines on which the eDCC are computed. Could be a list of two integers
            ([first_line, last_line]), a string ("first_line:last_line") or an integer. If several lines are given,
            the absolute relative error for each pair of projections correspond to the average value over the lines
        :type em_slice: Union[list, str, int]
        :param list_angle_rad: List of tuples where each tuple specifies the two angles (in radian) between which the
            eDCC are computed. If None, the eDCC are computed for all the combinations of angles
        :type list_angle_rad: list, optional
        :param divide_by_variance: If True, the absolute error of the eDCC value is divide by the standard deviation
            of the eDCC. Default is True
        :type divide_by_variance: bool, optional

        :return: If divide_by_variance is True, the return value is a array where each component is the absolute error
            of the eDCC divide by the standard deviation of the eDCC for a given pair of projections (defined by the
            list_angle_rad parameter). If divide_by_variance is False, each component of the array is the absolute
            relative error in percentage of the eDCC for a given pair of projections (defined by the
            list_angle_rad parameter)
        :rtype: numpy.ndarray
        """
        if list_angle_rad is None:
            list_angle_rad = [(self.angles_rad[i], self.angles_rad[j]) for i in range(len(self.angles_rad)) for j in
                              range(len(self.angles_rad)) if (np.abs(self.angles_rad[i] - self.angles_rad[j])> 0.001)
                              and (np.abs(np.abs(self.angles_rad[i] - self.angles_rad[j]) - np.pi) > np.deg2rad(1))]
        ind_min, ind_max = extract_index(em_slice)
        if ind_max == -1:
            ind_max = self.size[1]
        list_slice = np.arange(ind_min, ind_max)
        self.edcc = []
        variance_pij = []
        variance_pji = []
        count_angle = -1
        for phi in list_angle_rad:
            if np.abs(phi[0] - phi[1] + np.pi) < 0.0001 or np.abs(phi[0] - phi[1] - np.pi) < 0.0001 or np.abs(
                    phi[0] - phi[1]) < 0.0001:
                continue
            else:
                count_angle += 1
                sigma_ij = self.mu_zero * np.tan(0.5 * (phi[0] - phi[1]))
                sigma_ji = -1 * sigma_ij

                ind_phi_i = int(np.where(np.abs(phi[0] - self.angles_rad) < 0.0001)[0][0])
                ind_phi_j = int(np.where(np.abs(phi[1] - self.angles_rad) < 0.0001)[0][0])
                projection_i = self.numpy_image[ind_phi_i][list_slice, :]
                projection_j = self.numpy_image[ind_phi_j][list_slice, :]
                P_ij = np.sum(np.multiply(projection_i, np.exp(self.s * sigma_ij)) * (self.s[1] - self.s[0]), axis=1)
                P_ji = np.sum(np.multiply(projection_j, np.exp(self.s * sigma_ji)) * (self.s[1] - self.s[0]), axis=1)
                P_ij[np.sum(projection_i, axis=1) < 10] = 0
                P_ji[np.sum(projection_j, axis=1) < 10] = 0
                if divide_by_variance:
                    self.edcc.append(np.mean(np.abs(np.subtract(P_ij, P_ji))))
                    Var_ij = (self.s[1] - self.s[0]) ** 2 * np.sum(
                        np.multiply(self.conversion_factor.numpy_image[ind_phi_i, list_slice, :] ** 2,
                                    self.projections.numpy_image[
                                    ind_phi_i, list_slice, :]) * np.exp(2 * sigma_ij * self.s), axis=1)
                    Var_ji = (self.s[1] - self.s[0]) ** 2 * np.sum(
                        np.multiply(self.conversion_factor.numpy_image[ind_phi_j, list_slice, :] ** 2,
                                    self.projections.numpy_image[
                                    ind_phi_j, list_slice, :]) * np.exp(2 * sigma_ji * self.s), axis=1)
                    Var_ij[np.sum(projection_i, axis=1) < 10] = 0
                    Var_ji[np.sum(projection_j, axis=1) < 10] = 0
                    variance_pij.append(np.sum(Var_ij))
                    variance_pji.append(np.sum(Var_ji))

                else:

                    if absolute_relative_error:
                        value_diff = np.abs(
                            np.divide(np.subtract(P_ij, P_ji), np.add(P_ij, P_ji), where=np.add(P_ij, P_ji) != 0,
                                      out=np.zeros_like(P_ij)))
                        self.edcc.append(np.mean(value_diff) * 200)
                    else:
                        value_diff = np.abs((np.subtract(P_ij, P_ji)))
                        self.edcc.append(np.mean(value_diff))
        self.edcc = np.asarray(self.edcc)

        if divide_by_variance:
            variance_pij = np.asarray(variance_pij) / list_slice.size ** 2
            variance_pji = np.asarray(variance_pji) / list_slice.size ** 2
            self.variance = np.sqrt(variance_pij + variance_pji)
            self.edcc = np.divide(self.edcc, self.variance * np.sqrt(list_slice.size), where=self.variance != 0,
                                  out=self.edcc)
        return self.edcc
#



def translate_and_rotate_a_2D_projection(projection_numpy_image, angle, origin, spacing, size, delta_s=0, delta_l=0):
    '''
    Rotate a 2D projection by a given angle.
    :param projection_numpy_image: numpy array of the projection
    :type projection_numpy_image: numpy.ndarray
    :param angle: Rotation angle
    :type angle: float
    :param spacing:
    :param origin:
    :param direction:
    :return: numpy array of the projection after rotation
    '''
    pad = 0
    input = itk.GetImageFromArray(projection_numpy_image)
    input.SetSpacing(spacing)
    input.SetOrigin(origin)

    rotation_array = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rotationMatrix = itk.matrix_from_array(rotation_array)

    transform = itk.AffineTransform[itk.D, 2].New()
    transform.SetCenter([0] * 2)
    transform.SetTranslation([0] * 2)
    transform.SetMatrix(rotationMatrix)
    inverseTransform = itk.AffineTransform[itk.D, 2].New()
    transform.GetInverse(inverseTransform)

    tempImageType = itk.Image[itk.F, 2]

    resampleFilter = itk.ResampleImageFilter.New(Input=input)
    resampleFilter.SetOutputSpacing(spacing)

    new_origin = np.dot(np.transpose(rotation_array), [-delta_s, -delta_l]) + origin
    resampleFilter.SetOutputOrigin(new_origin)
    resampleDirection = itk.matrix_from_array(np.eye(2))
    resampleFilter.SetOutputDirection(resampleDirection)
    resampleFilter.SetSize(size)
    resampleFilter.SetTransform(transform)

    interpolator = itk.LinearInterpolateImageFunction[tempImageType, itk.D].New()
    resampleFilter.SetInterpolator(interpolator)
    resampleFilter.SetDefaultPixelValue(pad)
    resampleFilter.Update()

    return itk.GetArrayFromImage(resampleFilter.GetOutput())


def translate_and_rotate_a_2D_projection_torch(projection_torch_image_i, angle, origin, spacing, size, delta_s=0, delta_l=0):

    angle_deg = (angle.item() / (2 * torch.pi) * 360)

    rotation_array = [[np.cos(angle.item()), -np.sin(angle.item())], [np.sin(angle.item()), np.cos(angle.item())]]
    new_origin = np.dot(np.transpose(rotation_array), [-delta_s, -delta_l])
    new_origin_i = (-new_origin[0]/spacing[0])
    new_origin_j = (-new_origin[1]/spacing[1])


    projection_i_torch = torchvision.transforms.functional.affine(img=projection_torch_image_i[None, :, :],
                                                                  angle=-angle_deg,
                                                                  translate=(new_origin_i,new_origin_j),
                                                                  interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                  scale=1,
                                                                  shear=0)[0, :, :]
    return projection_i_torch


def translate_and_rotate_a_2D_projection_torch_vect_(
    projections,   # (N, X, Y)
    angles,        # (N,) radians
    spacing,       # (2,) mm
    origin,
    delta_s,       # (N,) mm
    delta_l        # (N,) mm
):
    """
    Vectorized rotation+translation for a batch of projections using F.grid_sample.

    projections: (N, X, Y) tensor
    angles:      (N,) rotation angles in radians
    origin:      (2,) physical coordinates in mm from top-left (s, l)
    spacing:     (2,) pixel spacing in mm (s_spacing, l_spacing)
    delta_s:     (N,) translation along s-axis in mm
    delta_l:     (N,) translation along l-axis in mm
    """


    device, dtype = projections.device, projections.dtype
    N, H, W = projections.shape
    projs = projections.unsqueeze(1)                          # (N,1,H,W)

    origin_px = -torch.as_tensor(origin, device=device, dtype=dtype) / torch.as_tensor(spacing, device=device, dtype=dtype)          # (2,)
    t_px = torch.stack([torch.as_tensor(delta_s, device=device, dtype=dtype),
                        torch.as_tensor(delta_l, device=device, dtype=dtype)], dim=1) / torch.as_tensor(spacing, device=device, dtype=dtype)  # (N,2)

    a = torch.as_tensor(-angles, device=device, dtype=dtype)
    c, s = torch.cos(a), torch.sin(a)
    M = torch.stack([torch.stack([c,  s], dim=1),
                     torch.stack([-s, c], dim=1)], dim=1)   # (N,2,2) -> this is R^T (inverse rotation)

    center = torch.tensor([(W - 1.0) / 2.0, (H - 1.0) / 2.0], device=device, dtype=dtype)  # (2,)
    S = torch.tensor([2.0 / (W - 1.0), 2.0 / (H - 1.0)], device=device, dtype=dtype)        # (2,)

    v = (center.unsqueeze(0) - origin_px.unsqueeze(0) - t_px).unsqueeze(-1)   # (N,2,1)
    offs = (S.unsqueeze(0) * torch.matmul(M.transpose(1,2), v).squeeze(-1)) + (S * origin_px) - 1.0   # (N,2)

    theta = torch.zeros((N, 2, 3), device=device, dtype=dtype)
    theta[:, :, :2] = M
    theta[:, :, 2] = offs

    grid = torch.nn.functional.affine_grid(theta, size=projs.size(), align_corners=True)    # (N,H,W,2)
    out = torch.nn.functional.grid_sample(projs, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return out.squeeze(1)   # (N,H,W)

    # device = projections.device
    # dtype  = projections.dtype
    # N, H, W = projections.shape
    #
    # # Convert to (N, 1, H, W) for grid_sample
    # projections = projections.unsqueeze(1)
    #
    # # -------------------
    # # Compute rotation matrices (in pixel space)
    # # -------------------
    # cos_a = torch.cos(angles)
    # sin_a = torch.sin(angles)
    #
    # # Rotation: torch.nn.functional.affine_grid expects matrices for normalized coords
    # # Build the 2x3 affine matrices in pixel coordinates
    # rotation_mats = torch.stack([
    #     torch.stack([cos_a, -sin_a], dim=1),
    #     torch.stack([sin_a,  cos_a], dim=1)
    # ], dim=1)  # (N, 2, 2)
    #
    # # -------------------
    # # Compute translations in pixel units
    # # -------------------
    # spacing = torch.as_tensor(spacing, dtype=dtype, device=device)
    # # -------------------
    # # Apply rotation to translation offsets
    # # -------------------
    # # Following your original logic: new_origin = R^T * [-delta_s, -delta_l]
    # translations = torch.stack([-delta_s, -delta_l], dim=1)  # (N, 2) in mm
    # # translations_px = torch.matmul(rotation_mats.transpose(1,2), translations_px.unsqueeze(-1)).squeeze(-1)
    #
    # translations_px=torch.einsum('iab,ib->ia', rotation_mats.transpose(1,2), translations)
    #
    # # These are pixel translations (i, j) we need in affine matrix
    # trans_i = translations_px[:, 0] / (spacing[0] * W * 2)  # normalize to [-1,1]
    # trans_j = translations_px[:, 1] / (spacing[1] * H * 2)  # normalize to [-1,1]
    #
    #
    # # Actually, easier: build full normalized affine directly
    # # Normalize scaling:
    # norm_rot = torch.zeros((N, 2, 3), dtype=dtype, device=device)
    # norm_rot[:, 0, 0] = cos_a   # stays cos_a
    # norm_rot[:, 0, 1] = -sin_a
    # norm_rot[:, 1, 0] = sin_a
    # norm_rot[:, 1, 1] = cos_a
    # norm_rot[:, 0, 2] = trans_i
    # norm_rot[:, 1, 2] = trans_j
    # # -------------------
    # # Generate grid and sample
    # # -------------------
    # grid = torch.nn.functional.affine_grid(norm_rot, size=projections.size(), align_corners=False)
    # out  = torch.nn.functional.grid_sample(projections, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    #
    # return out.squeeze(1)  # back to (N, H, W)


def translate_and_rotate_a_2D_projection_torch_vect(images, angles, delta_s, delta_l, spacing):
    """
    Apply rotation + translation to a batch of 2D images.

    Args:
        images: Tensor of shape (N, H, W) - batch of 2D images
        angles: Tensor of shape (N,) - rotation angles in radians
        delta_s: Tensor of shape (N,) - translation in x-direction (after rotation)
        delta_l: Tensor of shape (N,) - translation in y-direction (after rotation)

    Returns:
        transformed_images: Tensor of shape (N, H, W) - transformed images
    """

    idtype = images.dtype
    # tdtype= torch.float16
    # images = images.to(tdtype)
    # angles = angles.to(tdtype)
    # delta_s = delta_s.to(tdtype)
    # delta_l = delta_l.to(tdtype)

    N, H, W = images.shape
    device = images.device

    # Add channel dimension for grid_sample (expects NCHW format)
    images = images.unsqueeze(1)  # Shape: (N, 1, H, W)

    # Create rotation matrices for each image in the batch
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Create affine transformation matrices (2x3 format)
    # [cos(θ) -sin(θ) tx]
    # [sin(θ)  cos(θ) ty]
    theta = torch.zeros(N, 2, 3, device=device, dtype=images.dtype)

    # Rotation part
    theta[:, 0, 0] = cos_angles  # R11
    theta[:, 0, 1] = -sin_angles  # R12
    theta[:, 1, 0] = sin_angles  # R21
    theta[:, 1, 1] = cos_angles  # R22

    # Translation part (normalized to [-1, 1] coordinate system)
    # PyTorch's grid uses [-1, 1] coordinate system where:
    # -1 corresponds to 0, +1 corresponds to W-1 (for x) or H-1 (for y)
    theta[:, 0, 2] = 2.0 * -delta_s/spacing[0] / (W - 1)  # tx (normalized)
    theta[:, 1, 2] = 2.0 * -delta_l/spacing[1] / (H - 1)  # ty (normalized)

    # Generate sampling grid

    grid = torch.nn.functional.affine_grid(theta, images.size(), align_corners=True)

    # Sample the images using the generated grid
    transformed_images = torch.nn.functional.grid_sample(images, grid, align_corners=True,
                                       mode='bilinear', padding_mode='zeros')

    # print("-----------------------------------------------------------------")
    # print(f"||{images.dtype=}+{grid.dtype=} ----> {transformed_images.dtype=}|||")
    # print("-----------------------------------------------------------------")

    # Remove the channel dimension to return to original format
    transformed_images = transformed_images.squeeze(1)  # Shape: (N, H, W)

    return transformed_images

def define_angle(v1, v2, n):
    '''
    This function defines the rotation angle between two vectors v1 and v2 in the plane defined by the normal vector n.
    '''
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    a = np.arccos(np.round(np.dot(v1, v2), 5))
    if np.dot(np.cross(v1, v2), n) < 0:
        a = a * -1
    return a

def define_angle_torch(v1, v2, n):
    '''
    This function defines the rotation angle between two vectors v1 and v2 in the plane defined by the normal vector n.
    '''
    v1 = v1 / torch.linalg.norm(v1)
    v2 = v2 / torch.linalg.norm(v2)
    a = torch.arccos(torch.round(torch.dot(v1, v2), decimals=5))
    if torch.dot(torch.linalg.cross(v1, v2), n) < 0:
        a = a * -1
    return a

def define_angle_torch_vect(v1,v2,n):
    v1 = v1 / (torch.linalg.norm(v1,dim=1)[:,None].clamp(min=eps))
    v2 = v2 / (torch.linalg.norm(v2,dim=1)[:,None].clamp(min=eps))
    v1_dot_v2 = torch.einsum("nd,nd->n", v1,v2)

    a = torch.arccos(torch.clamp(v1_dot_v2, min=-1.0+eps, max=1.0-eps))

    v1_cross_v2_dot_n = torch.einsum("nd,nd->n", torch.linalg.cross(v1,v2), n)
    a[v1_cross_v2_dot_n<0] = - a[v1_cross_v2_dot_n<0]

    return a

def define_angle_torch_vect_bis(v1,v2,n):
    v1 = v1 / (torch.linalg.norm(v1,dim=2)[:,:,None]+1e-8)
    v2 = v2 / (torch.linalg.norm(v2,dim=2)[:,:,None]+1e-8)
    v1_dot_v2 = torch.einsum("ijd,ijd->ij", v1,v2)

    a = torch.arccos(torch.clamp(v1_dot_v2, min=-1.0, max=1.0))

    v1_cross_v2_dot_n = torch.einsum("ijd,ijd->ij", torch.linalg.cross(v1,v2), n)
    a[v1_cross_v2_dot_n<0] = - a[v1_cross_v2_dot_n<0]

    return a

def compute_rotation_matrix_to_select_the_same_plane(i, j, geometry, A):
    '''
    This function computes the rotation matrix to select the same plane for two projections i and j.
    :param i:
    :param j:
    :param geometry:
    :return:
    '''

    angles_rad = geometry.GetGantryAngles()
    alpha_i = [np.cos(angles_rad[i]), np.sin(angles_rad[i]), 0]
    gamma_i = [-np.sin(angles_rad[i]), np.cos(angles_rad[i]), 0]

    alpha_j = np.dot(A, [np.cos(angles_rad[j]), np.sin(angles_rad[j]), 0])
    gamma_j = np.dot(A, [-np.sin(angles_rad[j]), np.cos(angles_rad[j]), 0])
    n = np.cross(gamma_i, gamma_j)
    n = n / np.linalg.norm(n)
    new_alpha_i = np.cross(gamma_i, n) / np.linalg.norm(np.cross(gamma_i, n))
    new_alpha_j = np.cross(gamma_j, n) / np.linalg.norm(np.cross(gamma_j, n))
    angle_i = define_angle(new_alpha_i, alpha_i, gamma_i)
    angle_j = define_angle(new_alpha_j, alpha_j, gamma_j)

    return angle_i, angle_j, gamma_i, gamma_j, new_alpha_i, new_alpha_j

def compute_rotation_matrix_to_select_the_same_plane_torch(i, j, angles_rad, A_torch):

    alpha_i = torch.Tensor([torch.cos(angles_rad[i]), torch.sin(angles_rad[i]), 0.0],device=A_torch.device)
    gamma_i = torch.Tensor([-torch.sin(angles_rad[i]), torch.cos(angles_rad[i]), 0.0],device=A_torch.device)

    alpha_j = torch.Tensor([torch.cos(angles_rad[j]), torch.sin(angles_rad[j]), 0.0],device=A_torch.device)
    gamma_j = torch.Tensor([-torch.sin(angles_rad[j]), torch.cos(angles_rad[j]), 0.0],device=A_torch.device)
    alpha_j = torch.matmul(A_torch,alpha_j)
    gamma_j = torch.matmul(A_torch, gamma_j)

    n = torch.linalg.cross(gamma_i, gamma_j)
    n = n / torch.linalg.norm(n)
    new_alpha_i = torch.linalg.cross(gamma_i, n) / torch.linalg.norm(torch.linalg.cross(gamma_i, n))
    new_alpha_j = torch.linalg.cross(gamma_j, n) / torch.linalg.norm(torch.linalg.cross(gamma_j, n))
    angle_i = define_angle_torch(new_alpha_i, alpha_i, gamma_i)
    angle_j = define_angle_torch(new_alpha_j, alpha_j, gamma_j)

    return angle_i, angle_j, gamma_i, gamma_j, new_alpha_i, new_alpha_j

def compute_rotation_matrix_to_select_the_same_plane_torch_vect(tensor_indices_ij, angles_rad, A_torch):
    eps = 1e-8

    alpha_i = torch.tensor([[torch.cos(angles_rad[i]),
                             torch.sin(angles_rad[i]), 0.0] for i in tensor_indices_ij[:,0]],
                           device=A_torch.device,dtype=A_torch.dtype)
    gamma_i = torch.tensor([[-torch.sin(angles_rad[i]),
                             torch.cos(angles_rad[i]), 0.0] for i in tensor_indices_ij[:,0]],
                           device=A_torch.device,dtype=A_torch.dtype)

    alpha_j = torch.tensor([[torch.cos(angles_rad[j]),
                             torch.sin(angles_rad[j]), 0.0] for j in tensor_indices_ij[:,1]],
                           device=A_torch.device,dtype=A_torch.dtype)
    gamma_j = torch.tensor([[-torch.sin(angles_rad[j]),
                             torch.cos(angles_rad[j]), 0.0] for j in tensor_indices_ij[:,1]],
                           device=A_torch.device,dtype=A_torch.dtype)

    alpha_j = torch.einsum('ab,ib->ia',A_torch,alpha_j)
    gamma_j = torch.einsum('ab,ib->ia',A_torch,gamma_j)
    # print(f"{alpha_i.max()=}")

    n = torch.linalg.cross(gamma_i, gamma_j)
    n = n / (torch.linalg.norm(n,dim=1)[:,None]+eps)
    new_alpha_i = torch.linalg.cross(gamma_i, n) / (torch.linalg.norm(torch.linalg.cross(gamma_i, n),dim=1)[:,None].clamp(min=eps))
    new_alpha_j = torch.linalg.cross(gamma_j, n) / (torch.linalg.norm(torch.linalg.cross(gamma_j, n),dim=1)[:,None].clamp(min=eps))
    angle_i = define_angle_torch_vect(new_alpha_i, alpha_i, gamma_i)
    angle_j = define_angle_torch_vect(new_alpha_j, alpha_j, gamma_j)
    # print(f"{new_alpha_i.max()=}")
    # print(f"{angle_i.max()=}")

    return angle_i, angle_j, gamma_i, gamma_j, new_alpha_i, new_alpha_j


def compute_rotation_matrix_to_select_the_same_plane_torch_vect_bis(ind_i,ind_j, angles_rad, A_torch):
    alpha_i = torch.cat((torch.cos(angles_rad[ind_i][:,:,None]), torch.sin(angles_rad[ind_i][:,:,None]),0*ind_i[:,:,None]),dim=2)
    gamma_i = torch.cat((-torch.sin(angles_rad[ind_i][:,:,None]), torch.cos(angles_rad[ind_i][:,:,None]),0*ind_i[:,:,None]),dim=2)

    alpha_j = torch.cat((torch.cos(angles_rad[ind_j][:,:,None]), torch.sin(angles_rad[ind_j][:,:,None]),0*ind_i[:,:,None]),dim=2)
    gamma_j = torch.cat((-torch.sin(angles_rad[ind_j][:,:,None]), torch.cos(angles_rad[ind_j][:,:,None]),0*ind_i[:,:,None]),dim=2)

    alpha_j = torch.einsum('ab,ijb->ija',A_torch,alpha_j)
    gamma_j = torch.einsum('ab,ijb->ija',A_torch,gamma_j)

    n = torch.linalg.cross(gamma_i, gamma_j)

    n = n / torch.linalg.norm(n,dim=2)[:,:,None]
    new_alpha_i = torch.linalg.cross(gamma_i, n) / torch.linalg.norm(torch.linalg.cross(gamma_i, n),dim=2)[:,:,None]
    new_alpha_j = torch.linalg.cross(gamma_j, n) / torch.linalg.norm(torch.linalg.cross(gamma_j, n),dim=2)[:,:,None]
    angle_i = define_angle_torch_vect_bis(new_alpha_i, alpha_i, gamma_i)
    angle_j = define_angle_torch_vect_bis(new_alpha_j, alpha_j, gamma_j)

    # print(f"{angle_i.shape=},{angle_j.shape=},{gamma_i.shape=},{gamma_j.shape=},{new_alpha_i.shape=},{new_alpha_j.shape=}, ")

    return angle_i, angle_j, gamma_i, gamma_j, new_alpha_i, new_alpha_j


def compute_edcc_in_parallel(projections, ref_exponential_projection, em_slice, motion, mu_zero,
                             m=None, no_of_heads=1):
    '''
    This function compute edcc in parallel to reduce the computation time (hopefully)
    since for each pair of projection, the re-orientation to be projected from same slice of activity is varied
    :param projections: projection with patient rotation
    :type projections: ProjectionsClass
    :param ref_exponential_projection: reference exponential projection computed by original attenuation map and the reference geometry by the scanner
    :type ref_exponential_projection: ExponentialProjectionsClass
    :param em_slice: list of lines on which the eDCC are averaged
    :type em_slice: list
    :param rotation: 3D array of rotation angles [theta_x, theta_y, theta_z]
    :type rotation: np.ndarray
    :param m: selected index to divide the acquisition into two sets: before m and after m ( run index: eg for 2 detector heads, 60 runs but 120 projections)

    :type m: int
    :param no_of_heads: number of detector heads used for the acquisition (1 or 2)
    :return:
    '''
    # s1 = time.time()
    rotation = motion[3:6]
    translation = motion[:3]
    s1 = time.time()
    index_no_motion, index_with_motion = select_projection_affected_by_motion(projections.size[2], m, no_of_heads)
    A = multi_dot([[[np.cos(rotation[2]), -np.sin(rotation[2]), 0], [np.sin(rotation[2]), np.cos(rotation[2]), 0],
                    [0, 0, 1]], [[np.cos(rotation[1]), 0, np.sin(-rotation[1])], [0, 1, 0],
                                 [-np.sin(-rotation[1]), 0, np.cos(rotation[1])]],
                   [[1, 0, 0], [0, np.cos(rotation[0]), -np.sin(rotation[0])],
                    [0, np.sin(rotation[0]), np.cos(rotation[0])]]])
    # I did a mistake for my GATE simulation. Let's try this one for my GATE simulation only
    # In Gate A_gate = Rx(theta_x)Ry(theta_y)Rz(theta_z) and in RTK A_rtk = Rz(-theta_z)Ry(-theta_y)Rx(-theta_x)
    # A_gate = transpose(A_rtk)
    # And it will return the same result of rotation of Rz(-theta_z)Ry(-theta_y)Rx(-theta_x)
    # The code was validated on RTK and GATE simulated data for the same motion -20,-40,-60 mm and -30,-45,-60 degrees
    # However the derivation of the rotational matrix A is not fully followed the theory...
    # There are still something that does not  make sense but I would like to figure out later
    attenuation_map = ref_exponential_projection.itk_attenuation_map
    K_region = ref_exponential_projection.voxelized_region

    angles_rad = projections.angles_rad
    sids = np.asarray(ref_exponential_projection.geometry.GetSourceToIsocenterDistances())

    geometry_modify = rtk.ThreeDCircularProjectionGeometry.New()
    for j in range(len(angles_rad)):
        if j in index_no_motion:
            geometry_modify.AddProjectionInRadians(sids[j], 0, angles_rad[j], 0, 0, 0, 0)
        else:
            theta = angles_rad[j]
            # gamma = [-np.sin(theta), np.cos(theta), 0]
            gamma_tilde = np.dot(A, [-np.sin(theta), np.cos(theta), 0])  # A.gamma
            phi_tilde = np.arcsin(gamma_tilde[2])
            cos_theta_tilde = gamma_tilde[1] / np.cos(phi_tilde)
            sin_theta_tilde = -gamma_tilde[0] / np.cos(phi_tilde)
            theta_tilde = np.arctan2(sin_theta_tilde, cos_theta_tilde)
            alpha = [np.cos(theta), np.sin(theta), 0]
            beta = [0, 0, 1]
            alpha_tilde = np.dot(A, alpha)
            beta_tilde = np.dot(A, beta)

            # inverse M is rotation matrix applied on projection coordinates
            c00_component_of_inverse_matrix_M = np.dot(alpha_tilde, [np.cos(theta_tilde), sin_theta_tilde, 0])
            c01_component_of_inverse_matrix_M = np.dot(beta_tilde, [np.cos(theta_tilde), sin_theta_tilde, 0])
            inPlane_angle = np.arctan2(c01_component_of_inverse_matrix_M, c00_component_of_inverse_matrix_M)
            delta_s = translation[0] * np.cos(-theta) + translation[1] * np.sin(-theta)
            delta_l = translation[2]

            # print("delta_s: ", delta_s, "delta_l: ", delta_l)
            geometry_modify.AddProjectionInRadians(sids[j], 0, theta_tilde, delta_s, delta_l, phi_tilde, -inPlane_angle)

    # print("Time to modify geometry: ", time.time() - s1)
    s2 = time.time()
    empty_projection = rtk.constant_image_source(information_from_image=projections.itk_image,
                                                 ttype=(itk.Image[itk.F, 3]), constant=1)  # 3D float images
    AttToExp = rtk.AttenuatedToExponentialCorrectionImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
    AttToExp.SetGeometry(geometry_modify)
    AttToExp.SetInput(0, empty_projection)
    AttToExp.SetInput(1, attenuation_map)
    AttToExp.SetInput(2, K_region.itk_image)
    AttToExp.Update()

    original_geometry = ref_exponential_projection.geometry
    conversion_factor = VolumeClass(AttToExp.GetOutput())
    print("Time to compute conversion factor: ", time.time() - s2)
    projections.geometry = geometry_modify
    projections.update()
    # itk.imwrite(conversion_factor.itk_image, "conversion_factor_test.mha")
    angles_rad_modified = np.asarray(geometry_modify.GetGantryAngles())
    s3 = time.time()
    # With rotation, there are possibilities of 2 projections with the same angular positions, for example with rotation around the z axis
    list_indices = [[i, j] for i in index_no_motion for j in
                      index_with_motion
                      if (np.abs(angles_rad_modified[i] - angles_rad_modified[j]) > np.deg2rad(1)) and (
                              np.abs(np.abs(angles_rad_modified[i] - angles_rad_modified[j]) - np.pi) > np.deg2rad(
                          1))]  # prevent exponential term going to inf
    ind_min, ind_max = extract_index(em_slice)
    list_slice = np.arange(ind_min, ind_max)
    edcc = []
    for ij in list_indices:
        edcc_one_pair = _compute_edcc_one_pair(ij, projections, original_geometry, conversion_factor, mu_zero,
                                               list_slice, A)
        edcc.append(edcc_one_pair)
    print("Time to pair projections: ", time.time() - s3)
    # print("Number of pair for mid-acqusition motion ", len(edcc))
    return (edcc)


def _compute_edcc_one_pair(ij, projections, original_geometry, conversion_factor, mu_zero, list_slice, A):
    '''
    This function computes the eDCC for a pair of projections
    :param phi: pair of angular positions of the two projections
    :type phi: tuple
    :param projection_numpy_image: numpy array of the projections
    :type projection_numpy_image: numpy.ndarray
    :param conversion_factor: conversion factor
    :type conversion_factor: VolumeClass
    :param mu_zero: mu zero
    :type mu_zero: float
    :param s: Laplace transform point
    :type s: np.ndarray
    :param list_slice: list of lines on which the eDCC are computed
    :type list_slice: np.ndarray
    :param geometry_path: path to the geometry file
    :type geometry_path: str
    :param m: selected index to divide the acquisition into two sets: before m and after m ( run index: eg for 2 detector heads, 60 runs but 120 projections)
    :type m: int
    :param projection_inPlane_angles: List of in-plane angles for each projection defined from the inevrse matrix M
    :type projection_inPlane_angles: list
    :return: eDCC value
    :rtype: float
    '''

    geometry_modify = projections.geometry

    ind_phi_i = ij[0]
    ind_phi_j = ij[1]



    origin = projections.origin[:2]
    spacing = projections.spacing[:2]
    size = projections.size[:2]
    projection_numpy_image = projections.numpy_image
    s = np.asarray(projections.s)

    angle_i, angle_j, gamma_i, gamma_j, new_alpha_i, new_alpha_j = compute_rotation_matrix_to_select_the_same_plane(
        ind_phi_i, ind_phi_j, original_geometry, A)

    if np.abs(angle_i) < 0.0001:
        projection_i = projection_numpy_image[ind_phi_i]
        conversion_factor_i = conversion_factor.numpy_image[ind_phi_i]
    else:
        projection_i = translate_and_rotate_a_2D_projection(projection_numpy_image[ind_phi_i], angle_i, origin, spacing, size)
        conversion_factor_i = translate_and_rotate_a_2D_projection(conversion_factor.numpy_image[ind_phi_i], angle_i, origin, spacing, size)

    delta_s = geometry_modify.GetProjectionOffsetsX()[ind_phi_j]
    delta_l = geometry_modify.GetProjectionOffsetsY()[ind_phi_j]
    if np.abs(angle_j) < 0.0001 and np.abs(delta_s) < 0.1 and np.abs(delta_l) < 0.1:
        print("angle_j: ", angle_j, "delta_s: ", delta_s, "delta_l: ", delta_l)
        conversion_factor_j = conversion_factor.numpy_image[ind_phi_j]
        projection_j = projection_numpy_image[ind_phi_j]
    else:

        projection_j = translate_and_rotate_a_2D_projection(projection_numpy_image[ind_phi_j], angle_j, origin, spacing, size, delta_s, delta_l)
        conversion_factor_j = translate_and_rotate_a_2D_projection(conversion_factor.numpy_image[ind_phi_j], angle_j, origin, spacing, size, delta_s, delta_l)
    sign = np.sign(np.dot(np.add(new_alpha_i, new_alpha_j), np.subtract(gamma_j, gamma_i)))
    sigma_ij = sign * mu_zero /(np.dot(gamma_i, gamma_j)+ 1)* np.linalg.norm(np.cross(gamma_i, gamma_j))
    sigma_ji = -1 * sigma_ij


    if np.abs(sigma_ij) > 1:
        edcc_one_pair = 0
        # to prevent exp overflow
    else:

        exponential_projection_i = np.multiply(projection_i, conversion_factor_i)[list_slice, :]
        exponential_projection_j = np.multiply(projection_j, conversion_factor_j)[list_slice, :]
        P_ij = np.sum(np.multiply(exponential_projection_i, np.exp(s * sigma_ij)) * (s[1] - s[0]), axis=1)
        P_ji = np.sum(np.multiply(exponential_projection_j, np.exp(s * sigma_ji)) * (s[1] - s[0]), axis=1)
        P_ij[np.sum(exponential_projection_i, axis=1) < 10] = 0
        P_ji[np.sum(exponential_projection_j, axis=1) < 10] = 0

        abs_relative_difference = (np.mean(np.abs(np.subtract(P_ij, P_ji))))
        Var_ij = (s[1] - s[0]) ** 2 * np.sum(
            np.multiply(conversion_factor_i[ list_slice, :] ** 2,
                        projection_i[ list_slice, :]) * np.exp(2 * sigma_ij * s), axis=1)
        Var_ji = (s[1] - s[0]) ** 2 * np.sum(
            np.multiply(conversion_factor_j[list_slice, :] ** 2,
                        projection_j[ list_slice, :]) * np.exp(2 * sigma_ji * s), axis=1)
        Var_ij[np.sum(exponential_projection_i, axis=1) < 10] = 0
        Var_ji[np.sum(exponential_projection_j, axis=1) < 10] = 0
        variance_pij = np.sum(Var_ij) / list_slice.size ** 2
        variance_pji = np.sum(Var_ji) / list_slice.size ** 2
        variance = np.sqrt(variance_pij + variance_pji)
        edcc_one_pair = abs_relative_difference/ (variance * np.sqrt(list_slice.size))
    if np.isnan(edcc_one_pair):
        edcc_one_pair = 0

    return edcc_one_pair

def select_projection_affected_by_motion(N_proj, m, no_of_heads):
    """
    This function simply selects the projection indices that are affected by motion and those that are not affected by motion
        """
    if N_proj < m:
        print("Transition run should be smaller than the number of projections")
        exit()
    if no_of_heads == 1:
        index_no_motion = np.arange(0, m)
        index_with_motion = np.arange(m, N_proj)
    elif no_of_heads == 2:
        index_no_motion = np.append(np.arange(0, m),
                              np.arange(int(N_proj / 2), m + int(N_proj / 2)))
        index_with_motion = np.append(np.arange(m, int(N_proj / 2)),
                                np.arange(int(m + N_proj / 2), N_proj))
    elif no_of_heads == 12:
        no_of_projections_per_detector = N_proj/12
        no_motion_one_detector = np.append(np.full(m,True), np.full(int(no_of_projections_per_detector - m), False))
        no_motion_12_detectors = np.tile(no_motion_one_detector, 12)
        index_no_motion = np.arange(N_proj)[no_motion_12_detectors]
        index_with_motion = np.delete(np.arange(N_proj), index_no_motion)
    else:
        print("Not yet implemented")
    return index_no_motion, index_with_motion

def compute_edcc_in_parallel_torch(projections, ref_exponential_projection, em_slice, motion, mu_zero,
                             m=None, no_of_heads=1):
    '''
    This function compute edcc in parallel to reduce the computation time (hopefully)
    since for each pair of projection, the re-orientation to be projected from same slice of activity is varied
    :param projections: projection with patient rotation
    :type projections: ProjectionsClass
    :param ref_exponential_projection: reference exponential projection computed by original attenuation map and the reference geometry by the scanner
    :type ref_exponential_projection: ExponentialProjectionsClass
    :param em_slice: list of lines on which the eDCC are averaged
    :type em_slice: list
    :param rotation: 3D array of rotation angles [theta_x, theta_y, theta_z]
    :type rotation: np.ndarray
    :param m: selected index to divide the acquisition into two sets: before m and after m ( run index: eg for 2 detector heads, 60 runs but 120 projections)

    :type m: int
    :param no_of_heads: number of detector heads used for the acquisition (1 or 2)
    :return:
    '''
    # s1 = time.time()
    dev = motion.device
    dtype = motion.dtype
    rotation_torch = motion[3:6]
    translation_torch = motion[:3]


    rotation_array = rotation_torch.double().detach().cpu().numpy()
    translation_array = translation_torch.double().detach().cpu().numpy()

    s1 = time.time()
    index_no_motion, index_with_motion = select_projection_affected_by_motion(projections.size[2], m, no_of_heads)
    A = multi_dot([[[np.cos(rotation_array[2]), -np.sin(rotation_array[2]), 0], [np.sin(rotation_array[2]), np.cos(rotation_array[2]), 0],
                    [0, 0, 1]], [[np.cos(rotation_array[1]), 0, np.sin(-rotation_array[1])], [0, 1, 0],
                                 [-np.sin(-rotation_array[1]), 0, np.cos(rotation_array[1])]],
                   [[1, 0, 0], [0, np.cos(rotation_array[0]), -np.sin(rotation_array[0])],
                    [0, np.sin(rotation_array[0]), np.cos(rotation_array[0])]]])

    Rz = torch.stack([
        torch.stack([torch.cos(rotation_torch[2]), -torch.sin(rotation_torch[2]),
                     torch.tensor(0., device=rotation_torch.device)], dim=0),
        torch.stack([torch.sin(rotation_torch[2]), torch.cos(rotation_torch[2]),
                     torch.tensor(0., device=rotation_torch.device)], dim=0),
        torch.stack([torch.tensor(0., device=rotation_torch.device), torch.tensor(0., device=rotation_torch.device),
                     torch.tensor(1., device=rotation_torch.device)], dim=0)
    ], dim=0)

    Ry = torch.stack([
        torch.stack([torch.cos(rotation_torch[1]), torch.tensor(0., device=rotation_torch.device),
                     torch.sin(-rotation_torch[1])], dim=0),
        torch.stack([torch.tensor(0., device=rotation_torch.device), torch.tensor(1., device=rotation_torch.device),
                     torch.tensor(0., device=rotation_torch.device)], dim=0),
        torch.stack([-torch.sin(-rotation_torch[1]), torch.tensor(0., device=rotation_torch.device),
                     torch.cos(rotation_torch[1])], dim=0)
    ], dim=0)

    Rx = torch.stack([
        torch.stack([torch.tensor(1., device=rotation_torch.device), torch.tensor(0., device=rotation_torch.device),
                     torch.tensor(0., device=rotation_torch.device)], dim=0),
        torch.stack([torch.tensor(0., device=rotation_torch.device), torch.cos(rotation_torch[0]),
                     -torch.sin(rotation_torch[0])], dim=0),
        torch.stack([torch.tensor(0., device=rotation_torch.device), torch.sin(rotation_torch[0]),
                     torch.cos(rotation_torch[0])], dim=0)
    ], dim=0)

    A_torch = Rz @ Ry @ Rx

    # I did a mistake for my GATE simulation. Let's try this one for my GATE simulation only
    # In Gate A_gate = Rx(theta_x)Ry(theta_y)Rz(theta_z) and in RTK A_rtk = Rz(-theta_z)Ry(-theta_y)Rx(-theta_x)
    # A_gate = transpose(A_rtk)
    # And it will return the same result of rotation of Rz(-theta_z)Ry(-theta_y)Rx(-theta_x)
    # The code was validated on RTK and GATE simulated data for the same motion -20,-40,-60 mm and -30,-45,-60 degrees
    # However the derivation of the rotational matrix A is not fully followed the theory...
    # There are still something that does not  make sense but I would like to figure out later
    attenuation_map = ref_exponential_projection.itk_attenuation_map
    K_region = ref_exponential_projection.voxelized_region

    angles_rad = projections.angles_rad
    sids = np.asarray(ref_exponential_projection.geometry.GetSourceToIsocenterDistances())

    geometry_modify = rtk.ThreeDCircularProjectionGeometry.New()
    for j in range(len(angles_rad)):
        if j in index_no_motion:
            geometry_modify.AddProjectionInRadians(sids[j], 0, angles_rad[j], 0, 0, 0, 0)
        else:
            theta = angles_rad[j]
            gamma_tilde = np.dot(A, [-np.sin(theta), np.cos(theta), 0])  # A.gamma
            phi_tilde = np.arcsin(gamma_tilde[2])
            cos_theta_tilde = gamma_tilde[1] / np.cos(phi_tilde)
            sin_theta_tilde = -gamma_tilde[0] / np.cos(phi_tilde)
            theta_tilde = np.arctan2(sin_theta_tilde, cos_theta_tilde)
            alpha = [np.cos(theta), np.sin(theta), 0]
            beta = [0, 0, 1]
            alpha_tilde = np.dot(A, alpha)
            beta_tilde = np.dot(A, beta)

            # inverse M is rotation matrix applied on projection coordinates
            c00_component_of_inverse_matrix_M = np.dot(alpha_tilde, [np.cos(theta_tilde), sin_theta_tilde, 0])
            c01_component_of_inverse_matrix_M = np.dot(beta_tilde, [np.cos(theta_tilde), sin_theta_tilde, 0])
            inPlane_angle = np.arctan2(c01_component_of_inverse_matrix_M, c00_component_of_inverse_matrix_M)
            delta_s = translation_array[0] * np.cos(-theta) + translation_array[1] * np.sin(-theta)
            delta_l = translation_array[2]

            geometry_modify.AddProjectionInRadians(sids[j], 0, theta_tilde, delta_s, delta_l, phi_tilde, -inPlane_angle)



    s2 = time.time()


    device = rotation_torch.device


    with torch.no_grad():
        attenuation_map_tensor = torch.from_numpy(itk.array_from_image(attenuation_map)).to(device)
        K_region_tensor = torch.from_numpy(itk.array_from_image(K_region.itk_image)).to(device)
        mu0 = attenuation_map_tensor[K_region_tensor > 0].mean()
        conversion_factor_tensor = conversion_factor_vectorized(volume=attenuation_map_tensor[None,:,:,:],
                                                                kregion=K_region_tensor[None,:,:,:],
                                                                geometry=geometry_modify,mu0=mu0)
        conversion_factor_itk = itk.image_from_array(conversion_factor_tensor.detach().cpu().numpy())
    conversion_factor_itk.CopyInformation(projections.itk_image)
    conversion_factor = VolumeClass(conversion_factor_itk)
    print("Time to compute conversion factor: ", time.time() - s2)
    del attenuation_map_tensor,K_region_tensor,conversion_factor_tensor,mu0

    projections.geometry = geometry_modify
    projections.update()
    original_geometry = ref_exponential_projection.geometry
    angles_rad_modified = np.asarray(geometry_modify.GetGantryAngles())


    s3 = time.time()

    # With rotation, there are possibilities of 2 projections with the same angular positions, for example with rotation around the z axis
    list_indices = [[i, j] for i in index_no_motion for j in
                      index_with_motion
                      if (np.abs(angles_rad_modified[i] - angles_rad_modified[j]) > np.deg2rad(1)) and (
                              np.abs(np.abs(angles_rad_modified[i] - angles_rad_modified[j]) - np.pi) > np.deg2rad(
                          1))]  # prevent exponential term going to inf
    ind_min, ind_max = extract_index(em_slice)
    list_slice = np.arange(ind_min, ind_max)


    projections_torch = torch.from_numpy(projections.numpy_image).to(dev).to(dtype)
    conversion_factor_torch = torch.from_numpy(conversion_factor.numpy_image).to(dev).to(dtype)
    angles_rad_original_geometry = torch.Tensor(original_geometry.GetGantryAngles()).to(dev).to(dtype)

    angles_rad_tensor = torch.Tensor(angles_rad).to(dev).to(dtype)
    delta_s_tensor = translation_torch[0] * torch.cos(-angles_rad_tensor) + translation_torch[1] * torch.sin(-angles_rad_tensor)
    delta_l_tensor = translation_torch[2]*torch.ones_like(angles_rad_tensor,device=dev).to(dtype)
    delta_s_tensor[index_no_motion] = 0
    delta_l_tensor[index_no_motion] = 0
    delta_s_tensor = delta_s_tensor.to(dtype)
    delta_l_tensor = delta_l_tensor.to(dtype)

    s_tensor = torch.Tensor(projections.s).to(dev).to(dtype)
    A_torch = A_torch.to(dev).to(dtype)
    spacing = projections.spacing[:2]

    batch_size = 64
    tensor_indices_ij = torch.Tensor(list_indices).to(dev).to(torch.int)
    edccs = []
    for _ in range(4):
        random_indices = torch.randint(low=0, high=len(list_indices), size=(batch_size,))
        edcc = _compute_edcc_all_pairs_torch(tensor_indices_ij[random_indices],
                                          spacing,
                                          angles_rad_original_geometry,
                                          projections_torch,
                                          conversion_factor_torch,
                                          mu_zero, list_slice, A_torch,
                                          delta_s_tensor, delta_l_tensor,
                                          s_tensor
                                          )
        edccs.append(edcc[None,:])
    edccs = torch.cat(edccs,dim=0)

    print("Time to pair projections: ", time.time() - s3)
    print("edcc shape: ", edccs.shape)
    print("edcc max: ", edccs.max())
    return (edccs)


def _compute_edcc_one_pair_torch(ij, projections,
                                 angles_rad_original_geometry,
                                 projections_torch,
                                 conversion_factor_torch,
                                 mu_zero, list_slice, A_torch,
                                 delta_s_tensor,delta_l_tensor,
                                 s_tensor):
    '''
    This function computes the eDCC for a pair of projections
    :param phi: pair of angular positions of the two projections
    :type phi: tuple
    :param projection_numpy_image: numpy array of the projections
    :type projection_numpy_image: numpy.ndarray
    :param conversion_factor: conversion factor
    :type conversion_factor: VolumeClass
    :param mu_zero: mu zero
    :type mu_zero: float
    :param s: Laplace transform point
    :type s: np.ndarray
    :param list_slice: list of lines on which the eDCC are computed
    :type list_slice: np.ndarray
    :param geometry_path: path to the geometry file
    :type geometry_path: str
    :param m: selected index to divide the acquisition into two sets: before m and after m ( run index: eg for 2 detector heads, 60 runs but 120 projections)
    :type m: int
    :param projection_inPlane_angles: List of in-plane angles for each projection defined from the inevrse matrix M
    :type projection_inPlane_angles: list
    :return: eDCC value
    :rtype: float
    '''

    ind_phi_i = int(ij[0])
    ind_phi_j = int(ij[1])

    origin = projections.origin[:2]
    spacing = projections.spacing[:2]
    size = projections.size[:2]

    angle_i, angle_j, gamma_i, gamma_j, new_alpha_i, new_alpha_j= compute_rotation_matrix_to_select_the_same_plane_torch(
        ind_phi_i, ind_phi_j, angles_rad_original_geometry, A_torch)

    if torch.abs(angle_i) < 0.0001:
        projection_i = projections_torch[ind_phi_i]
        conversion_factor_i = conversion_factor_torch[ind_phi_i]
    else:
        projection_i = translate_and_rotate_a_2D_projection_torch(projections_torch[ind_phi_i], angle_i, origin, spacing, size)
        conversion_factor_i = translate_and_rotate_a_2D_projection_torch(conversion_factor_torch[ind_phi_i], angle_i, origin, spacing, size)



    delta_s = delta_s_tensor[ind_phi_j]
    delta_l = delta_l_tensor[ind_phi_j]

    if torch.abs(angle_j) < 0.0001 and torch.abs(delta_s) < 0.1 and torch.abs(delta_l) < 0.1:
        print("angle_j: ", angle_j, "delta_s: ", delta_s, "delta_l: ", delta_l)
        projection_j = projections_torch[ind_phi_j]
        conversion_factor_j = conversion_factor_torch[ind_phi_j]
    else:
        projection_j = translate_and_rotate_a_2D_projection_torch(projections_torch[ind_phi_j], angle_j, origin, spacing, size, delta_s.item(), delta_l.item())
        conversion_factor_j = translate_and_rotate_a_2D_projection_torch(conversion_factor_torch[ind_phi_j], angle_j, origin, spacing, size, delta_s.item(), delta_l.item())
    sign = torch.sign(torch.dot(torch.add(new_alpha_i, new_alpha_j), torch.subtract(gamma_j, gamma_i)))
    sigma_ij = sign * mu_zero /(torch.dot(gamma_i, gamma_j)+ 1)* torch.linalg.norm(torch.linalg.cross(gamma_i, gamma_j))
    sigma_ji = -1 * sigma_ij

    if torch.abs(sigma_ij) > 1:
        edcc_one_pair = 0
        # to prevent exp overflow
    else:
        exponential_projection_i = torch.multiply(projection_i, conversion_factor_i)[list_slice, :]
        exponential_projection_j = torch.multiply(projection_j, conversion_factor_j)[list_slice, :]
        P_ij = torch.sum(torch.multiply(exponential_projection_i, torch.exp(s_tensor * sigma_ij)) * (s_tensor[1] - s_tensor[0]), dim=1)
        P_ji = torch.sum(torch.multiply(exponential_projection_j, torch.exp(s_tensor * sigma_ji)) * (s_tensor[1] - s_tensor[0]), dim=1)
        P_ij[torch.sum(exponential_projection_i, dim=1) < 10] = 0
        P_ji[torch.sum(exponential_projection_j, dim=1) < 10] = 0

        abs_relative_difference = (torch.mean(torch.abs(torch.subtract(P_ij, P_ji))))

        Var_ij = (s_tensor[1] - s_tensor[0]) ** 2 * torch.sum(
            torch.multiply(conversion_factor_i[ list_slice, :] ** 2,
                        projection_i[ list_slice, :]) * torch.exp(2 * sigma_ij * s_tensor), dim=1)
        Var_ji = (s_tensor[1] - s_tensor[0]) ** 2 * torch.sum(
            torch.multiply(conversion_factor_j[list_slice, :] ** 2,
                        projection_j[ list_slice, :]) * torch.exp(2 * sigma_ji * s_tensor), dim=1)
        Var_ij[torch.sum(exponential_projection_i, dim=1) < 10] = 0
        Var_ji[torch.sum(exponential_projection_j, dim=1) < 10] = 0
        variance_pij = torch.sum(Var_ij) / list_slice.size ** 2
        variance_pji = torch.sum(Var_ji) / list_slice.size ** 2
        variance = torch.sqrt(variance_pij + variance_pji)
        edcc_one_pair = abs_relative_difference/ (variance * (list_slice.size)**0.5)
    if torch.isnan(edcc_one_pair):
        edcc_one_pair = 0
    else:
        edcc_one_pair=edcc_one_pair.item()
    return edcc_one_pair


def _compute_edcc_all_pairs_torch(tensor_indices_ij, spacing,
                                 angles_rad_original_geometry,
                                 projections_torch,
                                 conversion_factor_torch,
                                 mu_zero, list_slice, A_torch,
                                 delta_s_tensor,delta_l_tensor,
                                 s_tensor):
    angle_i, angle_j, gamma_i, gamma_j, new_alpha_i, new_alpha_j = compute_rotation_matrix_to_select_the_same_plane_torch_vect(
        tensor_indices_ij, angles_rad_original_geometry, A_torch)

    projection_i = translate_and_rotate_a_2D_projection_torch_vect(images=projections_torch[tensor_indices_ij[:,0],:,:],
                                                                    angles=angle_i,
                                                                    spacing=spacing,
                                                                    delta_l= angle_i*0,
                                                                    delta_s= angle_j*0
                                                                    )


    mask_i = (angle_i.abs()<0.0001).detach()
    projection_i[mask_i,:,:] = projections_torch[tensor_indices_ij[mask_i,0],:,:]


    conversion_factor_i = translate_and_rotate_a_2D_projection_torch_vect(
        images=conversion_factor_torch[tensor_indices_ij[:, 0], :, :],
        angles=angle_i,
        spacing=spacing,
        delta_l=angle_i * 0,
        delta_s=angle_j * 0
        )
    conversion_factor_i[mask_i, :, :] = conversion_factor_torch[tensor_indices_ij[mask_i,0], :, :]

    ##-------------------------

    delta_s = delta_s_tensor[tensor_indices_ij[:,1]]
    delta_l = delta_l_tensor[tensor_indices_ij[:,1]]

    projection_j = translate_and_rotate_a_2D_projection_torch_vect(images=projections_torch[tensor_indices_ij[:,1],:,:],
                                                                    angles=angle_j,
                                                                    spacing= spacing,
                                                                    delta_l= delta_l,
                                                                    delta_s= delta_s
                                                                    )
    conversion_factor_j = translate_and_rotate_a_2D_projection_torch_vect(images=conversion_factor_torch[tensor_indices_ij[:,1],:,:],
                                                                           angles=angle_j,
                                                                           spacing=spacing,
                                                                           delta_s=delta_s,
                                                                           delta_l=delta_l)

    mask_j = ((torch.abs(angle_j) < 0.0001)*(torch.abs(delta_s) < 0.1)*(torch.abs(delta_l) < 0.1)).detach()
    projection_j[mask_j,:,:] = projections_torch[tensor_indices_ij[mask_j,1],:,:]
    conversion_factor_j[mask_j,:,:] = conversion_factor_torch[tensor_indices_ij[mask_j,1],:,:]


    ##-------------------------


    sign = torch.sign(torch.einsum("nd,nd->n",torch.add(new_alpha_i, new_alpha_j), torch.subtract(gamma_j, gamma_i)))
    sigma_ij = sign * mu_zero /(torch.einsum("nd,nd->n",gamma_i, gamma_j)+ 1).clamp(min=eps)*\
               torch.linalg.norm(torch.linalg.cross(gamma_i, gamma_j,dim=1),dim=1)
    sigma_ji = -1 * sigma_ij
    

    exponential_projection_i = torch.multiply(projection_i, conversion_factor_i)[:,list_slice, :]
    exponential_projection_j = torch.multiply(projection_j, conversion_factor_j)[:,list_slice, :]

    temp_exp_ij = torch.exp((s_tensor[None,None,:] * sigma_ij[:,None,None]))
    P_ij = torch.sum(torch.multiply(exponential_projection_i, temp_exp_ij)
                     * (s_tensor[1] - s_tensor[0]), dim=2)

    temp_exp_ji=torch.exp((s_tensor[None, None, :] * sigma_ji[:, None, None]))
    P_ji = torch.sum(torch.multiply(exponential_projection_j, temp_exp_ji)
                     * (s_tensor[1] - s_tensor[0]), dim=2)
    # print(f"|{temp_exp_ij.max()=} / {temp_exp_ji.max()=}")
    P_ij[torch.sum(exponential_projection_i, dim=2) < 10] = 0
    P_ji[torch.sum(exponential_projection_j, dim=2) < 10] = 0
    abs_relative_difference = (torch.mean(torch.abs(torch.subtract(P_ij, P_ji)),dim=1))
    # print(f"|{P_ij.max()=} / {P_ji.max()=}")

    mask_big_sigma_ij = sigma_ij.abs()>1
    sigma_ij[mask_big_sigma_ij] = 0
    sigma_ji[mask_big_sigma_ij] = 0

    temp_ij = torch.multiply(conversion_factor_i[:, list_slice, :] ** 2,
                   projection_i[:, list_slice, :])
    sigma_s_ij = (2 * sigma_ij[:,None,None] * s_tensor[None,None,:])
    Var_ij = (s_tensor[1] - s_tensor[0]) ** 2 * torch.sum(
        temp_ij * torch.exp(sigma_s_ij), dim=2)

    # print("temp_ij:", temp_ij.min(), temp_ij.max(), temp_ij.dtype)
    # print("sigma_s_ij:", sigma_s_ij.min(), sigma_s_ij.max(), sigma_s_ij.dtype)
    # print("Var_ij:", Var_ij.min(), Var_ij.max(), Var_ij.dtype)
    temp_ji = torch.multiply(conversion_factor_j[:,list_slice, :] ** 2,
                    projection_j[:,list_slice, :])
    sigma_s_ji = (2 * sigma_ji[:,None,None] * s_tensor[None,None,:])

    Var_ji = (s_tensor[1] - s_tensor[0]) ** 2 * torch.sum(
         temp_ji * torch.exp(sigma_s_ji), dim=2)

    # print("temp_ji:", temp_ji.min(), temp_ji.max(), temp_ji.dtype)
    # print("sigma_s_ji:", sigma_s_ji.min(), sigma_s_ji.max(), sigma_s_ji.dtype)
    # print("Var_ji:", Var_ji.min(), Var_ji.max(), Var_ji.dtype)

    Var_ij[torch.sum(exponential_projection_i, dim=1) < 10] = 0
    Var_ji[torch.sum(exponential_projection_j, dim=1) < 10] = 0
    variance_pij = torch.sum(Var_ij,dim=1) / list_slice.size ** 2
    variance_pji = torch.sum(Var_ji,dim=1) / list_slice.size ** 2
    variance = torch.sqrt(variance_pij + variance_pji+eps)
    # print("Var:",variance.min(), variance.max(), variance.dtype)
    edcc_all_pairs = abs_relative_difference/ (variance * (list_slice.size)**0.5 + eps)

    edcc_all_pairs[mask_big_sigma_ij] = 0
    edcc_all_pairs[torch.isnan(edcc_all_pairs)] = 0

    return edcc_all_pairs