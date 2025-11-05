from volume import VolumeClass
from projections import ProjectionsClass
from exponential_projections import ExponentialProjectionsClass, ExponentialProjectionsGeometryClass,select_projection_affected_by_motion,compute_edcc_in_parallel_torch
import math
import itk
from itk import RTK as rtk
import gatetools as gt
import numpy as np
import os
from scipy import ndimage
import tempfile
from multiprocessing.pool import Pool
import time




class MotionCorrection(ProjectionsClass):
    '''
    Class to handel a multidimensional optimization problem with Nelder Mead method

    :param projections: could be either path to the itk image file, numpy array or itk image
    :type projections: Union[str, numpy.ndarray, itk.itkImagePython]
    :param attenuation_map: attenuation map
    :type attenuation_map: Union[str, numpy.ndarray, itk.itkImagePython]
    :param K_region: Voxelized convex region
    :type K_region: Union[str, numpy.ndarray, itk.itkImagePython]
    :param geometry: Geometry of the acquisition
    :type geometry: itk.rtkThreeDCircularProjectionGeometryPython.rtkThreeDCircularProjectionGeometry
    :param conversion_factor_path: path to the conversion factor
    :type conversion_factor_path: str
    :param m: selected index to divide the acquisition into two sets: before m and after m ( run index: eg for 2 detector heads, 60 runs but 120 projections)
    if m is marked at the first gantry position affected by motion, m is called motion index m_0
    :type m: int
    :param em_slice: number of selected horizontal lines the within the projection image
    :type em_slice: list
    :param no_of_heads: number of detector heads, there are two available options: 1 or 2
    :type no_of_heads: int 
    '''

    def __init__(self, projections, attenuation_map, K_region, geometry,
                 m, em_slice, no_of_heads=1):
        if not isinstance(projections, ProjectionsClass):
            self.projections = ProjectionsClass(projections, geometry)
        else:
            self.projections = projections
        if not isinstance(K_region, VolumeClass):
            self.K_region = VolumeClass(K_region)
        else:
            self.K_region = K_region

        super(MotionCorrection, self).__init__(self.projections.itk_image, geometry)
        self.attenuation_map = attenuation_map
        self.ref_exponential_projection = ExponentialProjectionsGeometryClass(self.attenuation_map, self.geometry,
                                                                              like = self.projections.itk_image,  voxelized_region = self.K_region)
        # self.ref_exponential_projection.read_conversion_factor(conversion_factor_path)
        self.N_proj = len(self.angles_rad)
        self.m = m
        self.em_slice = em_slice
        self.no_of_heads = no_of_heads
        print("there is (are) ", self.no_of_heads, " detector head(s)")
        self.index_no_motion, self.index_with_motion = select_projection_affected_by_motion(self.N_proj,
                                                                                            self.m,
                                                                                            self.no_of_heads)
        self.mask_for_translation = self.create_masking_region()
        
    def create_masking_region(self, geometry_modify = None):
        """
            Define the masking region to mask out scatter coming outside of the K region
            This mask is basically the forward projection of K region
            """
        empty_projection = rtk.constant_image_source(information_from_image= self.projections.itk_image,
                                                     ttype=(itk.Image[itk.F, 3]))
        Joseph = rtk.JosephForwardProjectionImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New()
        if geometry_modify is None:
            Joseph.SetGeometry(self.geometry)
        else:
            Joseph.SetGeometry(geometry_modify)
        Joseph.SetInput(0, empty_projection)
        Joseph.SetInput(1, self.K_region.itk_image)
        Joseph.Update()
        masking_array = itk.GetArrayFromImage(Joseph.GetOutput())
        masking_array[masking_array != 0] = 1
        return masking_array


    def translation_correction_on_selected_projection(self, translation, index):
        '''
        Correct the rigid translation on a selected projection
        :param translation: A 3D vector representing the translation
        :type translation: list
        :param index: index of the projection
        :type index: int
        :return: itk image of the corrected projection
        :rtype itk.itkImagePython
        '''
        matrix = [[1, 0, 0.], [0, 1, 0], [0, 0, 1]]
        matrix[0][2] = translation[0] * math.cos(-self.angles_rad[index]) + translation[1] * math.sin(
            -self.angles_rad[index])
        matrix[1][2] = translation[2]
        matrixParameter = itk.matrix_from_array(matrix)
        itk_projection = itk.GetImageFromArray(self.numpy_image[index])
        itk_projection.SetOrigin(self.origin[:2])
        itk_projection.SetSpacing(self.spacing[:2])
        itk_projection.SetDirection(self.direction[:2, :2])
        shifted_itk_projection = gt.applyTransformation(itk_projection, matrix=matrixParameter,
                                                        keep_original_canvas=True)
        return shifted_itk_projection

    def correct_all_projections_after_motion(self, translation):
        '''
        Correct the rigid translation on all projections
        :param translation: A 3D vector representing the translation applied to all projections affected by motion
        :type translation: list
        :return: itk image of the corrected projections
        :rtype itk.itkImagePython
        '''
        shifted_np_image = self.numpy_image.copy()
        
        for i in self.index_with_motion:
            shifted_itk_projection = self.translation_correction_on_selected_projection(translation, i)
            shifted_np_projection = itk.GetArrayFromImage(shifted_itk_projection)
            shifted_np_image[i] = shifted_np_projection
        return shifted_np_image

    def compute_cost_function_with_only_translation(self, translation):
        """
            Define the cost function for an inputing motion vector

            :param translation: A 3D vector representing the translation
            :type translation: list
            :return: the mean of the EDCC
            :rtype: float
            """

        shifted_np_image = self.correct_all_projections_after_motion(translation)
        # masking out scatter coming outside of K region
        shifted_np_image= np.multiply(shifted_np_image, self.mask_for_translation)

        shifted_projection = ProjectionsClass(shifted_np_image, self.geometry, 
                                              spacing= self.spacing, origin= self.origin, direction= self.direction)

    
        edcc_computation = ExponentialProjectionsClass(shifted_projection, self.ref_exponential_projection)
        list_angle_rad = [[self.angles_rad[i], self.angles_rad[j]] for i in self.index_no_motion for j in
                          self.index_with_motion]
        edcc = edcc_computation.compute_edcc(self.em_slice, list_angle_rad=list_angle_rad)
        
        return np.mean(edcc)

    def compute_cost_function_rigid_motion(self, motion, mean=True, normalize = True):
        """
        Define the cost function for an inputing motion vector
        :param rotation: A 3D vector representing the rotation [angle_x, angle_y, angle_z]
        :type rotation: list
        :return: the mean of the EDCC
        :rtype: float
        """
        # print("Input motion ", motion)
        if normalize:
            # limited the range of motion to [-50, 50] for translation and [-pi/3, pi/3] for rotation to fasten the optimization
            motion = np.subtract(np.multiply(motion, [40, 40, 40, math.pi / 18, math.pi / 18, math.pi / 18]),
                                 [20, 20, 20, math.pi / 36, math.pi / 36, math.pi / 36])


        mu_zero = self.ref_exponential_projection.get_mu_zero()
        edcc = compute_edcc_in_parallel(self.projections, self.ref_exponential_projection,
                                        self.em_slice, motion, mu_zero,  m=self.m, no_of_heads=self.no_of_heads)
        if mean == True:
            edcc = np.mean(edcc)
        else:
            edcc = np.array(edcc)
        return edcc

    def compute_cost_function_rigid_motion_torch(self, motion, mean=True, normalize = True):
        """
        Define the cost function for an inputing motion vector
        :param rotation: A 3D vector representing the rotation [angle_x, angle_y, angle_z]
        :type rotation: list
        :return: the mean of the EDCC
        :rtype: float
        """
        # print("Input motion ", motion)
        if normalize:
            # limited the range of motion to [-50, 50] for translation and [-pi/3, pi/3] for rotation to fasten the optimization
            motion = np.subtract(np.multiply(motion, [40, 40, 40, math.pi / 18, math.pi / 18, math.pi / 18]),
                                 [20, 20, 20, math.pi / 36, math.pi / 36, math.pi / 36])


        mu_zero = self.ref_exponential_projection.get_mu_zero()
        edcc = compute_edcc_in_parallel_torch(self.projections, self.ref_exponential_projection,
                                        self.em_slice, motion, mu_zero,  m=self.m, no_of_heads=self.no_of_heads
                                              )
        if mean == True:
            edcc = np.mean(edcc)
        return edcc

class MotionAddFromTwoProjection(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Class to create a motion set at a given motion index from two sets of projections

    :param volume1: 3D volume, could be either path to the itk image file, numpy array or itk image
    :type volume1: Union[str, numpy.ndarray, itk.itkImagePython]
    :param volume2: 3D volume, could be either path to the itk image file, numpy array or itk image
    :type volume2: Union[str, numpy.ndarray, itk.itkImagePython]
    :param m: selected index to divide the acquisition into two sets: before m and after m ( run index: eg for 2 detector heads, 60 runs but 120 projections)
    :type m: int
    :param no_of_projections: number of projections
    :type no_of_projections: int
    :param no_of_heads: number of detector heads, there are two available options: 1 or 2
    :type no_of_heads: int
    """

    def __init__(self, volume1, volume2, m, no_of_projections, no_of_heads=1):
        self.spacing = None
        self.origin = None
        self.direction = None
        self.size = None
        self.itk_image = None
        self.numpy_image = None
        if not isinstance(volume1, VolumeClass):
            volume1 = VolumeClass(volume1)
        if not isinstance(volume2, VolumeClass):
            volume2 = VolumeClass(volume2)

        if volume1.get_physical_coordinates() != volume2.get_physical_coordinates():
            raise ValueError("Two images do not have the same  physical coordinates.")
        if volume1.itk_image is None:
            self.img = volume1
            self.volume1 = volume1
            self.volume2 = volume2
        else:
            self.img = volume1.itk_image
            self.volume1 = volume1.itk_image
            self.volume2 = volume2.itk_image
        self.get_physical_coordinates()
        self.size = volume1.size

        self.add_motion_from_array(m, no_of_projections, no_of_heads)

    def add_motion_from_array(self, m, no_of_projections, no_of_heads):
        """
        Read two itk images and combine 30 first projections of the first + 30 last projections of the second

    :param volume1: 3D volume, could be either path to the itk image file, numpy array or itk image
    :type volume1: Union[str, numpy.ndarray, itk.itkImagePython]
    :param volume2: 3D volume, could be either path to the itk image file, numpy array or itk image
    :type volume2: Union[str, numpy.ndarray, itk.itkImagePython]
        """
        array_set1 = itk.GetArrayFromImage(self.volume1)
        array_set2 = itk.GetArrayFromImage(self.volume2)
        print("m ", m)
        print("no of frames ", no_of_projections)
        if no_of_heads == 1:
            array_motion = np.concatenate((array_set1[:m], array_set2[m:no_of_projections]),
                                          axis=0)
        elif no_of_heads == 2:
            array_motion = np.concatenate(
                (array_set1[:int(m)], array_set2[int(m): int(no_of_projections / 2)],
                 array_set1[int(no_of_projections / 2): int(no_of_projections / 2 + m)],
                 array_set2[int(no_of_projections / 2 + m): int(no_of_projections)]), axis=0)
        else:
            print("not yet implemented")
        self.itk_image = itk.GetImageFromArray(array_motion)
        self.size = list(self.itk_image.GetLargestPossibleRegion().GetSize())
        self.origin = [-0.5 * (self.size[i] - 1) * self.spacing[i] for i in range(3)]
        self.origin[-1] = 0
        self.itk_image.SetOrigin(self.origin)
        self.itk_image.SetSpacing(self.spacing)


    def get_physical_coordinates(self):
        """
        Get the origin, spacing and direction of the volume from the ITK image.
        """
        if self.img is None:
            raise ValueError("ITK image should be set before getting the physical coordinates.")
        self.spacing = list(self.img.GetSpacing())
        self.origin = list(self.img.GetOrigin())
        self.direction = itk.GetArrayFromMatrix(self.img.GetDirection())

    def save_motion_img(self, output_path, name, index, no_of_projection):
        """
        Save the motion as an ITK image.

        :param output_path: Path where to save the conversion factor
        :type output_path: str
        """
        if self.itk_image is None:
            self.add_motion_from_array((index, no_of_projection))
        itk.imwrite(self.itk_image, os.path.join(output_path, name))

