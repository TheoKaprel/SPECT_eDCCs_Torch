import random
import itk
from itk import RTK as rtk
import numpy as np
from volume import VolumeClass


class ProjectionsClass(VolumeClass):
    """
    Class to handle SPECT projections.

    :param volume: 3D volume, could be either path to the itk image file, numpy array or itk image
    :type volume: Union[str, numpy.ndarray, itk.itkImagePython]
    :param geometry: Geometry of the acquisition
    :type geometry: itk.rtkThreeDCircularProjectionGeometryPython.rtkThreeDCircularProjectionGeometry
    :param spacing: Spacing of the volume (only used if volume is a numpy array)
    :type spacing: list, optional
    :param origin: Origin of the volume (only used if volume is a numpy array)
    :type origin: list, optional
    :param direction: Direction of the volume (only used if volume is a numpy array)
    :type direction: numpy.ndarray, optional
    :param attenuation_map: Attenuation map
    :type attenuation_map: Union[itk.itkImagePython, str, VolumeClass], optional
    """

    def __init__(self, volume, geometry, spacing=None, origin=None, direction=None, attenuation_map=None):
        super(ProjectionsClass, self).__init__(volume, spacing, origin, direction)
        self.geometry = None
        self.angles_deg = []
        self.angles_rad = []
        if geometry is not None:
            self.set_geometry(geometry)
        self.attenuation_map = None
        if attenuation_map is not None:
            self.set_attenuation_map(attenuation_map)
        self.s = np.linspace(self.origin[0], self.origin[0] + (self.size[0] - 1) * self.spacing[0], self.size[0])

    def set_geometry(self, geometry):
        """
        Set the geometry. A list of angles in radian and another of angles in degree are created.

        :param geometry: Geometry of the acquition
        :type geometry: itk.rtkThreeDCircularProjectionGeometryPython.rtkThreeDCircularProjectionGeometry
        """
        if not isinstance(geometry, itk.rtkThreeDCircularProjectionGeometryPython.rtkThreeDCircularProjectionGeometry):
            raise TypeError("Geometry must be of type "
                            "itk.rtkThreeDCircularProjectionGeometryPython.rtkThreeDCircularProjectionGeometry")
        self.geometry = geometry
        self.angles_rad = 1. * np.asarray(self.geometry.GetGantryAngles())
        self.angles_deg = np.rad2deg(self.angles_deg)

    def set_attenuation_map(self, attenuation_map, spacing=None, origin=None, direction=None):
        """
        Set attenuation map.

        :param attenuation_map: Attenuation map
        :type attenuation_map: Union[VolumeClass, str, numpy.ndarray, itk.ImagePython]
        :param spacing: Spacing of the volume (only used if attenuation_map is a numpy array)
        :type spacing: list, optional
        :param origin: Origin of the volume (only used if attenuation_map is a numpy array)
        :type origin: list, optional
        :param direction: Direction of the volume (only used if attenuation_map is a numpy array)
        :type direction: numpy.ndarray, optional
        """
        if not isinstance(attenuation_map, VolumeClass):
            self.attenuation_map = VolumeClass(attenuation_map, spacing, origin, direction)
        else:
            self.attenuation_map = attenuation_map

    def add_poisson_noise(self):
        """
        Add Poisson noise in the projections.
        """
        rng = np.random.default_rng()
        self.numpy_image = rng.poisson(self.numpy_image)
        self.update()

    def normalize(self, reference_projections):
        """
        Normalize by the mean the projections with the input reference projections.

        :param reference_projections: Projections used as reference for the normalization.
        :type reference_projections: ProjectionsClass | str | itk.itkImagePython | numpy.ndarray
        """
        if not isinstance(reference_projections, ProjectionsClass):
            reference_projections = ProjectionsClass(reference_projections, self.geometry, self.spacing, self.origin,
                                                     self.direction, self.attenuation_map)
        mean_ref_proj = np.sum(reference_projections.numpy_image) / reference_projections.numpy_image.shape[0]
        mean_proj = np.sum(self.numpy_image) / self.numpy_image.shape[0]
        self.numpy_image *= mean_ref_proj / mean_proj
        self.update()

    def has_same_information(self, other_projections, tol=0.001):
        """
        Check if two set of projections have the same informations (size, spacing, origin, direction and angles).

        :param other_projections: Projections.
        :type other_projections: ProjectionsClass
        :param tol: Tolerance for origin, spacing and direction and angles.
        :type tol: float, optional
        :return: True or False.
        :rtype: bool
        """
        if not isinstance(other_projections, ProjectionsClass):
            raise TypeError("Wrong input type {}, must be of type ProjectionClass".format(type(other_volume)))
        if (self.size == other_projections.size or
                np.all(
                    np.asarray(self.spacing) - np.asarray(other_projections.spacing) < np.ones_like(
                        self.spacing) * tol) or
                np.all(
                    np.asarray(self.origin) - np.asarray(other_projections.origin) < np.ones_like(self.origin) * tol) or
                np.all(np.asarray(self.direction) - np.asarray(other_projections.direction) < np.ones_like(
                    self.direction) * tol) or
                np.all(np.asarray(self.angles_rad) - np.asarray(other_projections.angles_rad) < np.ones_like(
                    self.angles_rad) * tol)):
            return True
        else:
            return False

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES..
            if not isinstance(x, self._HANDLED_TYPES + (VolumeClass,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.numpy_image if isinstance(x, VolumeClass) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.numpy_image if isinstance(x, VolumeClass) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return_list = []
            for res in result:
                if isinstance(res, np.ndarray):
                    new_itk_volume = itk.GetImageFromArray(res.astype(np.float32))
                    new_itk_volume.CopyInformation(self.itk_image)
                    return_list.append(
                        ProjectionsClass(new_itk_volume, self.geometry, attenuation_map=self.attenuation_map))
                else:
                    return_list.append(res)
            return tuple(return_list)
        elif method == 'at':
            # no return value
            return None
        else:
            if isinstance(result, np.ndarray):
                new_itk_volume = itk.GetImageFromArray(result.astype(np.float32))
                new_itk_volume.CopyInformation(self.itk_image)
                return ProjectionsClass(new_itk_volume, self.geometry, attenuation_map=self.attenuation_map)
            else:
                return result


def compute_scatter_correction(projections_principal, projections_scatter, k=1.1):
    """
    Compute the scatter correction with the Double Energy Window method.

    :param projections_principal: Projections of the principal energy window.
    :type projections_principal: ProjectionsClass
    :param projections_scatter: Projections of the scatter energy window.
    :type projections_scatter: ProjectionsClass
    :param k: Scatter multiplier ("k" value) used for the correction.
    :type k: float
    :return: Projections corrected.
    :rtype: ProjectionsClass
    """
    if not isinstance(projections_principal, ProjectionsClass) or not isinstance(projections_scatter, ProjectionsClass):
        raise TypeError("Projections must be of type ProjectionsClass")
    projections_corrected = np.subtract(projections_principal.numpy_image, k * projections_scatter.numpy_image,
                                        out=np.zeros_like(projections_principal.numpy_image),
                                        where=projections_principal.numpy_image > k * projections_scatter.numpy_image)
    projections_corrected = itk.GetImageFromArray(projections_corrected.astype(np.float32))
    projections_corrected.CopyInformation(projections_principal.itk_image)
    projections_corrected = ProjectionsClass(projections_corrected, projections_principal.geometry)

    return projections_corrected


def read_geometry(geometry_path):
    """
    Read the RTK geometry from a xml geometry file.

    :param geometry_path: Path to the xml geometry file.
    :type geometry_path: str
    :return: RTK geometry.
    :rtype: itk.rtkThreeDCircularProjectionGeometryPython.rtkThreeDCircularProjectionGeometry
    """
    xml_reader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    xml_reader.SetFilename(geometry_path)
    xml_reader.GenerateOutputInformation()
    geometry = xml_reader.GetOutputObject()

    return geometry


def write_geometry(geometry, geometry_path):
    """
    Write the RTK geometry in a xml geometry file.

    :param geometry: RTK geometry object.
    :type geometry: itk.rtkThreeDCircularProjectionGeometryPython.rtkThreeDCircularProjectionGeometry
    :param geometry_path: Path where to write the xml geometry file.
    :type geometry_path: str
    """
    xml_writer = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
    xml_writer.SetFilename(geometry_path)
    xml_writer.SetObject(geometry)
    xml_writer.WriteFile()


def mix_projections(proj_1, proj_2, nb_mix=None, position_mix=None):
    """
    Take two sets of projections and mix them to create a new set of projections. This function is mostly used to
    create a set of projections with movement in it.

    :param proj_1: First set of projections.
    :type proj_1: ProjectionsClass
    :param proj_2: Second set of projections.
    :type proj_2: ProjectionsClass
    :param nb_mix: Number of mixes to do. If None, a random number between 0 and 10 is generated.
    :type nb_mix: int, optional
    :param position_mix: Indexes of the angle where the mixes occured. If None, the indexes are randomly chosen.
    :type position_mix: int | list[int], optional

    :returns: Projections corresponding to the mix of the first two inputs and the position of the mixes.
    :rtype: tuple(ProjectionsClass, list[int])
    """
    if not isinstance(proj_1, ProjectionsClass) or not isinstance(proj_2, ProjectionsClass):
        raise TypeError("Input projections must be of type ProjectionClass")
    if not proj_1.has_same_information(proj_2):
        raise ValueError("The two projections must have the same information (origin, spacing, direction and angles)")
    if nb_mix is None:
        nb_mix = random.randint(0, 10)
        position_mix = random.sample(range(1, proj_1.size[2]), nb_mix)
    else:
        if not isinstance(nb_mix, int):
            raise TypeError("Wrong input type {}, nb_mix must be of type int".format(type(nb_mix)))
        if isinstance(position_mix, int):
            position_mix = [position_mix]
        if isinstance(position_mix, list):
            if not all(isinstance(x, int) for x in position_mix):
                raise TypeError("Wrong input type, position_mix must be type int or list of int")
            elif len(position_mix) != nb_mix:
                raise ValueError("Length of position_mix must be equal to nb_mix")
        elif position_mix is None:
            position_mix = random.sample(range(proj_1.size[2]), nb_mix)
        else:
            raise TypeError("Wrong input type, position_mix must be type int or list of int")
    position_mix.sort()
    new_proj = ProjectionsClass(proj_1.itk_image, proj_1.geometry)
    ref_proj = proj_1
    move_proj = proj_2
    prev_index = 0
    for index in position_mix:
        new_proj[prev_index:index, :, :] = ref_proj[prev_index:index, :, :]
        new_proj[index:, :, :] = move_proj[index:, :, :]
        ref_proj, move_proj = move_proj, ref_proj
        prev_index = index
    new_proj.update()

    return new_proj, position_mix
