import itk
import matplotlib.pyplot as plt
import numpy as np
from numbers import Number
import gatetools as gt

HANDLED_FUNCTIONS = {}


class VolumeClass(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Class to handle 3D images. This class works as a numpy array while keeping updated the itk image.

    :param volume: 3D volume, could be either path to the itk image file, numpy array or itk image
    :type volume: Union[str, numpy.ndarray, itk.itkImagePython]
    :param spacing: Spacing of the volume (only used if volume is a numpy array)
    :type spacing: list, optional
    :param origin: Origin of the volume (only used if volume is a numpy array)
    :type origin: list, optional
    :param direction: Direction of the volume (only used if volume is a numpy array)
    :type direction: numpy.ndarray, optional
    """

    def __init__(self, volume, spacing=None, origin=None, direction=None):
        self.spacing = None
        self.origin = None
        self.direction = None
        self.size = None
        self.itk_image = None
        self.numpy_image = None
        if isinstance(volume, str):
            self.itk_image = itk.imread(volume, itk.F)
            self.numpy_image = np.asarray(self.itk_image)
            self.get_physical_coordinates()
            self.size = list(self.itk_image.GetLargestPossibleRegion().GetSize())
        elif isinstance(volume, np.ndarray):
            self.numpy_image = volume
            if spacing is None or origin is None or direction is None:
                raise TypeError("When giving volume as numpy array, spacing, origin and direction must be specified.")
            self.itk_image = itk.GetImageFromArray(self.numpy_image.astype(np.float32))
            self.set_physical_coordinates(spacing, origin, direction)
            self.size = list(self.itk_image.GetLargestPossibleRegion().GetSize())
        else:
            self.itk_image = volume
            try:
                self.numpy_image = itk.GetArrayFromImage(self.itk_image)
                self.get_physical_coordinates()
                self.size = list(self.itk_image.GetLargestPossibleRegion().GetSize())
            except RuntimeError:
                raise TypeError("volume must be either a numpy array, a itk image or a path to an itk image.")

    _HANDLED_TYPES = (np.ndarray, Number)

    def set_physical_coordinates(self, spacing, origin, direction):
        """
        Set the physical coordinates of the volume.

        :param spacing: Spacing of the volume
        :type spacing: Union[list, numpy.ndarray]
        :param origin: Origin of the volume
        :type origin: Union[list, numpy.ndarray]
        :param direction: Direction of the volume
        :type direction: numpy.ndarray
        """
        self.itk_image.SetSpacing(spacing)
        self.itk_image.SetOrigin(origin)
        self.itk_image.SetDirection(direction)
        self.get_physical_coordinates()

    def get_physical_coordinates(self):
        """
        Get the origin, spacing and direction of the volume from the ITK image.
        """
        if self.itk_image is None:
            raise ValueError("ITK image should be set before getting the physical coordinates.")
        self.spacing = list(self.itk_image.GetSpacing())
        self.origin = list(self.itk_image.GetOrigin())
        self.direction = itk.GetArrayFromMatrix(self.itk_image.GetDirection())

    def get_index_to_physical_point_matrix(self):
        """
        Compute matrix to transform volume index to physical point (mm). The index [x, y, z] is the one of the volume in
        itk format and corresponds to the index [z, y, x] in numpy format.

        :return: Homogeneous matrix to transform volume index to physical point (mm).
        :rtype: numpy.ndarray
        """
        numpy_matrix = np.zeros((4, 4))
        self.get_physical_coordinates()
        for j in range(3):
            index = [0] * 3
            index[j] = 1
            point = self.itk_image.TransformIndexToPhysicalPoint(index)
            for i in range(3):
                numpy_matrix[i, j] = point[i] - self.origin[i]
        for i in range(3):
            numpy_matrix[i, 3] = self.origin[i]
        numpy_matrix[3, 3] = 1

        return numpy_matrix

    def get_physical_point_to_index_matrix(self):
        """
        Compute matrix to transform physical point (mm) to volume index. The index [x, y, z] is the one of the volume in
        itk format and corresponds to the index [z, y, x] in numpy format.

        :return: Homogeneous matrix to transform physical point (mm) to volume index.
        :rtype: numpy.ndarray
        """
        numpy_matrix = self.get_index_to_physical_point_matrix()
        itk_matrix = itk.GetMatrixFromArray(numpy_matrix)
        numpy_matrix = itk.GetArrayFromVnlMatrix(itk_matrix.GetInverse().as_matrix())

        return numpy_matrix

    def save(self, output_path):
        """
        Save the volume on the disk.

        :param output_path: Path where to save the volume
        :type output_path: str
        """
        if self.itk_image is None:
            raise ValueError("No volume to save")
        self.update()
        itk.imwrite(self.itk_image, output_path)

    def show(self, x, y, z, **kwargs):
        """
        Display a 2D slice of the volume specified by the x, y, z index.

        :param x: Index on the x-axis to be show. Could be a list of two integers ([x_min, x_max]),
            a string ("x_min:x_max") or an integer
        :type x: Union[int, str, list]
        :param y: Index on the x-axis to be show. Could be a list of two integers ([y_min, y_max]),
            a string ("y_min:y_max") or an integer
        :type y: Union[int, str, list]
        :param z: Index on the x-axis to be show. Could be a list of two integers ([z_min, z_max]),
            a string ("z_min:z_max") or an integer
        :type z: Union[int, str, list]
        :param kwargs: Additional parameters passed to the matplotlib imshow function.
        """
        x_min, x_max = extract_index(x)
        y_min, y_max = extract_index(y)
        z_min, z_max = extract_index(z)
        plt.imshow(self.numpy_image[z_min:z_max, y_min:y_max, x_min:x_max].squeeze(), **kwargs)
        plt.show()

    def update(self):
        """
        Update the itk image of the volume from the numpy image.
        """
        self.get_physical_coordinates()
        self.itk_image = itk.GetImageFromArray(self.numpy_image.astype(np.float32))
        self.set_physical_coordinates(self.spacing, self.origin, self.direction)

    def has_same_information(self, other_volume, tol=0.001):
        """
        Check if two volumes have the same information (size, spacing, origin, direction).

        :param other_volume: Volume.
        :type other_volume: VolumeClass
        :param tol: Tolerance for origin, spacing and direction.
        :type tol: float, optional
        :return: True or False.
        :rtype: bool
        """
        if not isinstance(other_volume, VolumeClass):
            raise TypeError("Wrong input type {}, must be of type VolumeClass".format(type(other_volume)))
        if (self.size == other_volume.size or
                np.all(
                    np.asarray(self.spacing) - np.asarray(other_volume.spacing) < np.ones_like(self.spacing) * tol) or
                np.all(
                    np.asarray(self.origin) - np.asarray(other_volume.origin) < np.ones_like(self.origin) * tol) or
                np.all(np.asarray(self.direction) - np.asarray(other_volume.direction) < np.ones_like(
                    self.direction) * tol)):
            return True
        else:
            return False

    def __array__(self, dtype=None):
        return self.numpy_image

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
                    return_list.append(VolumeClass(new_itk_volume))
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
                return VolumeClass(new_itk_volume)
            else:
                return result

    def __getitem__(self, item):
        return self.numpy_image[item]

    def __setitem__(self, item, val):
        self.numpy_image[item] = val
        self.update()

    def __str__(self):
        return self.itk_image.__str__()


def extract_index(input_index):
    """
    Extract the index of start and stop from the input.

    :param input_index: Could be a list of two integers ([start, stop]), a string
        ("start:stop") or an integer
    :type input_index: Union[list, str, int]

    :returns: Index min and max extract from the input parameter
    :rtype: int
    """
    if isinstance(input_index, str):
        if ":" in input_index:
            input_index = [int(i) if i.isdigit() else i for i in input_index.split(":")]
            if isinstance(input_index[0], int):
                ind_min = input_index[0]
            else:
                ind_min = 0
            if isinstance(input_index[1], int):
                ind_max = input_index[1]
            else:
                ind_max = -1
        else:
            ind_min = int(input_index)
            ind_max = int(input_index) + 1
    elif isinstance(input_index, list):
        if isinstance(input_index[0], int):
            ind_min = input_index[0]
        else:
            ind_min = 0
        if isinstance(input_index[1], int):
            ind_max = input_index[1]
        else:
            ind_max = -1
    elif isinstance(input_index, int):
        ind_min = input_index
        ind_max = input_index + 1
    else:
        raise (TypeError("Wrong input type {} for {}".format(type(input_index), input_index)))

    return ind_min, ind_max


def apply_affine_transform(input_volume, matrix, like=None, new_size=None, new_origin=None, new_spacing=None,
                           new_direction=None, pad=0., interpolation_mode="linear"):
    """
    Apply of 3D affine transform to the input volume based on the transformation matrix.

    :param input_volume: Input volume.
    :type input_volume: VolumeClass
    :param matrix: Homogeneous affine transform matrix.
    :type matrix: numpy.ndarray
    :param like: Copy information from this image for the output volume (origin, size, spacing, direction).
    :type like: str | itk.itkImagePython | VolumeClass, optional
    :param new_size: Size of the output volume.
    :type new_size: list | numpy.ndarray, optional
    :param new_origin: Origin of the output volume.
    :type new_origin: list | numpy.ndarray, optional
    :param new_spacing: Spacing of the output volume.
    :type new_spacing: list | numpy.ndarray, optional
    :param new_direction: Direction of the output volume.
    :type new_direction: numpy.ndarray, optional
    :param pad: Pad value. Default = 0.
    :type pad: float
    :param interpolation_mode: Interpolation used to resample the volume (NN for nearest neighbor,
        linear for linear interpolation and BSpline for BSpline interpolation).
    :type interpolation_mode: str

    :return: Transformed volume.
    :rtype: VolumeClass
    """
    if not isinstance(input_volume, VolumeClass):
        raise TypeError("Wrong type for input volume {}, should be of type VolumeClass".format(type(input_volume)))
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Wrong type for matrix {}, should be of type numpy.ndarray".format(type(matrix)))
    matrix = itk.matrix_from_array(matrix)
    if like is not None:
        if not isinstance(like, VolumeClass):
            like = VolumeClass(like)
        like = like.itk_image
    new_itk_volume = gt.applyTransformation(input_volume.itk_image, matrix=matrix, like=like, newsize=new_size,
                                            neworigin=new_origin, newspacing=new_spacing, newdirection=new_direction,
                                            pad=pad, interpolation_mode=interpolation_mode, force_resample=True,
                                            adaptive=False, keep_original_canvas=False)
    new_volume = VolumeClass(new_itk_volume)

    return new_volume
