import itk
import vtk
from volume import VolumeClass


class SurfaceClass(object):
    """
    Class to handle surface of a volume. The surface can be read from a stl file or can be passed as a VTK image.

    :param surface: surface, could be either path to the stl file, or a VTK image.
    :type surface: Union[str, vtkImageData]
    """

    def __init__(self, surface):
        self.surface = None
        if isinstance(surface, str):
            self.read_stl(surface)
        elif isinstance(surface, (vtk.vtkImageData, vtk.vtkPolyData)):
            self.surface = surface
        else:
            raise TypeError("Wrong type for surface {}, must be of type str or vtkImageData".format(type(surface)))

    def read_stl(self, stl_path):
        """
        Read surface from stl file.

        :param stl_path: Path to the stl file.
        :type stl_path: str
        """
        reader_stl = vtk.vtkSTLReader()
        reader_stl.SetFileName(stl_path)
        reader_stl.Update()
        self.surface = reader_stl.GetOutput()

    def write_stl(self, stl_path):
        """
        Write surface into a stl file.

        :param stl_path: Path to the stl file.
        :type stl_path: str
        """
        writer = vtk.vtkSTLWriter()
        writer.SetInputData(self.surface)
        writer.SetFileTypeToBinary()
        writer.SetFileName(stl_path)
        writer.Write()

    def show(self):
        """
        Show the surface in a vtk render window.
        """
        colors = vtk.vtkNamedColors()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.surface)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetDiffuseColor(colors.GetColor3d('LightSteelBlue'))
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(60.0)

        # Create a rendering window and renderer
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetWindowName('ReadSTL')

        # Create a renderwindowinteractor
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        # Assign actor to the renderer
        ren.AddActor(actor)
        ren.SetBackground(colors.GetColor3d('DarkOliveGreen'))

        # Enable user interface interactor
        iren.Initialize()
        renWin.Render()
        iren.Start()


class VolumeToSurface(object):
    """
    Extract surface from a volume using VTK marching cubes algorithm.

    :param volume: Volume.
    :type volume: VolumeClass
    """

    def __init__(self, volume):
        if isinstance(volume, VolumeClass):
            self.volume = volume
            self.vtk_volume = itk.vtk_image_from_image(volume.itk_image)
        else:
            raise TypeError("Wrong type for volume {}, must be of type VolumeClass".format(type(volume)))
        self.surface = None

    def extract_surface(self):
        """
        Extract surface of volume using marching cubes algorithm.

        :return: Surface.
        :rtype: SurfaceClass
        """
        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputData(self.vtk_volume)
        dmc.GenerateValues(1, 1, 1)
        dmc.Update()
        self.surface = SurfaceClass(dmc.GetOutput())

        return self.surface
