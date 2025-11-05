#!/usr/bin/env python3

import argparse
import itk
import numpy as np
import gatetools as gt
from itk import RTK as rtk
from motion_correction import MotionCorrection,MotionAddFromTwoProjection


def main():
    print(args)

    translation = [float(t) for t in args.translation.split(",")]
    rotation = [float(r)*np.pi/180 for r in args.rotation.split(",")]
    motion = [translation[0], translation[1], translation[2],rotation[0], rotation[1], rotation[2]]

    A = np.dot([[np.cos(rotation[2]), -np.sin(rotation[2]), 0], [np.sin(rotation[2]), np.cos(rotation[2]), 0],
           [0, 0, 1]], np.dot([[np.cos(rotation[1]), 0, np.sin(rotation[1])], [0, 1, 0],
           [-np.sin(rotation[1]), 0, np.cos(rotation[1])]], [[1, 0, 0], [0, np.cos(rotation[0]), -np.sin(rotation[0])],
           [0, np.sin(rotation[0]), np.cos(rotation[0])]]))

    A = np.array([[A[0][0], A[0][1], A[0][2], 0],
                  [A[1][0], A[1][1], A[1][2], 0],
                  [A[2][0], A[2][1], A[2][2], 0],
                  [0, 0, 0, 1]])

    A_translate = np.array([[1, 0, 0, motion[0]],[0, 1, 0, motion[1]],[0, 0, 1, motion[2]],[0, 0, 0, 1]])
    B = (np.dot(A,A_translate))
    A_transpose = np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0],[0, 0, 0, 1]])
    B = np.dot(A_transpose,(np.dot(B, A_transpose)))
    # file = open("matrix_6dof_motion.mat", "w")
    # Write B without brackets
    # file.write(str(B).replace('[', '').replace(']', ''))
    # file.close()
    source_original = itk.imread(args.source)
    attmap_original = itk.imread(args.attenuationmap)
    projections_original = itk.imread(args.projections)
    projections_original_array = itk.array_from_image(projections_original)

    matrix = itk.matrix_from_array(B)

    source_shifted = gt.applyTransformation(input=source_original,matrix=matrix,force_resample=True)
    attmap_shifted = gt.applyTransformation(input=attmap_original,matrix=matrix,force_resample=True)

    if args.output_source is not None:
        itk.imwrite(source_shifted,args.output_source)

    Dimension = 3
    pixelType = itk.F
    imageType = itk.Image[pixelType, Dimension]
    forward_projector = rtk.JosephForwardAttenuatedProjectionImageFilter[imageType, imageType].New()

    zero_proj_array = np.zeros_like(projections_original_array)
    zero_proj_itk = itk.image_from_array(zero_proj_array)
    zero_proj_itk.CopyInformation(projections_original)

    forward_projector.SetInput(0, zero_proj_itk)
    forward_projector.SetInput(1, source_shifted)
    forward_projector.SetInput(2, attmap_shifted)


    xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    xmlReader.SetFilename(args.geom)
    xmlReader.GenerateOutputInformation()
    geometry = xmlReader.GetOutputObject()
    forward_projector.SetGeometry(geometry)

    forward_projector.Update()
    projections_shifted=forward_projector.GetOutput()
    itk.imwrite(projections_shifted, args.output_projections)
    rotation_angles = (np.asarray(geometry.GetGantryAngles()))
    N = len(rotation_angles)


    projection_with_motion = MotionAddFromTwoProjection(projections_original, projections_shifted,
                                                        m = args.m, no_of_heads = 2, no_of_projections= N)

    itk.imwrite(projection_with_motion.itk_image, args.output_projections)
    print(f"Output projections saved in : {args.output_projections}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--attenuationmap")
    parser.add_argument("--geom")
    parser.add_argument("-m", type = int, help="Index of the shift")
    parser.add_argument("--projections", help = "Original Projections, without motion")
    parser.add_argument("--translation", help = "Translation motion to apply, in mm. Ex --translation 10,20,30.")
    parser.add_argument("--rotation", help = "Rotation motion to apply, in deg. Ex --rotation -5,10,45.")
    parser.add_argument("--output_source", help= "Optional, in case ou want to save the shifted source.")
    parser.add_argument("--output_projections", help= "Optional, in case ou want to save the shifted source.")
    args = parser.parse_args()

    main()
