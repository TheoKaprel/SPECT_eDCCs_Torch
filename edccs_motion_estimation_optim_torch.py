#!/usr/bin/env python3
# Welcome
# ------------------------------------------------------------------------------
# Load packages
from itk import RTK as rtk
from motion_correction import MotionCorrection,MotionAddFromTwoProjection
from exponential_projections import *
import time
import argparse
import os


def main():

    # ------------------------------------------------------------------------------
    # Geometry and references
    # ------------------------------------------------------------------------------
    geometryReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    geometryReader.SetFilename(args.geometry)
    geometryReader.GenerateOutputInformation()
    geometry = geometryReader.GetGeometry()
    attenuation_map = itk.imread(args.attenuationmap)
    K_region = VolumeClass(args.k_region)


    projection_with_motion_itk = itk.imread(args.projections)
    projection_with_motion = ProjectionsClass(projection_with_motion_itk, geometry)


    em_slice = [0, projection_with_motion.size[1]]
    print(f"em slice: ", em_slice)

    optim = MotionCorrection(projection_with_motion, attenuation_map, K_region, geometry,
                     30, em_slice, no_of_heads=2)


    # Compute cost of true motion
    true_motion = [10, 30, 60, -np.pi/10, -np.pi/15, -np.pi/20]
    true_motion_array = np.array(true_motion)
    print(f"True motion: {true_motion}")

    dev = torch.device("cuda")
    dtype=torch.float64
    min_motion = torch.Tensor([0,20,50, -torch.pi/10-torch.pi/10, -torch.pi/15-torch.pi/10, -torch.pi/20-torch.pi/10]).to(dev).to(dtype)
    max_motion = torch.Tensor([20,40,70, -torch.pi/10+torch.pi/10, -torch.pi/15+torch.pi/10, -torch.pi/20+torch.pi/10]).to(dev).to(dtype)


    motion_scaled = torch.rand(6, device=dev, dtype=dtype)
    motion_scaled = torch.nn.Parameter(motion_scaled)
    optimizer = torch.optim.Adam([motion_scaled,],lr = 0.01)
    nepochs = 100
    l_edcc = []
    l_mse_translation = []
    l_mse_rotation = []
    for epoch in range(nepochs):
        t0 = time.time()
        optimizer.zero_grad()
        motion_scaled_clamped = torch.clamp(motion_scaled, 0.0, 1.0)
        motion = min_motion + (max_motion - min_motion) * motion_scaled_clamped
        edccs_torch = optim.compute_cost_function_rigid_motion_torch(motion, normalize = False,mean=False)
        cost_torch = edccs_torch.mean()
        cost_torch.backward()
        optimizer.step()
        motion_array = motion.detach().cpu().numpy()
        print("Motion: ", motion_array.tolist())
        print(f"Grad: {motion_scaled.grad}")
        print(f"Epoch [{epoch}/{nepochs}] Cost function : ", cost_torch.item())
        print(f"TIME: {time.time()-t0}")
        print("-----------------------------")

        l_edcc.append(cost_torch.item())
        mse_t = np.sqrt(((true_motion_array[:3]-motion_array[:3])**2).sum())
        mse_r = np.sqrt(((true_motion_array[3:]-motion_array[3:])**2).sum())
        l_mse_translation.append(mse_t)
        l_mse_rotation.append(mse_r)


    np.save(os.path.join(args.output_folder,f"l_edccs_{nepochs}.npy"), l_edcc)
    np.save(os.path.join(args.output_folder,f"l_mse_rotation_{nepochs}.npy"), l_mse_rotation)
    np.save(os.path.join(args.output_folder,f"l_mse_translation_{nepochs}.npy"), l_mse_translation)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--attenuationmap")
    parser.add_argument("-k","--k_region")
    parser.add_argument("-g","--geometry")
    parser.add_argument("-p","--projections", help="Projections with motion")
    parser.add_argument("-o","--output_folder")
    args = parser.parse_args()

    main()
