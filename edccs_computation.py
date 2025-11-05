#!/usr/bin/env python3

import argparse
import torch
import itk
from itk import RTK as rtk
import time
import numpy as np
import matplotlib.pyplot as plt


from exponential_projections import conversion_factor_vectorized,_compute_edcc_all_pairs_torch
from volume import extract_index
from projections import ProjectionsClass

def main():
    print(args)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    attenuation_map = itk.imread(args.attenuationmap)
    K_region = itk.imread(args.k_region)
    projections_itk = itk.imread(args.projections)
    projections_tensor = torch.from_numpy(itk.array_from_image(projections_itk)).to(device)
    Nprojs = projections_tensor.shape[0]
    geometryReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    geometryReader.SetFilename(args.geometry)
    geometryReader.GenerateOutputInformation()
    geometry = geometryReader.GetGeometry()
    with torch.no_grad():
        attenuation_map_tensor = torch.from_numpy(itk.array_from_image(attenuation_map)).to(device)
        K_region_tensor = torch.from_numpy(itk.array_from_image(K_region)).to(device)
        mu0 = attenuation_map_tensor[K_region_tensor > 0].mean()
        t0 = time.time()
        conversion_factor_tensor = conversion_factor_vectorized(volume=attenuation_map_tensor[None,:,:,:],
                                                                kregion=K_region_tensor[None,:,:,:],
                                                                geometry=geometry,mu0=mu0)

        em_slice = [0, projections_tensor.shape[1]]
        print("Time to compute conversion factor: ", round(time.time() - t0,3), " s")

        t0 = time.time()
        angles_rad = np.asarray(geometry.GetGantryAngles())

        list_indices = [[i, j] for i in range(Nprojs) for j in
                        range(Nprojs)
                        if (np.abs(angles_rad[i] - angles_rad[j]) > np.deg2rad(1)) and (
                                np.abs(np.abs(angles_rad[i] - angles_rad[j]) - np.pi) > np.deg2rad(
                            1))]  # prevent exponential term going to inf
        angles_rad_tensor = torch.from_numpy(angles_rad).to(device)

        ind_min, ind_max = extract_index(em_slice)
        list_slice = np.arange(ind_min, ind_max)

        tensor_indices_ij = torch.Tensor(list_indices).to(device).to(torch.int)
        spacing = np.array(projections_itk.GetSpacing())[:2]
        projection = ProjectionsClass(projections_itk, geometry)
        s_tensor = torch.Tensor(projection.s).to(device)
        A_torch = torch.eye(3).to(device)
        delta_s_tensor,delta_l_tensor = torch.zeros_like(angles_rad_tensor),torch.zeros_like(angles_rad_tensor)
        list_eddcs = []
        batch_size = 1024
        N_index = tensor_indices_ij.shape[0]
        p=0
        while p<N_index:
            p_max = min(p+batch_size, N_index)
            edccs = _compute_edcc_all_pairs_torch(tensor_indices_ij[p:p_max], spacing,
                                      angles_rad_tensor,
                                      projections_tensor,
                                      conversion_factor_tensor,
                                      mu0, list_slice, A_torch,
                                      delta_s_tensor, delta_l_tensor,
                                      s_tensor)
            list_eddcs.append(edccs)
            p=p_max

        list_eddcs = torch.cat(list_eddcs, dim=0)
        print("Time to compute eDCCs: ", round(time.time() - t0,3), " s")
        print(list_eddcs.shape[0], " pairs")
        print(f"Mean: {list_eddcs.mean()}")
        print(f"Std: {list_eddcs.std()}")

        fig,ax = plt.subplots()
        ax.hist(list_eddcs.detach().cpu().numpy(), bins=100)
        ax.set_title("eDCC values for all pairs")
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--attenuationmap")
    parser.add_argument("-k","--k_region")
    parser.add_argument("-g","--geometry")
    parser.add_argument("-p","--projections")
    parser.add_argument("-o","--output")
    args = parser.parse_args()

    main()
