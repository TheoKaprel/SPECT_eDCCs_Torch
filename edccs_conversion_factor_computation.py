#!/usr/bin/env python3

import argparse
import torch
import itk
from itk import RTK as rtk
import time

from exponential_projections import conversion_factor_vectorized

def main():
    print(args)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attenuation_map = itk.imread(args.attenuationmap)
    K_region = itk.imread(args.k_region)
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
    conversion_factor_itk = itk.image_from_array(conversion_factor_tensor.detach().cpu().numpy())

    conversion_factor_itk.CopyInformation(itk.imread(args.projections_original))
    print("Time to compute conversion factor: ", time.time() - t0)
    itk.imwrite(conversion_factor_itk, args.output)
    print(f"Saved in : {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--attenuationmap")
    parser.add_argument("-k","--k_region")
    parser.add_argument("-g","--geometry")
    parser.add_argument("-p","--projections_original")
    parser.add_argument("-o","--output")

    args = parser.parse_args()

    main()
