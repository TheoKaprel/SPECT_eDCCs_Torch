# SPECT_eDCCs_Torch


### Conversion Factor computation

To just compute the conversion factor with PyTorch:

    python edccs_conversion_factor_computation.py --attenuationmap attenuation_map.mha --k_region K_region.mha --geometry geom.xml --projections_original projections.mha -o cf.mha

### eDCCs computation

To compute (with torch) and plot eDCCs:

    python edccs_computation.py --attenuationmap attenuation_map.mha --k_region K_region.mha --geometry geom.xml --projections projections.mha

### Apply Motion


To apply motion to a set of projection, for testing, you need the true source/attenuationmap, the original geometry and run the following command:

    python apply_motion.py --translation 10,30,60 --rotation 18,12,9 --source source_original.mha --output_source sourge_shifted.mha --output_projections projections_shifted.mha --attenuationmap attenuation_map_original.mha --geom geom.xml --projections projection_rtk_original.mha -m 20


### Motion estimation

To estimate the motion, from shifted projections, with a gradient descent implemented in PyTorch (Adam Optimizer), use the following command:

    python edccs_motion_estimation_optim_torch.py --attenuationmap attenuation_map_original.mha --k_region k_region.mha --geometry geom.xml --projections projections_shifted.mha --output_folder .