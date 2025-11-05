# SPECT_eDCCs_Torch


### Conversion Factor computation

To just compute the conversion factor with PyTorch:

    python edccs_conversion_factor_computation.py --attenuationmap attenuation_map.mha --k_region K_region.mha --geometry geom.xml --projections_original projections.mha -o cf.mha

### eDCCs computation

To compute (with torch) and plot eDCCs:

    python edccs_computation.py --attenuationmap attenuation_map.mha --k_region K_region.mha --geometry geom.xml --projections projections.mha

### Motion estimation


