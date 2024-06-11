"Pyrazine" contains the MD data used to construct the pyrazine spectral densities.

"pyrazine_water_cyclohexane" contains the water and cyclohexane scripts and chain coefficients generated from the MD data. For vacuum scripts and chain coefficients, see "si_pyrazine_vac_correlations"/"si_pyrazine_vac_convergences".

"msa30", "msa200", "msb30", "msb200" contain the absorption (constructed from the 203.2 fs trajectory) and 60 sampled emission lineshapes which need to be averaged to form the predicted lineshape.

"msa200_jld" contains the hdf5 files which store the s0, s1, s2 populations throughout the trajectory.

The "si_[...]" directories contains the data needed to reconstruct the SI figures.

ms_spectral densities contains the spectral densities used to construct the model system chain coefficients. Different cross-correlation spectral densities can be constructed by scaling with -1 or 0.

"si_pyrazine_vac_coupling_scaling" contains the data used to construct the coupling scaling figure in the SI. Likewise for "si_pyrazine_vac_correlations" and "si_pyrazine_vac_convergences".
