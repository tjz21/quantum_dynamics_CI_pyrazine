Raw data of diabatic energy gap and couplings used to generate chain coefficients 
for T-TEDOPA, and adiabatic energy gaps and dipole moment used as input for the 
Gaussian Non-Condon Theory can be found here.

To generate spectral densities of diabatic energies and couplings, run the 
compute_diabatic_sds.py script using the text files with diabatic energies
and couplings along the MD trajectory as input. The spectral_dens_to_chain
script can then be used to generate the appropriate T-TEDOPA chain coefficients.

For the Gaussian non-Condon theory, see the GNCT/ folder for an example input
file for the MolSpeckPy code to generate an optical spectrum. 
