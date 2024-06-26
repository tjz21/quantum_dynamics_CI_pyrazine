## Computing linear optical spectra in complex environments
Publication data for "Computing linear optical spectra of molecules in complex environments on Graphics Processing Units using molecular 
dynamics simulations and tensor-network approaches" by Evan Lambertson, Dayana Bashirova, Kye E. Hunter, Benhardt Hansen, and Tim J. Zuehlsdoff.

This repository contains all raw data for results presented in both the main text and the SI (Results), as well as input files necessary
to reproduce all calculations performed in this work, including the model systems (Model_systems), molecular dynamics simulations 
(Molecular_dynamics), the quantum dynamics with T-TEDOPA (Quantum_dynamics), as well as spectra in the Gaussian Non-Condon Theory 
produced with the MolSpeckPy code (GNCT). 

Additionally, the repository contains a number of python scripts to process some of the data generated by certain calculations.

Tensor network dynamics were run using the MPSDynamics package (github.com/shareloqs/MPSDynamics.git) with Julia version 1.8.1.

```bash
├── Model_systems # Spectral densities, chain coefficients (hdf5), and MPSdynamics scripts for each model system
│   ├── MSA200
│   ├── MSA30
│   ├── MSB200
│   └── MSB30
├── Molecular_dynamics # Input files used to perform MM equilibration, QM/MM MD, and TDDFT calculations  
│   ├── Cyclohexane
│   │   ├── Cyc_box
│   │   ├── MM_equilibration
│   │   ├── QM:MM
│   │   └── TDDFT
│   └── Water
│       ├── MM_equilibration
│       ├── QM:MM
│       └── TDDFT
├── Python_scripts # Python scripts used to generate the thermalize spectral densities/generate chain coefficients and compute linear response lineshapes
├── Quantum_dynamics # Chain coefficients and MPSDynamics_scripts used to compute pyrazine vacuum/water/cyclohexane trajectories
│   ├── Chain_coefficients
│   └── MPSdynamics_scripts
└── Results # Raw data for all results presented in the main manuscript and SI
    ├── Main_text
    │   ├── GCT
    │   ├── GNCT
    │   ├── MSA200
    │   ├── MSA30
    │   ├── MSB200
    │   │   └── MSB200_populations
    │   ├── MSB30
    │   ├── Pyrazine_absorption_environment
    │   └── S1_populations
    └── SI
        ├── Model_system_correlations
        │   ├── MSA200_FNC
        │   ├── MSA200_FPC
        │   └── MSA200_UC
        ├── Pyrazine_vac_convergences
        │   ├── Bond_dimension
        │   ├── Chain_length
        │   └── Fock_states
        ├── Pyrazine_vac_correlations
        └── Pyrazine_vac_coupling_scaling


