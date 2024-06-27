#! /usr/bin/env python

#     compute_diabatic_sds.py
#  
# Routine takes as an input diabatic energies, oscillator strengths, and 
# diabatic couplings between two or more excited states and computes four kinds
# of spectral densities (state-energy auto-correlations, coupling 
# auto-correlations, energy-energy cross-correlations, and energy-coupling 
# cross-correlations), as well as the average value of each data set and the 
# reorganization energy of the spectral densities.
#
# 1 required command-line argument:
#   $1: input file name. The file should be organized in columns with all the 
#     state energies and couplings first, in the order they would appear in an 
#     upper triangular matrix, and the oscillator strengths afterwards: 
#       S1   S1-S2 S1-S3 ...
#            S2    S2-S3 ...
#                  S3    ...
#                        ...
#       OS_S1 OS_S2 OS_S3...
#     All the input energies should be in eV.
# 5 parameters are specified in the `main` function that might need to be 
# adjusted by the user: 
#   time_step_in_fs=float (default=2.0): the timestep between individual data 
#     points in the energy gap fluctuations provided as input
#   T=float (default=300.0): the temperature at which the original classical MD 
#     that produced the data was run
#   interpolation_density=int (default=8): the number of points added to the 
#     trajectory using a cublic spline, e.g., 2 would double the resolution of 
#     the trajectory before taking Fourier transforms, increasing the maximum 
#     frequency of the spectral densities.
#   padding_factor=int (default=8): the number of points added the end of the 
#     trajectory by adding zeros, e.g., 2 will double the length of the 
#     trajectory, doubling the resolution of the resulting spectral densities.
#   decay_constant=float (default=500.0): A decay constant (in fs) for a 
#     decaying exponential that is applied to all autocorrelation functions in 
#     the time domain. This guarantees well-behaved Fourier transforms by 
#     forcing the correlation function to go to zero in the long timescale 
#     limit. It is equivalent to a Lorentzian broadening applied to the entire 
#     range of the spectral density. 
#
# Last updated 20240226 by Kye Hunter
# Written by Tim J. Zuehlsdorff and Kye E. Hunter.
# Based on routines from the MolSpeckPy package by Tim J. Zuehlsdorff, available 
# on GitHub: https://github.com/tjz21/Spectroscopy_python_code

import numpy as np
import sys, math, scipy.interpolate
from scipy import integrate

# Global constants 
pi           = 3.141592653589793
kb_in_eV     = 8.617333262145179e-05
eV_to_Ha     = 0.03674932217565436
fs_to_Ha     = 41.341373335182446
hbar_in_eVfs = 0.6582119569509066

def svsum(scalar, vec):
    '''Add a scalar to each element in a list'''
    return [scalar + v for v in vec]

def svprod(scalar, vec):
    '''Multiply each element in a list by a scalar'''
    return [scalar * v for v in vec]

def smprod(scalar, mat):
    '''Multiply each element in a list of lists by a scalar'''
    return [svprod(scalar, row) for row in mat]

def vprod(vec1, vec2):
    '''Multiply each pair of elements in two lists'''
    return [v1 * v2 for v1, v2 in zip(vec1, vec2)]

def read_dat(file_name):
    '''Read a space delimited data file.'''
    return list(zip(*[
        list(map(float, line.strip().split()))
        for line in open(file_name)
        if len(line.strip()) > 0 and (
            line.strip()[0].isnumeric() or line.strip()[0].startswith('-')
        )
    ]))

def write_dat(data, file_name):
    '''Write a list of lists as a space delimited table'''
    output_file = open(file_name, 'w')
    output_file.write('\n'.join([
        ('{:19.12e} ' * len(line)).strip().format(*line)
        for line in data
    ]))

def compute_reorg(spectral_dens):
    '''Calculate the reorganization energy of an SD.'''
    integrant = [
        (dens / pi) / freq if freq != 0 else 0 
        for freq, dens in spectral_dens
    ]
    return integrate.simps(
        integrant, 
        dx = spectral_dens[1][0] - spectral_dens[0][0]
    )

def compute_spectral_dens(corr_func, kbT, sample_rate, time_step):
    '''Apply a fourier transform and harmonic prefactor to a correlation func.
    '''
    corr_freq = time_step * np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(corr_func))
    )

    dim = int((corr_freq.shape[-1] + 1) / 2.0)
    shift_index = corr_freq.shape[-1] - dim

    freqs = np.fft.fftshift(
        np.fft.fftfreq(len(corr_func), d = 1.0 / sample_rate)
    ) * eV_to_Ha
    spectral_dens = [
        [freqs[i + shift_index], (
            freqs[i + shift_index] * 0.5 / kbT * corr_freq[i + shift_index].real
        )] for i in range(dim)
    ]

    return spectral_dens

def main(
    filename, 
    time_step_in_fs = 2.0, 
    T = 300.0, 
    interpolation_density = 8,
    padding_factor = 8,
    decay_constant = 500.0
):
    '''Caclulate spectral densities from trajectory data.
    
    Extract input parameters of excited state energies and coupling energy gap 
    fluctuations. Find the mean, build correlation functions, and then finally 
    construct spectral densities for excited-state auto- and cross-correlations 
    and the couplings. The spectral densities are stored as text files with the 
    first column denoting the frequency in Ha, the second column the intensity 
    in Ha. 
    '''
    # Read user input 
    print(
        f'\nGenerating spectral densities:\n'
        f'Input file name: {filename}\n'
        f'Fluctuation timestep in fs: {time_step_in_fs}\n'
        f'Temperature of the underlying MD simulation: {T}\n'
        f'MD interpolation density: {interpolation_density}\n'
        f'Fourier transform padding factor: {padding_factor}\n'
        'Decay constant (in fs) applied to all correlation funcs: '
            f'{decay_constant}\n'
    )

    # Some derived constants
    tau = decay_constant * fs_to_Ha
    # Time step of the fluctuations in a.u
    time_step = time_step_in_fs * fs_to_Ha
    kbT = kb_in_eV * T * eV_to_Ha
    sample_rate = hbar_in_eVfs * pi * 2.0 / time_step_in_fs

    # Read the input data
    input_vals = read_dat(filename)
    
    # Determine the number of excitations, and sort the input data out
    num_states = int(((len(input_vals) * 8 + 9) ** 0.5 - 3) / 2)
    energy_gap_cols = [
        int(state * num_states - state * (state - 1) / 2) 
        for state in range(num_states)
    ]
    energy_gaps = [input_vals[col] for col in energy_gap_cols]
    oscillator_strength_cols = list(range(
        len(input_vals) - num_states, 
        len(input_vals)
    ))
    oscillator_strengths = [input_vals[col] for col in oscillator_strength_cols]
    coupling_cols = sorted(
        set(range(len(input_vals))) 
        - set(energy_gap_cols) 
        - set(oscillator_strength_cols)
    )
    couplings = [input_vals[col] for col in coupling_cols]

    # Convert energies to atomic units
    energy_gaps = smprod(eV_to_Ha, energy_gaps)
    couplings = smprod(eV_to_Ha, couplings)

    # Find the means of everything
    mean_energy_gaps = [sum(state) / len(state) for state in energy_gaps]
    mean_oscillator_strengths = [
        sum(state) / len(state) for state in oscillator_strengths
    ]
    mean_couplings = [sum(state) / len(state) for state in couplings]
    mean_dipole_moments = [
        (oscillator / energy * 3.0 / 2.0) ** 0.5 
        for energy, oscillator 
        in zip(mean_energy_gaps, mean_oscillator_strengths)
    ]
    
    for istate in range(num_states):
        print(
            f'Diabatic state {istate + 1}:\n'
            f'Average energy (Ha):          {mean_energy_gaps[istate]}\n'
            'Average oscillator strength:  '
                f'{mean_oscillator_strengths[istate]}\n'
            f'Average dipole moment (a.u.): {mean_dipole_moments[istate]}\n'
        )
    coupling_list = [
        [num_states - y + 1, num_states - z + 2] 
        for y in range(num_states, 1, -1) for z in range(y, 1, -1)
    ]
    print(
        'Average couplings (Ha):\n' + '\n'.join(
            [
                f'S{coupling[0]}-S{coupling[1]}: {mean_couplings[icoupling]}'
                for icoupling, coupling in enumerate(coupling_list)
            ]
        ) + '\n'
    )

    # Shift the energy gaps and couplings to fluctuations around the mean
    energy_gaps = [
        svsum(-1.0 * mean, state) 
        for mean, state in zip(mean_energy_gaps, energy_gaps)
    ]
    couplings = [
        svsum(-1.0 * mean, coupling)
        for mean, coupling in zip(mean_couplings, couplings)
    ]

    # Interpolate the trajectory data so the spectral density is defined at 
    # higher frequencies (and to reduce high frequency noise)
    energy_splines = [
        scipy.interpolate.CubicSpline(range(len(state)), state) 
        for state in energy_gaps
    ]
    coupling_splines = [
        scipy.interpolate.CubicSpline(range(len(coupling)), coupling) 
        for coupling in couplings
    ]
    energy_gaps = [[
        spline(val / interpolation_density) 
        for val in range(int((len(state) - 1) * interpolation_density + 1))
    ] for state, spline in zip(energy_gaps, energy_splines)]
    couplings = [[
        spline(val / interpolation_density) 
        for val in range(int((len(coupling) - 1) * interpolation_density + 1))
    ] for coupling, spline in zip(couplings, coupling_splines)]
    
    time_step /= interpolation_density
    sample_rate *= interpolation_density

    # Compute correlation functions using numpy.correlate()
    # Find the self correlations:
    energy_gap_self_correlations = [
        np.correlate(state, state, mode = 'full') 
        for state in energy_gaps
    ]
    coupling_self_correlations = [
        np.correlate(coupling, coupling, mode = 'full') 
        for coupling in couplings
    ]

    # Find the cross correlation between each energy fluctuation with each other 
    # energy fluctuation, and for each coupling find the correlations with the 
    # two energy fluctuations it's associated with
    energy_cross_energy_correlations = [
        np.correlate(energy_gaps[istate1], energy_gaps[istate2], mode = 'full')
        for istate1 in range(num_states) for istate2 in range(num_states) 
        if istate1 < istate2
    ]
    energy_cross_coupling_list = [
        [num_states - y, num_states - z + 1] 
        for y in range(num_states, 1, -1) for z in range(y, 1, -1)
    ]
    energy_cross_coupling_correlations = [
        np.correlate(energy_gaps[istate], couplings[icoupling], mode = 'full')
        for icoupling in range(len(couplings)) for istate in range(num_states)
        if istate in energy_cross_coupling_list[icoupling]
    ]

    # Multiply each correlation function by a decaying exponential, and divide 
    # by (half of) the dimesion of the correlation function
    eff_decay_length = tau / time_step
    len_corr_func = len(energy_gap_self_correlations[0]) // 2
    decaying_exp_scaling = [
        math.exp(-abs(x) / eff_decay_length) / len_corr_func 
        for x in range(-1 * len_corr_func, len_corr_func + 1)
    ]
    
    energy_gap_self_correlations = [
        vprod(state, decaying_exp_scaling) 
        for state in energy_gap_self_correlations
    ]
    coupling_self_correlations = [
        vprod(coupling, decaying_exp_scaling)
        for coupling in coupling_self_correlations
    ]
    energy_cross_energy_correlations = [
        vprod(function, decaying_exp_scaling) 
        for function in energy_cross_energy_correlations
    ]
    energy_cross_coupling_correlations = [
        vprod(function, decaying_exp_scaling) 
        for function in energy_cross_coupling_correlations
    ]

    # Pad the correlation functions with 0s so the spectral density is defined 
    # more densely
    energy_gap_self_correlations = [
        [0.0] * (padding_factor - 1) * (len(function) // 2) + function 
        + [0.0] * (padding_factor - 1) * (len(function) // 2) 
        for function in energy_gap_self_correlations
    ]
    coupling_self_correlations = [
        [0.0] * (padding_factor - 1) * (len(function) // 2) + function 
        + [0.0] * (padding_factor - 1) * (len(function) // 2) 
        for function in coupling_self_correlations
    ]
    energy_cross_energy_correlations = [
        [0.0] * (padding_factor - 1) * (len(function) // 2) + function 
        + [0.0] * (padding_factor - 1) * (len(function) // 2) 
        for function in energy_cross_energy_correlations
    ]
    energy_cross_coupling_correlations = [
        [0.0] * (padding_factor - 1) * (len(function) // 2) + function 
        + [0.0] * (padding_factor - 1) * (len(function) // 2) 
        for function in energy_cross_coupling_correlations
    ]
    
    # Calculate the spectral densities
    energy_spectral_densities = [
        compute_spectral_dens(corr, kbT, sample_rate, time_step) 
        for corr in energy_gap_self_correlations
    ]
    coupling_spectral_densities = [
        compute_spectral_dens(corr, kbT, sample_rate, time_step) 
        for corr in coupling_self_correlations
    ]
    energy_cross_energy_spectral_densities = [
        compute_spectral_dens(corr, kbT, sample_rate, time_step) 
        for corr in energy_cross_energy_correlations
    ]
    energy_cross_coupling_spectral_densities = [
        compute_spectral_dens(corr, kbT, sample_rate, time_step) 
        for corr in energy_cross_coupling_correlations
    ]

    # Compute reorganization energies
    reorg_energy_gaps = [
        compute_reorg(spec_dens) 
        for spec_dens in energy_spectral_densities
    ]
    reorg_couplings = [
        compute_reorg(spec_dens) 
        for spec_dens in coupling_spectral_densities
    ]
    
    print(
        'Reorganization energies (Ha):\n' +
        '\n'.join([
            f'State {i + 1}: {reorg}' 
            for i, reorg in enumerate(reorg_energy_gaps)
        ]) +
        '\n' +
        '\n'.join([
            f'S{coupling[0]}-S{coupling[1]} coupling: {reorg_couplings[i]}'
            for i, coupling in enumerate(coupling_list)
        ])
    )

    # Write the outputs
    for istate, state in enumerate(energy_spectral_densities):
        write_dat(state, f'spectral_density_S{istate + 1}.dat')
    for icoupling, coupling in enumerate(coupling_spectral_densities):
        write_dat(
            coupling,
            f'spectral_density_S{coupling_list[icoupling][0]}'
            f'_S{coupling_list[icoupling][1]}_coupling.dat'
        )
    for idensity, density in enumerate(energy_cross_energy_spectral_densities):
        write_dat(
            density,
            f'spectral_density_S{coupling_list[idensity][0]}_cross'
            f'_S{coupling_list[idensity][1]}.dat'
        )
    for idensity, density in enumerate(
        energy_cross_coupling_spectral_densities
    ):
        write_dat(
            density,
            f'spectral_density_S{coupling_list[idensity//2][idensity%2]}'
            f'_cross_S{coupling_list[idensity//2][0]}'
            f'_S{coupling_list[idensity//2][1]}_coupling.dat'
        )

    print('Done!')
    return [
        energy_spectral_densities, 
        coupling_spectral_densities, 
        energy_cross_energy_spectral_densities, 
        energy_cross_coupling_spectral_densities
    ]

# Run `main` immediately if called as a script, but don't when loaded as a
# module
if __name__ == "__main__":
    args = sys.argv[1]
    print(args)
    result = main(args)

