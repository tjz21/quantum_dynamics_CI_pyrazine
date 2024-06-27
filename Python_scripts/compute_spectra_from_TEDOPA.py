#! /usr/bin/env python

#      compute_spectra_from_TEDOPA
#
# Computes a linear absorption spectrum from .jld output produced by an MPSDynamics.jl run. The
# resulting spectrum is saved in the linear_spectrum.dat file. Also produced are the populations.txt
# and the dipole_response.txt files containing the raw dipole-dipole linear response function and the 
# S1/S2 population dynamics. 
#
# 4 required command-line arguments:
#   $1: filename of the .jdl file to be read in. 
#   $2: Vertical excitation energy of the lowest excited state in the LVC Hamiltonian (in Ha)
#   $3: Spectral width over which the spectrum will be computed (in Ha)
#   $4: num_ponts: number of data points over which the spectrum is computed. 
#
#
# Last updated 20240614 by Tim J Zuehlsdorff
# Based on previous versions by Tim J Zuehlsdorff and Kye E Hunter

import h5py
import numpy as np
import cmath
import math
from scipy import integrate
import sys

# User-defined input
# first input: hdf5 filename to be processed
# second input: Reference excitation energy for the lowest excited state in the Condon region. 
# The second input defines where the linear optical spectrum will be centered 
# Third input: Third input defines the width over which the spectrum is computed (in Ha) 
filename = str(sys.argv[1])
total_E = float(sys.argv[2])
spectral_width=float(sys.argv[3])
num_points=int(sys.argv[4])  # number of points over which the spectrum is computed

# Integrant for the computation of the full optical spectrum from the linear response function.
def full_spectrum_integrant(response_func,E_val):
    integrant=np.zeros(response_func.shape[0])
    counter=0
    while counter<integrant.shape[0]:
        integrant[counter]=(response_func[counter,1]*cmath.exp(1j*response_func[counter,0]*E_val)).real
        counter=counter+1
    return integrant

def full_spectrum(response_func,steps_spectrum,start_val,end_val):
    spectrum=np.zeros((steps_spectrum,2))
    counter=0

    step_length=((end_val-start_val)/steps_spectrum)
    while counter<spectrum.shape[0]:
        E_val=start_val+counter*step_length
        prefac=1.0   # prefactor to compare directly to experiment. Set to 1 
        integrant=full_spectrum_integrant(response_func,E_val)
        spectrum[counter,0]=E_val
        spectrum[counter,1]=prefac*(integrate.simps(integrant,dx=response_func[1,0].real-response_func[0,0].real))
        counter=counter+1

    np.savetxt('linear_spectrum.dat',spectrum)


with h5py.File(filename, "r") as f:
    conv_key = list(f.keys())[2]

    # obtain real and imaginary dipole correlation functions
    group=f[conv_key]
    dipole_real = group["dcf-re"][()]
    dipole_im = group["dcf-im"][()]
    time=group["times"][()]

    # construct response function.

    response_func=np.zeros((time.shape[0],5))
   
    for i in range(response_func.shape[0]):
        response_func[i,0]=time[i]
        response_func[i,1]=dipole_real[i]
        response_func[i,2]=dipole_im[i]
        eff_cmplx=cmath.polar(dipole_real[i]+1j*dipole_im[i])
        response_func[i,3]=eff_cmplx[0]
        response_func[i,4]=eff_cmplx[1]


    # also save S1 and S2 population dynamics:
    s1_pop=group["s1"][()]
    s2_pop=group["s2"][()]
    print(s1_pop)

    s1s2_func=np.zeros((time.shape[0],3))
    for i in range(s1s2_func.shape[0]):
        s1s2_func[i,0]=time[i]
        s1s2_func[i,1]=s1_pop[i]
        s1s2_func[i,2]=s2_pop[i]
    # print S1 and S2 populations 
    np.savetxt('populations.txt',s1s2_func)

    # now make sure the phase is a continuous function
    counter = 0
    phase_fac = 0.0
    while counter < response_func.shape[0] - 1:
        response_func[counter, 4] = response_func[counter, 4] + phase_fac
        if (
            abs(response_func[counter, 4] - phase_fac - response_func[counter + 1, 4]) > 0.7 * math.pi
        ):  # check for discontinuous jump.
            diff = response_func[counter + 1, 4] - (response_func[counter, 4] - phase_fac)
            frac = diff / math.pi
            n = int(round(frac))
            phase_fac = phase_fac - math.pi * n

        counter = counter + 1

    # now apply energy shift to phase:
    for i in range(response_func.shape[0]):
        response_func[i,4]=response_func[i,4]-total_E*response_func[i,0]   # mean energy shift.

    # print total dipole response func
    np.savetxt('dipole_response.txt',response_func)

    eff_response=np.zeros((response_func.shape[0],2),dtype=np.complex_)
    for i in range(eff_response.shape[0]):
        eff_response[i,0]=response_func[i,0]
        eff_response[i,1]=response_func[i,3]*cmath.exp(1j*response_func[i,4])

    # spectrum start and end point in Ha atomic units --> centered around total_E
    spectrum_start=total_E-spectral_width/2.0
    spectrum_end=total_E+spectral_width/2.0

    # compute full spectrum
    full_spectrum(eff_response,num_points,spectrum_start,spectrum_end)
