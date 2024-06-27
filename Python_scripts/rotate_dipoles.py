#! /usr/bin/env python

import os.path
import numpy as np
import math
import linecache

# Quarternion eckart frame definition following J. Chem. Phys. 140, 154104 (2014)
def construct_Cmat(geom, reference_geom,mass_vec):
    # Make sure we are in the COM frame
    com_geom=np.zeros(3)
    com_ref=np.zeros(3)
    total_mass=np.sum(mass_vec)
    for i in range(mass_vec.shape[0]):
        com_geom[:]=com_geom[:]+mass_vec[i]*geom[i,:]
        com_ref[:]=com_ref[:]+mass_vec[i]*reference_geom[i,:]

    com_geom=com_geom/total_mass
    com_ref=com_ref/total_mass

    for i in range(geom.shape[0]):
        geom[i,:]=geom[i,:]-com_geom
        reference_geom[i,:]=reference_geom[i,:]-com_ref

    Cmat=np.zeros((4,4))
    for i in range(geom.shape[0]):  # loop over atomic coordinates
        xp=reference_geom[i,0]+geom[i,0]
        xm=reference_geom[i,0]-geom[i,0]
        yp=reference_geom[i,1]+geom[i,1]
        ym=reference_geom[i,1]-geom[i,1]
        zp=reference_geom[i,2]+geom[i,2]
        zm=reference_geom[i,2]-geom[i,2]

        Cmat[0,0]=Cmat[0,0]+mass_vec[i]*(xm*xm+ym*ym+zm*zm)
        Cmat[0,1]=Cmat[0,1]+mass_vec[i]*(yp*zm-ym*zp)
        Cmat[0,2]=Cmat[0,2]+mass_vec[i]*(xm*zp-xp*zm)
        Cmat[0,3]=Cmat[0,3]+mass_vec[i]*(xp*ym-xm*yp)
        Cmat[1,1]=Cmat[1,1]+mass_vec[i]*(xm*xm+yp*yp+zp*zp)
        Cmat[1,2]=Cmat[1,2]+mass_vec[i]*(xm*ym-xp*yp)
        Cmat[1,3]=Cmat[1,3]+mass_vec[i]*(xm*zm-xp*zp)
        Cmat[2,2]=Cmat[2,2]+mass_vec[i]*(xp*xp+ym*ym+zp*zp)
        Cmat[2,3]=Cmat[2,3]+mass_vec[i]*(ym*zm-yp*zp)
        Cmat[3,3]=Cmat[3,3]+mass_vec[i]*(xp*xp+yp*yp+zm*zm)

    # now make sure the matrix is symmetric
    Cmat[1,0]=Cmat[0,1]
    Cmat[2,0]=Cmat[0,2]
    Cmat[3,0]=Cmat[0,3]
    Cmat[2,1]=Cmat[1,2]
    Cmat[3,1]=Cmat[1,3]
    Cmat[3,2]=Cmat[2,3]

    return Cmat

def construct_Tmat_quarternion(geom,reference_geom,mass_vec):
    Cmat=construct_Cmat(geom, reference_geom,mass_vec)
    evals,evecs=np.linalg.eigh(Cmat)
    evec=evecs[:,0]
    Tmat=np.zeros((3,3))
    Tmat[0,0]=evec[0]*evec[0]+evec[1]*evec[1]-evec[2]*evec[2]-evec[3]*evec[3]
    Tmat[0,1]=2.0*(evec[1]*evec[2]+evec[0]*evec[3])
    Tmat[0,2]=2.0*(evec[1]*evec[3]-evec[0]*evec[2])
    Tmat[1,0]=2.0*(evec[1]*evec[2]-evec[0]*evec[3])
    Tmat[1,1]=evec[0]*evec[0]-evec[1]*evec[1]+evec[2]*evec[2]-evec[3]*evec[3]
    Tmat[1,2]=2.0*(evec[2]*evec[3]+evec[0]*evec[1])
    Tmat[2,0]=2.0*(evec[1]*evec[3]+evec[0]*evec[2])
    Tmat[2,1]=2.0*(evec[2]*evec[3]-evec[0]*evec[1])
    Tmat[2,2]=evec[0]*evec[0]-evec[1]*evec[1]-evec[2]*evec[2]+evec[3]*evec[3]

    return Tmat

def apply_eckhart_conditions_quarternion(dipole_mom,geom,ref_geom,mass_vec):
    Tmat=construct_Tmat_quarternion(geom,ref_geom,mass_vec)
    dipole_transformed=np.dot(Tmat,dipole_mom)
    return dipole_transformed

snapshot_start=1
counter=snapshot_start
snapshot_end=10001
num_atoms=10
output_data1=open('pyrazine_adiabatic_energy_dipole_S1_camb3lyp_TDA_vacuum.dat',"w")
output_data2=open('pyrazine_adiabatic_energy_dipole_S2_camb3lyp_TDA_vacuum.dat',"w")
output_data5=open('full_energy_coupling_dipole_pyrazine_camb3lyp_TDA_vacuum_diabatic_coupling.dat',"w")
output_list=open('missing_list.dat',"w")
ref_geom=get_coors_frame(num_atoms,'pyrazine.xyz')
masses=get_atomic_masses(num_atoms,'pyrazine.xyz')


while counter<snapshot_end:
    name5=str(counter)+'/pyrazine.out'
    name_molecule=str(counter)+'/pyrazine.xyz'
    print(counter)
    if os.path.exists(name5):
        excitation5=get_excitation_energy(name5,4)    
        
        if excitation5.shape[0]<4:
            output_list.write(str(counter)+'\t')
            output_line=str(0.0)+'\t'+str(0.0)+'\n'
            output_data5.write(output_line)

        else:
            dipole_mom=get_dipole_mom(name5,excitation5.shape[0])
            z_vec,x_vec=get_molecule_vec(name_molecule)
            current_geom=get_coors_frame(num_atoms,name_molecule)
            energies_2states=np.zeros((2,2))
            dipoles_2states=np.zeros((2,3))
            energies_2states[0,:]=excitation5[0,:]
            dipoles_2states[0,:]=dipole_mom[0,:]
  
            second_state=np.argmax(excitation5[:,1])
            energies_2states[1,:]=excitation5[second_state,:]
            dipoles_2states[1,:]=dipole_mom[second_state,:]

            overlap1=np.dot(z_vec,dipoles_2states[0,:])
            overlap2=np.dot(x_vec,dipoles_2states[1,:])

            if overlap1<0:
                dipoles_2states[0,:]=-1.0*dipoles_2states[0,:]
            if overlap2<0:
                dipoles_2states[1,:]=-1.0*dipoles_2states[1,:]

            diabatic,coupling=get_diabatic_states(energies_2states,dipoles_2states)
            output_line=str(diabatic[0,0])+'\t'+str(coupling)+'\t'+str(diabatic[1,0])+'\t'+str(diabatic[0,1])+'\t'+str(diabatic[1,1])+'\n'
            #output_line=str(diabatic[0,0])+'\t'+str(diabatic[0,1])+'\t'+str(diabatic[1,0])+'\t'+str(diabatic[1,1])+'\t'+str(coupling)+'\n'
            output_data5.write(output_line)

            # rotate dipoles:
            dipole1_rot=apply_eckhart_conditions_quarternion(dipoles_2states[0,:],current_geom,ref_geom,masses) 
            dipole2_rot=apply_eckhart_conditions_quarternion(dipoles_2states[1,:],current_geom,ref_geom,masses)     

	    # print dipoles
            print('Diople before rtation: ') 
            print(dipoles_2states[0,:])
            print('Dipoles after rotation')
            print(dipole1_rot)

            output_line=str(energies_2states[0,0])+'\t'+str(energies_2states[0,1])+'\t'+str(dipole1_rot[0])+'\t'+str(dipole1_rot[1])+'\t'+str(dipole1_rot[2])+'\n'
            output_data1.write(output_line)
       
            output_line=str(energies_2states[1,0])+'\t'+str(energies_2states[1,1])+'\t'+str(dipole2_rot[0])+'\t'+str(dipole2_rot[1])+'\t'+str(dipole2_rot[2])+'\n'
            output_data2.write(output_line)


    else:
        output_line=str(0.0)+'\t'+str(0.0)+'\n'
        output_data5.write(output_line)
        output_list.write(str(counter)+'\t')

    counter=counter+1

