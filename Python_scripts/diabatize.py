
def get_diabatic_states(energies,dipole_mom):
    diabatic_matrix=np.zeros((2,2))
    diabatic_matrix[0,0]=np.dot(dipole_mom[0,:],dipole_mom[0,:])
    diabatic_matrix[1,1]=np.dot(dipole_mom[1,:],dipole_mom[1,:])
    diabatic_matrix[0,1]=np.dot(dipole_mom[0,:],dipole_mom[1,:]) 
    diabatic_matrix[1,0]=diabatic_matrix[0,1]

    energy_matrix=np.zeros((2,2))
    energy_matrix[0,0]=energies[0,0]
    energy_matrix[1,1]=energies[1,0]

    evals,evecs=np.linalg.eig(diabatic_matrix)

    evals[0]=evals[0]*2.0/3.0*energies[0,0]/27.211399
    evals[1]=evals[1]*2.0/3.0*energies[1,0]/27.211399

    energy_states=np.dot(np.transpose(evecs),np.dot(energy_matrix,evecs))

    diabatic_state=np.zeros((2,2))
    if evals[0]<evals[1]:
        diabatic_state[0,0]=energy_states[0,0]
        diabatic_state[0,1]=evals[0]
        diabatic_state[1,0]=energy_states[1,1]
        diabatic_state[1,1]=evals[1]
    else:
        diabatic_state[0,0]=energy_states[1,1]
        diabatic_state[0,1]=evals[1]
        diabatic_state[1,0]=energy_states[0,0]
        diabatic_state[1,1]=evals[0]

    return diabatic_state,energy_states[0,1]

