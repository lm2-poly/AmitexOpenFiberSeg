from numpy import *
import copy


#Generate a tensor, any order, any size
#Value is the default value, commonly 0
def initTensor(value, *lengths):
	list = []
	dim = len(lengths)
	if dim == 1:
		for i in range(lengths[0]):
			list.append(value)
	elif dim > 1:
		for i in range(lengths[0]):
			list.append(initTensor(value, *lengths[1:]))
	return list

#Tenseur alternateur
def generate_epsilon_ijk():
	epsilon_ijk = initTensor( 0., 3, 3, 3)
	
	for i in range(0, len(epsilon_ijk) ):
		for j in range(0, len(epsilon_ijk[0]) ):
			for k in range(0, len(epsilon_ijk[0][0]) ):
				
				if (i == j) or (j == k) or (i == k):
					epsilon_ijk[i][j][k] = 0.
				if ("{}{}{}".format(i,j,k) == "012") or ("{}{}{}".format(i,j,k) == "120") or ("{}{}{}".format(i,j,k) == "201"):
					epsilon_ijk[i][j][k] = 1.
				if ("{}{}{}".format(i,j,k) == "021") or ("{}{}{}".format(i,j,k) == "102") or ("{}{}{}".format(i,j,k) == "210"):
					epsilon_ijk[i][j][k] = -1.
			
	return epsilon_ijk

	
def kronecker( i, j ):
	if ( i == j ):
		return 1.
	else:
		return 0.
		
def indentity12_12():
	I = initTensor(0., 12, 12)
	
	for i in range(0, 12):
		I[i][i] = 1.
		
	return I

def generate_I_tensor4():
	I_tensor4 = initTensor(0., 3, 3, 3, 3)
	for i in range( len( I_tensor4[0][0][0] ) ):
		for j in range( len( I_tensor4[0][0][0] ) ):
			for k in range( len( I_tensor4[0][0][0] ) ):
				for l in range( len( I_tensor4[0][0][0] ) ):
					I_tensor4[i][j][k][l]=(1./2.)*( kronecker(i,k)*kronecker(j,l)+kronecker(i,l)*kronecker(j,k) )
	return I_tensor4
	
def generate_J_tensor4():
	J_tensor4 = initTensor(0., 3, 3, 3, 3)
	for i in range( len( J_tensor4[0][0][0] ) ):
		for j in range( len( J_tensor4[0][0][0] ) ):
			for k in range( len( J_tensor4[0][0][0] ) ):
				for l in range( len( J_tensor4[0][0][0] ) ):
					J_tensor4[i][j][k][l]=(1./3.)*kronecker(i,j)*kronecker(k,l)
	return J_tensor4
	
def generate_K_tensor4():
	I_tensor4 = generate_I_tensor4()
	J_tensor4 = generate_J_tensor4()
	K_tensor4 = initTensor(0., 3, 3, 3, 3)
	for i in range( len( K_tensor4[0][0][0] ) ):
		for j in range( len( K_tensor4[0][0][0] ) ):
			for k in range( len( K_tensor4[0][0][0] ) ):
				for l in range( len( K_tensor4[0][0][0] ) ):
					K_tensor4[i][j][k][l]= ( I_tensor4[i][j][k][l]-J_tensor4[i][j][k][l] )		
	return K_tensor4

#Isotropic transverse
def generate_iT_matrix( axis, printBool=False ):
	if printBool:
		print("=====================================================================")
		print("Determining iT:")
		print("AXIS for transverse isotropy is", axis)
		
	identity_matrix = initTensor(0., 3, 3)
	for i in range(0, len(identity_matrix)):
		for j in range(0, len(identity_matrix)):
			identity_matrix[i][j] = kronecker(i, j)
	
	n = initTensor(0., 3)
	for i in range(0, len(n)):
		if (i == axis):
			n[i] = 1.
	
	nXn = outer(n, n)
	
	iT = initTensor(0., 3, 3)
	if printBool:print("Thus, iT=")
	for i in range(0, len(identity_matrix)):
		for j in range(0, len(identity_matrix)):
			iT[i][j] = identity_matrix[i][j] - nXn[i][j]
	if printBool:print(iT)
	return iT
	
#Isotropic transverse
def generate_EL_tensor( axis ,printBool=False):
	if printBool:
		print("=====================================================================")
		print("Determining EL:")
		print("AXIS for transverse isotropy is", axis)
	n = initTensor(0., 3)
	for i in range(0, len(n)):
		if (i == axis):
			n[i] = 1.
	
	#EL = n X n X n X n
	EL = initTensor(0., 3, 3, 3, 3)
	for i in range( len( EL[0][0][0] ) ):
		for j in range( len( EL[0][0][0] ) ):
			for k in range( len( EL[0][0][0] ) ):
				for l in range( len( EL[0][0][0] ) ):
					EL[i][j][k][l]=n[i]*n[j]*n[k]*n[l]
					
	
	EL_voigt = tensor4_to_voigt4( EL )
	if printBool: 
		print("Thus, EL in voigt notations:")
		for i in range(0, len(EL_voigt)):
			print( EL_voigt[i])

	return EL
		
def generate_JT_tensor( iT, printBool=False ):
	if printBool:
		print( "=====================================================================")
		print( "Determining JT:")

	#EL = n X n X n X n
	JT = initTensor(0., 3, 3, 3, 3)
	for i in range( len( JT[0][0][0] ) ):
		for j in range( len( JT[0][0][0] ) ):
			for k in range( len( JT[0][0][0] ) ):
				for l in range( len( JT[0][0][0] ) ):
					JT[i][j][k][l]=(1./2.)*iT[i][j]*iT[k][l]
		
	if printBool:print( "Thus, JT in voigt notations:")
	JT_voigt = tensor4_to_voigt4( JT )
	
	if printBool:
		for i in range(0, len(JT_voigt)):
			print( JT_voigt[i])
	
	return JT
		
def generate_IT_matrix( axis, printBool=False ):
	if printBool:
		print( "=====================================================================")
		print( "Determining IT:")
		print( "AXIS for transverse isotropy is", axis)
	
	IT = initTensor(0., 6, 6)
	for i in range(0, 3):
		for j in range(0, 3):
			if ( i == j) and (axis != i):
				IT[i][j] = 1.
	if (axis == 0):
		IT[3][3] = 1.
	if (axis == 1):
		IT[4][4] = 1.
	if (axis == 2):
		IT[5][5] = 1.
		
	if printBool:
		print( "Thus, IT, which is a matri(6X6):")
		for i in range(0, len(IT)):
			print( IT[i] )
		
	return IT
		
def generate_KE_tensor( axis, iT_matrix ,printBool=False ):
	if printBool:
		print( "=====================================================================")
		print( "Determining KE:")
	
	n = initTensor(0, 3)
	for i in range(0, len(n)):
		if (i == axis):
			n[i] = 1.

	KE = initTensor(0., 3, 3, 3, 3)
	for i in range( len( KE[0][0][0] ) ):
		for j in range( len( KE[0][0][0] ) ):
			for k in range( len( KE[0][0][0] ) ):
				for l in range( len( KE[0][0][0] ) ):
					KE[i][j][k][l] = (1./6.)*(2.*n[i]*n[j] - iT_matrix[i][j])*( 2.*n[k]*n[l]-iT_matrix[k][l] )
					

	KE_voigt = tensor4_to_voigt4( KE )
	if printBool:
		print( "Thus, KE in voigt notations:" )
		for i in range(0, len(KE_voigt)):
			print( KE_voigt[i] )
		
	return KE
	
def generate_KT_tensor( IT_matrix, JT_tensor ,printBool=False ):
	if printBool:
		print( "=====================================================================" )
		print( "Determining KT:" )

	IT_tensor = voigt4_to_tensor4( IT_matrix )
	
	KT_tensor = initTensor(0., 3, 3, 3, 3)
	for i in range( len( KT_tensor[0][0][0] ) ):
		for j in range( len( KT_tensor[0][0][0] ) ):
			for k in range( len( KT_tensor[0][0][0] ) ):
				for l in range( len( KT_tensor[0][0][0] ) ):
					KT_tensor[i][j][k][l] = IT_tensor[i][j][k][l] - JT_tensor[i][j][k][l]
					
	if printBool:print( "Thus, KT in voigt notations:")
	KT_voigt = tensor4_to_voigt4( KT_tensor )
	if printBool:
		for i in range(0, len(KT_voigt)):
			print( KT_voigt[i])
		
	return KT_tensor
	
def generate_KL_tensor( KT, KE, printBool=False ):
	if printBool:
		print( "=====================================================================")
		print( "Determining KL:")
	K = generate_K_tensor4()
	
	KL_tensor = initTensor(0., 3, 3, 3, 3)
	for i in range( len( KL_tensor[0][0][0] ) ):
		for j in range( len( KL_tensor[0][0][0] ) ):
			for k in range( len( KL_tensor[0][0][0] ) ):
				for l in range( len( KL_tensor[0][0][0] ) ):
					KL_tensor[i][j][k][l] = K[i][j][k][l] - KT[i][j][k][l] - KE[i][j][k][l]
					
	if printBool:print( "Thus, KL in voigt notations:")
	KL_voigt = tensor4_to_voigt4( KL_tensor )
	if printBool:
		for i in range(0, len(KL_voigt)):
			print( KL_voigt[i])
		
	return KL_tensor
	
def generate_F_tensors( axis, iT_matrix, printBool=False):
	if printBool:
		print( "=====================================================================")
		print( "Determining F:")
	
	n = initTensor(0, 3)
	for i in range(0, len(n)):
		if (i == axis):
			n[i] = 1.
			
	F_tensor = initTensor(0, 3, 3, 3, 3)
	for i in range( len( F_tensor[0][0][0] ) ):
		for j in range( len( F_tensor[0][0][0] ) ):
			for k in range( len( F_tensor[0][0][0] ) ):
				for l in range( len( F_tensor[0][0][0] ) ):
					F_tensor[i][j][k][l]=sqrt(2)/2.*(iT_matrix[i][j]*n[k]*n[l]);
					
				
	if printBool:print( "Thus, F in voigt notations:")
	F_voigt = tensor4_to_voigt4_Facu( F_tensor )
	FT_voigt=copy.deepcopy(F_voigt)
	for i in range(6):
		for j in range(6):
			FT_voigt[i][j]=F_voigt[j][i]
		if printBool:print(F_voigt[i])

	if printBool:
		print( "Thus, F_T in voigt notations:")
		for i in range(6):
			print(FT_voigt[i])	

	FT_tensor=voigt4_to_tensor4_Facu(FT_voigt)		

	return F_tensor,FT_tensor


#=============================================================
# Symmetries
#=============================================================
#	check_tensor_minor_symmetry( tensor )
#	check_tensor_major_symmetry( tensor )
#	apply_minor_sym_to_tensor_term( A_tensor4, i, j, k, l )
#	apply_major_sym_to_tensor_term( A_tensor4, i, j, k, l )

#=============================================================
# Voigt
#=============================================================
#	tensor4_to_voigt4( A_tensor4 )
#	voigt4_to_tensor4( A_voigt4 )

#=============================================================
# Base change
#=============================================================
#	tensorial_base_change( P, tensorA )

#=============================================================
# Matrix
#=============================================================
#	matrix_dot_matrix( matrixa, matrixb )

def check_tensor_minor_symmetry( tensor ):
	for i in range(len(tensor[0][0][0])):
		for j in range(len(tensor[0][0][0])):
			for k in range(len(tensor[0][0][0])):
				for l in range(len(tensor[0][0][0])):
					
					if (tensor[i][j][k][l] != tensor[j][i][k][l]):
						print( "[check_tensor_minor_symmetry] Tensor is not symmetrical ")
						print( i,j,k,l, tensor[i][j][k][l], tensor[j][i][k][l])
						return False
						
					if (tensor[i][j][k][l] != tensor[i][j][l][k]):
						print( "[check_tensor_minor_symmetry] Tensor is not symmetrical ")
						print( i,j,k,l, tensor[i][j][k][l], tensor[i][j][l][k])
						return False
						
					if (tensor[i][j][k][l] != tensor[j][i][l][k]):
						print( "[check_tensor_minor_symmetry] Tensor is not symmetrical ")
						print( i,j,k,l, tensor[i][j][k][l], tensor[j][i][l][k])
						return False
	return True


def check_tensor_major_symmetry( tensor ):
	for i in range(len(tensor[0][0][0])):
		for j in range(len(tensor[0][0][0])):
			for k in range(len(tensor[0][0][0])):
				for l in range(len(tensor[0][0][0])):
					
					if (tensor[i][j][k][l] != tensor[k][l][i][j]):
						print( "[check_tensor_major_symmetry] Tensor is not symmetrical ")
						print( i,j,k,l, tensor[i][j][k][l], tensor[k][l][i][j])
						return False
	return True


def tensor4_to_voigt4( A_tensor4 ):
	A_voigt4 = initTensor(0., 6, 6)
	
	#blue
	for i in range( 0, 3 ):
		for j in range( 0, 3 ):
			A_voigt4[i][j] = A_tensor4[i][i][j][j]
		

	for i in range( 0, 3 ):
		#print( i
		A_voigt4[3][i] = sqrt(2) * A_tensor4[1][2][i][i]
		A_voigt4[4][i] = sqrt(2) * A_tensor4[2][0][i][i]
		A_voigt4[5][i] = sqrt(2) * A_tensor4[0][1][i][i]
		
	for j in range( 0, 3 ):
		#print( j
		A_voigt4[j][3] = sqrt(2) * A_tensor4[j][j][1][2]
		A_voigt4[j][4] = sqrt(2) * A_tensor4[j][j][2][0]
		A_voigt4[j][5] = sqrt(2) * A_tensor4[j][j][0][1]
		
	A_voigt4[3][3] = 2 * A_tensor4[1][2][1][2]
	A_voigt4[4][3] = 2 * A_tensor4[2][0][1][2]
	A_voigt4[5][3] = 2 * A_tensor4[0][1][1][2]
	
	A_voigt4[3][4] = 2 * A_tensor4[1][2][2][0]
	A_voigt4[4][4] = 2 * A_tensor4[2][0][2][0]
	A_voigt4[5][4] = 2 * A_tensor4[0][1][2][0]

	A_voigt4[3][5] = 2 * A_tensor4[1][2][0][1]
	A_voigt4[4][5] = 2 * A_tensor4[2][0][0][1]
	A_voigt4[5][5] = 2 * A_tensor4[0][1][0][1]
	
	#print( "AVOIGT4[0]:"
	#for i in range(len(A_voigt4[0])):
		#print( A_voigt4[i]

	return A_voigt4

def tensor4_to_voigt4_Facu( inputT4 ):
	outputVoigtT2 = initTensor(0., 6, 6)

	outputVoigtT2[0][0]=inputT4[0][0][0][0]
	outputVoigtT2[1][0]=inputT4[1][1][0][0]
	outputVoigtT2[2][0]=inputT4[2][2][0][0]
	outputVoigtT2[3][0]=inputT4[1][2][0][0]*sqrt(2)
	outputVoigtT2[4][0]=inputT4[2][0][0][0]*sqrt(2)
	outputVoigtT2[5][0]=inputT4[0][1][0][0]*sqrt(2)
	outputVoigtT2[0][1]=inputT4[0][0][1][1]
	outputVoigtT2[1][1]=inputT4[1][1][1][1]
	outputVoigtT2[2][1]=inputT4[2][2][1][1]
	outputVoigtT2[3][1]=inputT4[1][2][1][1]*sqrt(2)
	outputVoigtT2[4][1]=inputT4[2][0][1][1]*sqrt(2)
	outputVoigtT2[5][1]=inputT4[0][1][1][1]*sqrt(2)
	outputVoigtT2[0][2]=inputT4[0][0][2][2]
	outputVoigtT2[1][2]=inputT4[1][1][2][2]
	outputVoigtT2[2][2]=inputT4[2][2][2][2]
	outputVoigtT2[3][2]=inputT4[1][2][2][2]*sqrt(2)
	outputVoigtT2[4][2]=inputT4[2][0][2][2]*sqrt(2)
	outputVoigtT2[5][2]=inputT4[0][1][2][2]*sqrt(2)
	outputVoigtT2[0][3]=inputT4[0][0][1][2]*sqrt(2)
	outputVoigtT2[1][3]=inputT4[1][1][1][2]*sqrt(2)
	outputVoigtT2[2][3]=inputT4[2][2][1][2]*sqrt(2)
	outputVoigtT2[3][3]=inputT4[1][2][1][2]
	outputVoigtT2[4][3]=inputT4[2][0][1][2]
	outputVoigtT2[5][3]=inputT4[0][1][1][2]

	outputVoigtT2[0][4]=inputT4[0][0][2][0]*sqrt(2)
	outputVoigtT2[1][4]=inputT4[1][1][2][0]*sqrt(2)
	outputVoigtT2[2][4]=inputT4[2][2][2][0]*sqrt(2)
	outputVoigtT2[3][4]=inputT4[1][2][2][0]
	outputVoigtT2[4][4]=inputT4[2][0][2][0]
	outputVoigtT2[5][4]=inputT4[0][1][2][0]
	outputVoigtT2[0][5]=inputT4[0][0][0][1]*sqrt(2)
	outputVoigtT2[1][5]=inputT4[1][1][0][1]*sqrt(2)
	outputVoigtT2[2][5]=inputT4[2][2][0][1]*sqrt(2)
	outputVoigtT2[3][5]=inputT4[1][2][0][1]
	outputVoigtT2[4][5]=inputT4[2][0][0][1]
	outputVoigtT2[5][5]=inputT4[0][1][0][1]

	return outputVoigtT2

#Takes A_voigt(6,6) gives back A_tensor(3,3,3,3), with symmetries
def voigt4_to_tensor4( A_voigt4 ):

	A_tensor4 = initTensor(0., 3, 3, 3, 3)
	
	a = 0
	b = 0
	A_voigt4_length = len( A_voigt4 )
	for a in range(0, A_voigt4_length ):
		for b in range(0, A_voigt4_length ):

			flaga = False
			flagb = False
			
			if (a == 0):
				i = 0
				j = 0
				
			if (b == 0):
				k=0
				l=0
				
			if (a == 1):
				i=1
				j=1
				
			if (b == 1):
				k=1
				l=1
				
			if (a == 2):
				i=2
				j=2

			if (b == 2):
				k=2
				l=2
				
				
			if (a == 3):
				i=0
				j=1
				ip=1
				jp=2
				flaga= True

			if (b ==3):
				k=0
				l=1
				kp=1
				lp=2
				flagb= True	

			if (a == 4):
				i=1
				j=2
				ip=2
				jp=0
				flaga= True
	
			if (b == 4):
				k=1
				l=2
				kp=2
				lp=0
				flagb= True
	
			if (a == 5):
				i=0
				j=2
				ip=0
				jp=1
				flaga= True

			if (b == 5):
				k=0
				l=2
				kp=0
				lp=1
				flagb = True
			
			
			if (flaga is True) and (flagb is True):
				A_tensor4[ip][jp][kp][lp] = A_voigt4[a][b]/2.
				A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, ip, jp, kp, lp )
				A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, ip, jp, kp, lp )
			if (flaga is True) and (flagb is False):
				A_tensor4[ip][jp][k][l] = A_voigt4[a][b]/(sqrt(2.))
				A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, ip, jp, k, l )
				A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, ip, jp, k, l )
			if(flagb is True) and (flaga is False):
				A_tensor4[i][j][kp][lp] = A_voigt4[a][b]/(sqrt(2.))
				A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, i, j, kp, lp )
				A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, i, j, kp, lp )
				
			if (flaga is False) and (flagb is False):
			
				A_tensor4[i][j][k][l] = A_voigt4[a][b]
				A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, i, j, k, l )
				A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, i, j, k, l )
			
			#print( "A_voigt4[",a ,"][", b, "] = " , "A_tensor4[", i, "][", j, "][", k, "][", l, "]" 
			#print( A_tensor4[i][j][k][l], A_voigt4[a][b]
			#print( "-------------------------------------------------------------------"

	return A_tensor4

def voigt4_to_tensor4_Facu(inputT2):
	inputT2=array(inputT2)
	outputT4=initTensor(0., 3, 3, 3, 3)

	outputT4[0][0][0][0]=inputT2[0,0]
	outputT4[1][1][0][0]=inputT2[1,0]
	outputT4[2][2][0][0]=inputT2[2,0]
	outputT4[1][2][0][0]=inputT2[3,0]/sqrt(2)
	outputT4[2][0][0][0]=inputT2[4,0]/sqrt(2)
	outputT4[0][1][0][0]=inputT2[5,0]/sqrt(2)

	outputT4[0][0][1][1]=inputT2[0,1]
	outputT4[1][1][1][1]=inputT2[1,1]
	outputT4[2][2][1][1]=inputT2[2,1]
	outputT4[1][2][1][1]=inputT2[3,1]/sqrt(2)
	outputT4[2][0][1][1]=inputT2[4,1]/sqrt(2)
	outputT4[0][1][1][1]=inputT2[5,1]/sqrt(2)
	
	outputT4[0][0][2][2]=inputT2[0,2]
	outputT4[1][1][2][2]=inputT2[1,2]
	outputT4[2][2][2][2]=inputT2[2,2]
	outputT4[1][2][2][2]=inputT2[3,2]/sqrt(2)
	outputT4[2][0][2][2]=inputT2[4,2]/sqrt(2)
	outputT4[0][1][2][2]=inputT2[5,2]/sqrt(2)
                         
	outputT4[0][0][1][2]=inputT2[0,3]/sqrt(2)
	outputT4[1][1][1][2]=inputT2[1,3]/sqrt(2)
	outputT4[2][2][1][2]=inputT2[2,3]/sqrt(2)
	outputT4[1][2][1][2]=inputT2[3,3]/2
	outputT4[2][0][1][2]=inputT2[4,3]/2
	outputT4[0][1][1][2]=inputT2[5,3]/2

	outputT4[0][0][2][0]=inputT2[0,4]/sqrt(2)
	outputT4[1][1][2][0]=inputT2[1,4]/sqrt(2)
	outputT4[2][2][2][0]=inputT2[2,4]/sqrt(2)
	outputT4[1][2][2][0]=inputT2[3,4]/2
	outputT4[2][0][2][0]=inputT2[4,4]/2
	outputT4[0][1][2][0]=inputT2[5,4]/2
                         
	outputT4[0][0][0][1]=inputT2[0,5]/sqrt(2)
	outputT4[1][1][0][1]=inputT2[1,5]/sqrt(2)
	outputT4[2][2][0][1]=inputT2[2,5]/sqrt(2)
	outputT4[1][2][0][1]=inputT2[3,5]/2
	outputT4[2][0][0][1]=inputT2[4,5]/2
	outputT4[0][1][0][1]=inputT2[5,5]/2

	# % Symetries
    #outputT4(0,0,0,0)=inputT2(0,0)
    #outputT4(1,1,0,0)=inputT2(1,0)
    #outputT4(2,2,0,0)=inputT2(2,0)
	outputT4[2][1][0][0]=outputT4[1][2][0][0]#=inputT2(3,0)
	outputT4[0][2][0][0]=outputT4[2][0][0][0]#=inputT2(4,0)
	outputT4[1][0][0][0]=outputT4[0][1][0][0]#=inputT2(5,0)

    #outputT4(0,0,1,1)=inputT2(0,1)
    #outputT4(1,1,1,1)=inputT2(1,1)
    #outputT4(2,2,1,1)=inputT2(2,1)
	outputT4[2][1][1][1]=outputT4[1][2][1][1]#=inputT2(3,1)
	outputT4[0][2][1][1]=outputT4[2][0][1][1]#=inputT2(4,1)
	outputT4[1][0][1][1]=outputT4[0][1][1][1]#=inputT2(5,1)
	
    #outputT4(0,0,2,2)=inputT2(0,2)
    #outputT4(1,1,2,2)=inputT2(1,2)
    #outputT4(2,2,2,2)=inputT2(2,2)
	outputT4[2][1][2][2]=outputT4[1][2][2][2]#=inputT2(3,2)
	outputT4[0][2][2][2]=outputT4[2][0][2][2]#=inputT2(4,2)
	outputT4[1][0][2][2]=outputT4[0][1][2][2]#=inputT2(5,2)
	
	outputT4[0][0][2][1]=outputT4[0][0][1][2]#=inputT2(0,3)
	outputT4[1][1][2][1]=outputT4[1][1][1][2]#=inputT2(1,3)
	outputT4[2][2][2][1]=outputT4[2][2][1][2]#=inputT2(2,3)
	outputT4[2][1][1][2]=outputT4[1][2][1][2]#=inputT2(3,3)
	outputT4[1][2][2][1]=outputT4[1][2][1][2]
	outputT4[2][1][2][1]=outputT4[1][2][1][2]
	outputT4[0][2][1][2]=outputT4[2][0][1][2]#=inputT2(4,3)
	outputT4[2][0][2][1]=outputT4[2][0][1][2]
	outputT4[0][2][2][1]=outputT4[2][0][1][2]
	outputT4[1][0][1][2]=outputT4[0][1][1][2]#=inputT2(5,3)
	outputT4[0][1][2][1]=outputT4[0][1][1][2]
	outputT4[1][0][2][1]=outputT4[0][1][1][2]

	outputT4[0][0][0][2]=outputT4[0][0][2][0]#=inputT2(0,4)
	outputT4[1][1][0][2]=outputT4[1][1][2][0]#=inputT2(1,4)
	outputT4[2][2][0][2]=outputT4[2][2][2][0]#=inputT2(2,4)
	outputT4[2][1][2][0]=outputT4[1][2][2][0]#=inputT2(3,4)
	outputT4[1][2][0][2]=outputT4[1][2][2][0]
	outputT4[2][1][0][2]=outputT4[1][2][2][0]
	outputT4[0][2][2][0]=outputT4[2][0][2][0]#=inputT2(4,4)
	outputT4[2][0][0][2]=outputT4[2][0][2][0]
	outputT4[0][2][0][2]=outputT4[2][0][2][0]
	outputT4[1][0][2][0]=outputT4[0][1][2][0]#=inputT2(5,4)
	outputT4[0][1][0][2]=outputT4[0][1][2][0]
	outputT4[1][0][0][2]=outputT4[0][1][2][0]
                         
	outputT4[0][0][1][0]=outputT4[0][0][0][1]#=inputT2(0,5)
	outputT4[1][1][1][0]=outputT4[1][1][0][1]#=inputT2(1,5)
	outputT4[2][2][1][0]=outputT4[2][2][0][1]#=inputT2(2,5)
	outputT4[2][1][0][1]=outputT4[1][2][0][1]#=inputT2(3,5)
	outputT4[1][2][1][0]=outputT4[1][2][0][1]
	outputT4[2][1][1][0]=outputT4[1][2][0][1]
	outputT4[0][2][0][1]=outputT4[2][0][0][1]#=inputT2(4,5)
	outputT4[2][0][1][0]=outputT4[2][0][0][1]
	outputT4[0][2][1][0]=outputT4[2][0][0][1]
	outputT4[1][0][0][1]=outputT4[0][1][0][1]#=inputT2(5,5)
	outputT4[0][1][1][0]=outputT4[0][1][0][1]
	outputT4[1][0][1][0]=outputT4[0][1][0][1]

	return outputT4
	
#Takes A_voigt(6,6) gives back A_tensor(4,4,4,4), with symmetries
def voigt4_to_tensor4_no_symmetry( A_voigt4 ):

	A_tensor4 = initTensor(0., 3, 3, 3, 3)
	
	a = 0
	b = 0
	A_voigt4_length = len( A_voigt4 )
	for a in range(0, A_voigt4_length ):
		for b in range(0, A_voigt4_length ):

			flaga = False
			flagb = False
			
			if (a == 0):
				i = 0
				j = 0
				
			if (b == 0):
				k=0
				l=0
				
			if (a == 1):
				i=1
				j=1
				
			if (b == 1):
				k=1
				l=1
				
			if (a == 2):
				i=2
				j=2

			if (b == 2):
				k=2
				l=2
				
				
			if (a == 3):
				i=0
				j=1
				ip=1
				jp=2
				flaga= True

			if (b ==3):
				k=0
				l=1
				kp=1
				lp=2
				flagb= True	

			if (a == 4):
				i=1
				j=2
				ip=2
				jp=0
				flaga= True
	
			if (b == 4):
				k=1
				l=2
				kp=2
				lp=0
				flagb= True
	
			if (a == 5):
				i=0
				j=2
				ip=0
				jp=1
				flaga= True

			if (b == 5):
				k=0
				l=2
				kp=0
				lp=1
				flagb = True
			
			
			if (flaga is True) and (flagb is True):
				A_tensor4[ip][jp][kp][lp] = A_voigt4[a][b]/2.
				#A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, ip, jp, kp, lp )
				#A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, ip, jp, kp, lp )
			if (flaga is True) and (flagb is False):
				A_tensor4[ip][jp][k][l] = A_voigt4[a][b]/(sqrt(2.))
				#A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, ip, jp, k, l )
				#A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, ip, jp, k, l )
			if(flagb is True) and (flaga is False):
				A_tensor4[i][j][kp][lp] = A_voigt4[a][b]/(sqrt(2.))
				#A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, i, j, kp, lp )
				#A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, i, j, kp, lp )
				
			if (flaga is False) and (flagb is False):
			
				A_tensor4[i][j][k][l] = A_voigt4[a][b]
				#A_tensor4 = apply_minor_sym_to_tensor_term( A_tensor4, i, j, k, l )
				#A_tensor4 = apply_major_sym_to_tensor_term( A_tensor4, i, j, k, l )
			
			#print( "A_voigt4[",a ,"][", b, "] = " , "A_tensor4[", i, "][", j, "][", k, "][", l, "]" 
			#print( A_tensor4[i][j][k][l], A_voigt4[a][b]
			#print( "-------------------------------------------------------------------"

	return A_tensor4
	
def apply_minor_sym_to_tensor_term( A_tensor4, i, j, k, l ):
	A_tensor4[j][i][k][l] = A_tensor4[i][j][k][l]
	A_tensor4[i][j][l][k] = A_tensor4[i][j][k][l]
	A_tensor4[j][i][l][k] = A_tensor4[i][j][k][l]
	return A_tensor4
						
def apply_major_sym_to_tensor_term( A_tensor4, i, j, k, l ):
	A_tensor4[k][l][i][j] = A_tensor4[i][j][k][l]
	return A_tensor4
	
	
#Only convenient if initial and final base given, else, determine P by hand, much faster
def generate_trans_matrix( init_base, final_base ):
	P = initTensor(0, 3, 3)
	for i in range(len(init_base)):
		for j in range(len(init_base)):
			#print( "--------------------------"
			#print( init_base[i], final_base[j]
			P[i][j] = vector_dot_vector( init_base[i], final_base[j] )
			#print( i, j, P[i][j]
			
	return P

#Scalar product
def vector_dot_vector( vectora, vectorb ):
	sumdot = 0
	for i in range(len(vectora)):
		sumdot = sumdot + vectora[i] * vectorb[i]
		#print( vectora[i], " * ", vectorb[i]
	#print( vectora, " times ", vectorb, " = ", sumdot
	return sumdot



def matrix_dot_vector( matrixa, vector ):
	C = initTensor( 0, len(matrixa[0]), len(vector))
	if (len(matrixa[0]) != len(vector)):
		print( "This is not good... You're trying to perform a matrix dot vector with impossible dimensions!!")
	     
	for i in range( len(matrixa) ):
		for j in range( len(matrixa[0])):
			for k in range( len( vector )):
				C[i][j] = C[i][j] + matrixa[i][k]*vector[k]
	return C


def tensorial_base_change( P, tensorA ):
	tensorB = initTensor( 0, 3, 3, 3, 3)
	for m in range( len( tensorA[0][0][0] ) ):
		for n in range( len( tensorA[0][0][0] ) ):
			for o in range( len( tensorA[0][0][0] ) ):
				for p in range( len( tensorA[0][0][0] ) ):
					for i in range( len( tensorA[0][0][0] ) ):
						for j in range( len( tensorA[0][0][0] ) ):
							for k in range( len( tensorA[0][0][0] ) ):
								for l in range( len( tensorA[0][0][0] ) ):
									tensorB[m][n][o][p] = tensorB[m][n][o][p] + P[i][m]*P[j][n]*P[k][o]*P[l][p]*tensorA[i][j][k][l]
	return tensorB

	
def tensorial_base_change_Facu(P,inputT4):
    # prend comme argument un tenseur d'ordre 4 et une matrice de passage
    # et qui retourne le tenseur d'ordre 4 dans la nouvelle base

    outputT4=  initTensor( 0, 3, 3, 3, 3)
    tempT4=    initTensor( 0, 3, 3, 3, 3)
    tempT4_=   initTensor( 0, 3, 3, 3, 3)
    tempT4__=  initTensor( 0, 3, 3, 3, 3)

    for p in range(3):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    tempT4[p][i][j][k]= P[0,p]*inputT4[i][j][k][0]+\
                                        P[1,p]*inputT4[i][j][k][1]+\
                                        P[2,p]*inputT4[i][j][k][2]

    # print("tempT4[0,0,0,0]={: 5.4f}".format(tempT4[0][0][0][0]))
    # print("tempT4[0,0,1,0]={: 5.4f}".format(tempT4[0][0][1][0]))

    for o in range(3):
        for p in range(3):
            for i in range(3):
                for j in range(3):
                    tempT4_[o][p][i][j]=P[0,o]*tempT4[p][i][j][0]+\
                                        P[1,o]*tempT4[p][i][j][1]+\
                                        P[2,o]*tempT4[p][i][j][2]

    # print("tempT4_[0,0,0,0]={: 5.4f}".format(tempT4_[0][0][0][0]))
    # print("tempT4_[0,0,0,1]={: 5.4f}".format(tempT4_[0][0][0][1]))


    for n in range(3):
        for o in range(3):
            for p in range(3):
                for i in range(3):
                    tempT4__[n][o][p][i]=P[0,n]*tempT4_[o][p][i][0]+\
                                         P[1,n]*tempT4_[o][p][i][1]+\
                                         P[2,n]*tempT4_[o][p][i][2]

    # print("tempT4__[0,0,0,0]={: 5.4f}".format(tempT4__[0][0][0][0]))
    # print("tempT4__[2,2,2,2]={: 5.4f}".format(tempT4__[2][2][2][2]))

    for m in range(3):
        for n in range(3):
            for o in range(3):
                for p in range(3):
                    outputT4[m][n][o][p]=P[0,m]*tempT4__[n][o][p][0]+\
                                         P[1,m]*tempT4__[n][o][p][1]+\
                                         P[2,m]*tempT4__[n][o][p][2]

    # print("outputT4[0,0,0,0]={: 5.4f}".format(outputT4[0][0][0][0]))
    # print("outputT4[2,2,2,2]={: 5.4f}".format(outputT4[2][2][2][2]))

    return outputT4

def voigt_to_matrix( A_tensor2_voigtshape ):
	A_tensor2_matrix = initTensor(0, 3, 3)
	
	A_tensor2_matrix[0][0] = A_tensor2_voigtshape[0]
	A_tensor2_matrix[1][1] = A_tensor2_voigtshape[1]
	A_tensor2_matrix[2][2] = A_tensor2_voigtshape[2]
		
	A_tensor2_matrix[1][2] = A_tensor2_voigtshape[3]/sqrt(2)
	A_tensor2_matrix[2][1] = A_tensor2_voigtshape[3]/sqrt(2)
		
	A_tensor2_matrix[2][0] = A_tensor2_voigtshape[4]/sqrt(2)
	A_tensor2_matrix[0][2] = A_tensor2_voigtshape[4]/sqrt(2)
		
	A_tensor2_matrix[0][1] = A_tensor2_voigtshape[5]/sqrt(2)
	A_tensor2_matrix[1][0] = A_tensor2_voigtshape[5]/sqrt(2)
	
	return A_tensor2_matrix
	

#Square matrix, any size
def transpose_matrix( A_matrix ):
	A_transposed = initTensor( 0., len(A_matrix[0]), len(A_matrix[0]) )
	for i in range( 0, len(A_matrix[0])):
		for j in range( 0, len(A_matrix[0])):
			A_transposed[j][i] = A_matrix[i][j]
	return A_transposed


def tensor4_contract4_tensor4( A, B ):
	temp_sum = 0
	
	for i in range(0, len(A[0][0][0])):
		for j in range(0, len(A[0][0][0])):
			for k in range(0, len(A[0][0][0])):
				for l in range(0, len(A[0][0][0])):
					temp_sum = temp_sum + A[i][j][k][l]*B[i][j][k][l]
	return temp_sum


#Put an isotropic tensor in, get alpha and beta to build S_invert
def extract_isotropic_parameters( C_iso,printBool=False ):
	#input in voigt notation (6,6)-> to (3,3,3,3)
	C_tensor4 = voigt4_to_tensor4( C_iso )
	
	J_tensor4 = generate_J_tensor4()
	
	#I_tensor4 = generate_I_tensor4()
	
	K_tensor4 = generate_K_tensor4()
	
	alpha = tensor4_contract4_tensor4( J_tensor4 , C_tensor4 )
	beta = tensor4_contract4_tensor4( K_tensor4, C_tensor4 )/5.

	nu = (alpha-beta)/(beta+2.0*alpha)
	E  = alpha*(1-2*nu)

	if printBool:

		print( "Alpha =", alpha)

		print( "beta =", beta)
		
		print( "")
		print( "To get S_tensor4 invert:")
		print( "S_invert = 1/alpha * J_tensor4 + 1/beta * K_tensor4")
		print( "1/alpha =", 1./alpha)
		print( "1/beta =", 1./beta)
	
	return alpha, beta, E, nu


#Put an transverse isotropic compliance tensor S (or C) in, get parameters to rebuild it	
def extract_trans_iso_parameters_from_S( S_trans_iso, axis):
	#input in voigt notation (6,6)-> to (3,3,3,3)
	S_tensor4 = voigt4_to_tensor4( S_trans_iso )

	#auxiliairy matrices
	iT_matrix=generate_iT_matrix( axis )
	IT_matrix=generate_IT_matrix( axis )

	EL_tensor=			generate_EL_tensor( axis )
	JT_tensor=			generate_JT_tensor( generate_iT_matrix( axis ) )
	KE_tensor=			generate_KE_tensor( axis, generate_iT_matrix( axis )  )
	KT_tensor=			generate_KT_tensor( generate_IT_matrix( axis ), JT_tensor  )
	KL_tensor=			generate_KL_tensor( KT_tensor, KE_tensor )
	F_tensor,FT_tensor= generate_F_tensors( axis, iT_matrix)


	alpha= 			tensor4_contract4_tensor4( EL_tensor, S_tensor4 )
	beta = 			tensor4_contract4_tensor4( JT_tensor, S_tensor4 )
	gamma= 			tensor4_contract4_tensor4( FT_tensor, S_tensor4 )
	gamma_prime= 	tensor4_contract4_tensor4( F_tensor,  S_tensor4 )
	delta=		0.5*tensor4_contract4_tensor4( KT_tensor, S_tensor4 )
	delta_prime=0.5*tensor4_contract4_tensor4( KL_tensor, S_tensor4 )

	return alpha, beta, gamma, gamma_prime, delta, delta_prime

#Put an isotropic tensor in, get alpha and beta to build S_invert
def extract_cubic_parameters( S_matrix ):
	
	S_tensor4 = voigt4_to_tensor4( S_matrix )
	
	e1 = [ 1., 0., 0. ]
	e2 = [ 0., 1., 0. ]
	e3 = [ 0., 0., 1. ]
	
	Z_tensor4 = initTensor( 0., 3, 3, 3, 3 )
	
	for i in range(0, len(Z_tensor4[0][0][0])):
		for j in range(0, len(Z_tensor4[0][0][0])):
			for k in range(0, len(Z_tensor4[0][0][0])):
				for l in range(0, len(Z_tensor4[0][0][0])):
					Z_tensor4[i][j][k][l] = e1[i]*e1[j]*e1[k]*e1[l]+e2[i]*e2[j]*e2[k]*e2[l]+e3[i]*e3[j]*e3[k]*e3[l]

	J_tensor4 = generate_J_tensor4()
	
	I_tensor4 = generate_I_tensor4()
	
	K_tensor4 = generate_K_tensor4()
	
	KA_tensor4 = initTensor( 0., 3, 3, 3, 3 )
	for i in range( len( KA_tensor4[0][0][0] ) ):
		for j in range( len( KA_tensor4[0][0][0] ) ):
			for k in range( len( KA_tensor4[0][0][0] ) ):
				for l in range( len( KA_tensor4[0][0][0] ) ):
					KA_tensor4[i][j][k][l]= (Z_tensor4[i][j][k][l]-J_tensor4[i][j][k][l])
					
	KB_tensor4 = initTensor( 0., 3, 3, 3, 3 )
	for i in range( len( KB_tensor4[0][0][0] ) ):
		for j in range( len( KB_tensor4[0][0][0] ) ):
			for k in range( len( KB_tensor4[0][0][0] ) ):
				for l in range( len( KB_tensor4[0][0][0] ) ):
					KB_tensor4[i][j][k][l]= (I_tensor4[i][j][k][l]-Z_tensor4[i][j][k][l])
					
					
	alpha = tensor4_contract4_tensor4( J_tensor4, S_tensor4 )
	print( "Alpha =", alpha)
	beta = tensor4_contract4_tensor4( KA_tensor4, S_tensor4 )
	print( "beta =", beta/2.)
	gamma = tensor4_contract4_tensor4( KB_tensor4, S_tensor4 )
	print( "gamma =", gamma/3.)
	
	print( "")
	print( "S_inv = 1./alpha * J + 1./beta * Ka + 1./gamma * Kb ")
	print( "1/alpha = ", 1./alpha)
	print( "1/beta = ", 1./beta)
	print( "1/gamma = ", 1./gamma)
	
	return alpha, beta, gamma
	
def generate_symmetric_matrix66_from_list( C ):

	C_matrix = [ [ C[0] , C[1] , C[2] , C[3] , C[4] , C[5] ],
	[	C[1] , C[6] , C[7] , C[8] , C[9] , C[10]],
	[	C[2] , C[7] , C[11], C[12], C[13], C[14]],
	[	C[3] , C[8] , C[12], C[15], C[16], C[17]],
	[	C[4] , C[9] , C[13], C[16], C[18], C[19]],
	[	C[5] , C[10], C[14], C[17], C[19], C[20]] ]
	
	return C_matrix

def isotropic_projector_Facu(M):
    _alpha,_beta,E,nu=extract_isotropic_parameters( M )
    J_tensor4 = generate_J_tensor4()
    J_matrix4 = tensor4_to_voigt4( J_tensor4 )
    
    K_tensor4 = generate_K_tensor4()
    K_matrix4 = tensor4_to_voigt4( K_tensor4 )

    isotropic_M=dot(_alpha,J_matrix4)+dot(_beta,K_matrix4)

    return isotropic_M, _alpha,_beta, E, nu


def transverse_isotropic_projector(C,axis):
	"""input tensor has to be C, which is inverted into S, because relations between alpha beta etc and the 
	engineering constants E_l E_t etc are more complicated for C than for S"""
	S=linalg.inv(C)
	
	alpha,beta,gamma,gamma_prime,delta,delta_prime=extract_trans_iso_parameters_from_S(S,axis)

	trans_iso_S=generate_trans_isoS_from_params(alpha, beta, gamma, gamma_prime, delta, delta_prime,axis)
	trans_iso_C=linalg.inv(trans_iso_S)

	E_l=1/alpha
	nu_t=(delta-beta)/(delta+beta)
	E_t=(1-nu_t)/beta
	nu_l=-gamma*E_l/sqrt(2.)
	G_l=0.5/delta_prime

	return array(trans_iso_C), alpha,beta,gamma,gamma_prime,delta,delta_prime, E_l,E_t,nu_l,nu_t,G_l


def matrix_distance(M,isoM):
    diff_M = M - isoM
    
    # Frobenius' norm ||M|| = sqrt(trace(tM*M))
    return sqrt(trace(dot(diff_M.T,diff_M)))

def generate_isoC_from_E_nu(E,nu):
    _alpha=E/(1-2*nu)
    _beta=E/(1+nu)
    
    J_tensor4 = generate_J_tensor4()
    J_matrix4 = tensor4_to_voigt4( J_tensor4 )
    
    K_tensor4 = generate_K_tensor4()
    K_matrix4 = tensor4_to_voigt4( K_tensor4 )

    isoC=dot(_alpha,J_matrix4)+dot(_beta,K_matrix4)

    return isoC

def convert_E_nu_to_kappa_mu(E,nu):
    kappa   =E/(3*(1-2*nu))
    mu      =E/(2*(1+nu))
    return kappa,mu

def convert_kappa_mu_to_E_nu(kappa,mu):
    # kappa : module d'élasticité isostatique
    # mu    : module de cisaillement
    
    nu=(3*kappa-2*mu)/(6*kappa+2*mu)

    E=2*(1+nu)*mu

    return E,nu

def convert_kappa_mu_to_lambda_mu(kappa,mu):

    E_0,nu_0=convert_kappa_mu_to_E_nu(kappa, mu)

    lambda_0 = E_0*nu_0/((1 + nu_0)*(1 - 2*nu_0))
    mu_0 = E_0/(2*(1+nu_0))

    return lambda_0,mu_0

def generate_isoC_from_alpha_beta(alpha,beta):

    J_tensor4 = generate_J_tensor4()
    J_matrix4 = tensor4_to_voigt4( J_tensor4 )
    
    K_tensor4 = generate_K_tensor4()
    K_matrix4 = tensor4_to_voigt4( K_tensor4 )

    isoC=dot(alpha,J_matrix4)+dot(beta,K_matrix4)

    return isoC 

def sum_list_of_lists(A,B):

	my_sum=list(A)

	for i in range(3):
		for j in range(3):		
			for k in range(3):
				for l in range(3):
					my_sum[i][j][k][l]+=B[i][j][k][l]

	return my_sum

def multiply_list(list,scalar):

	for i in range(3):
		for j in range(3):
			for k in range(3):
				for l in range(3):
					list[i][j][k][l]=list[i][j][k][l]*scalar

	return list

def generate_trans_isoS_from_params(alpha,beta,gamma,gamma_prime,delta,delta_prime,axis):
	#auxiliairy matrices
	iT_matrix=generate_iT_matrix( axis )
	IT_matrix=generate_IT_matrix( axis )

	EL_tensor=			generate_EL_tensor( axis )
	JT_tensor=			generate_JT_tensor( generate_iT_matrix( axis ) )
	KE_tensor=			generate_KE_tensor( axis, generate_iT_matrix( axis )  )
	KT_tensor=			generate_KT_tensor( generate_IT_matrix( axis ), JT_tensor  )
	KL_tensor=			generate_KL_tensor( KT_tensor, KE_tensor )
	F_tensor,FT_tensor= generate_F_tensors( axis, iT_matrix)

	trans_iso_S=multiply_list(EL_tensor,alpha)
	trans_iso_S=sum_list_of_lists(trans_iso_S,multiply_list(JT_tensor,beta))
	trans_iso_S=sum_list_of_lists(trans_iso_S,multiply_list(F_tensor,gamma))
	trans_iso_S=sum_list_of_lists(trans_iso_S,multiply_list(FT_tensor,gamma_prime))
	trans_iso_S=sum_list_of_lists(trans_iso_S,multiply_list(KT_tensor,delta))
	trans_iso_S=sum_list_of_lists(trans_iso_S,multiply_list(KL_tensor,delta_prime))

	return tensor4_to_voigt4(trans_iso_S)

def generate_trans_isoC_from_E_nu_G(E_l,E_t,nu_l,nu_t,G_l,axis):
	alpha_S=1./E_l
	beta_S=(1.-nu_t)/E_t
	gamma_S=-sqrt(2.)*nu_l/E_l
	gamma_prime_S=gamma_S
	delta_S=(1.+nu_t)/E_t
	delta_prime_S=1./(2.*G_l)

	S=generate_trans_isoS_from_params(alpha_S,beta_S,gamma_S,gamma_prime_S,delta_S,delta_prime_S,axis)

	C_trans_iso=linalg.inv(S)

	return C_trans_iso

def sanity_check_isotropic(C,iso_C, alpha, beta, E, nu):

	print('\n##############################################################\n\n')
	print('                         Sanity Check		\n\n')
	print('##############################################################\n')

	test_C=isotropic_projector_Facu(iso_C)[0]

	print('\nOriginal C from craft output:\n')
	print(C)
	
	print('\n\nIsotropic projection of C:\n')
	print(iso_C)

	print('\n\nDistance of projection to original={0:0.6f}\n'.format(matrix_distance(C,iso_C)))

	print('\n\nIsotropic projection of iso_C (should be the same as above):\n')

	print(test_C)

	print('\n\nRebuilding of C through extracted E={0:0.4f},nu={1:0.4f}:\n'.format(E,nu))
	print(generate_isoC_from_E_nu(E,nu))

	print('\n\nRebuilding of C through extracted alpha={0:0.4f}, beta={1:0.4f}:\n'.format(alpha,beta))
	print(generate_isoC_from_alpha_beta(alpha,beta))

	print('\n\n##############################################################\n')

def sanity_check_trans_iso(C, trans_iso_C,alpha_S,beta_S,gamma_S,gamma_prime_S,delta_S,\
	delta_prime_S,E_l,E_t,nu_l,nu_t,G_l,axis ):

	print('\n##############################################################\n\n')
	print('                         Sanity Check		\n\n')
	print('##############################################################\n')

	test_C=transverse_isotropic_projector(C,axis)[0]

	print('\nOriginal C from craft output:\n')
	printVoigt4(C)
	
	print('\n\nTransversely isotropic projection of C:\n')
	printVoigt4(trans_iso_C)

	print('\n\nDistance of projection to original={0:0.6f}\n'.format(matrix_distance(C,trans_iso_C)))

	print('\n\nIsotropic projection of iso_C (should be the same as above):\n')

	printVoigt4(test_C)

	print('Rebuilding of C through extracted parameters:\n')
	print(' E_l\t\t={:0.6f},\n E_t\t\t={:0.6f},\n nu_l\t\t={:0.6f},\n nu_t\t\t={:0.6f},\n G_l\t\t={:0.6f}\n'.format(E_l,E_t,nu_l,nu_t,G_l) )
	printVoigt4(generate_trans_isoC_from_E_nu_G(E_l,E_t,nu_l,nu_t,G_l,axis))

	print('Rebuilding of C through extracted parameters:\n')
	print(' alpha_S\t={:0.6f},\n beta_S\t\t={:0.6f},\n gamma_S\t={:0.6f},\n delta_S\t={:0.6f},\n delta_prime_S\t={:0.6f}\n'.\
		format(alpha,beta,gamma,delta,delta_prime) )

	S=generate_trans_isoS_from_params(alpha_S,beta_S,gamma_S,gamma_prime_S,delta_S,delta_prime_S,axis)

	printVoigt4( linalg.inv(S) )

	print('\n\n##############################################################\n')
	

def printVoigt4(tensor4,prec_total=8,prec_decimal=4):
	"""tensor4 is a list[6][6], prec_total is the total number of digits, 
	prec_decimal is the number of decimal places"""
	prec="{{: >{}.{}f}}, ".format(prec_total,prec_decimal)
	precString="["+prec*6+"],"
	for i in range(6):
		print(precString.format(*tensor4[i]))
	print('\n')	

			
def modifyConvention(C):
    """Amitex and Abaqus output tensors are in notation 11 22 33 12 13 23, 
    this converts into notation 11 22 33 23 13 12"""

    temp=copy.deepcopy(C)
    temp[5]=C[3]
    temp[3]=C[5]

    C_mod=copy.deepcopy(temp)

    for i in range(6):
        C_mod[i][5]=temp[i][3]
        C_mod[i][3]=temp[i][5]       

    return C_mod

def write_to_file_Voigt4(
    filename,
    tensor4,
    prec_total=8,
    prec_decimal=4,
    material=None,
    material_tag=None,
    modifyConventionBool=True,
    loadingType=False,
    time=None,
    units=None
    ):

    """tensor4 is a list[6][6] (or array[numTimeSteps,6,6] for viscoelasticity), 
    prec_total is the total number of digits, prec_decimal is the number of decimal places"""
    prec="{{: >{}.{}f}}, ".format(prec_total,prec_decimal)
    precString="["+prec*6+"],\n"
    
    if loadingType == "relaxation":
        
        with open(filename,'w') as f:

            outputString=prec*39+"\n"
            for iTimeStep in range(len(tensor4)):
                
                print("\n timeStep:{}, time: {: >12.4f}".format(iTimeStep,time[iTimeStep]))

                printVoigt4(tensor4[iTimeStep])

                C_iso,alphaIso,betaIso,E,nu = isotropic_projector_Facu(modifyConvention(tensor4[iTimeStep]))
                
                outputStringTerminal="\nalpha\t= \t"+prec+"\nbeta\t= \t"+prec+"\nE\t\t= \t"+prec+"\nnu\t\t= \t"+prec+"\n"
                print(outputStringTerminal.format(alphaIso,betaIso,E,nu))

                f.write(outputString.format(time[iTimeStep],*(ravel(tensor4[iTimeStep])),alphaIso/3,betaIso/2))

                if iTimeStep==0:
                    E_l,E_t,nu_l,nu_t,G_l=transverse_isotropic_projector(
                        modifyConvention(tensor4[iTimeStep]),
                        axis=2
                    )[-5:]
            
            return (E_l,E_t,nu_l,nu_t,G_l)


    if loadingType in ["creep","StressTimeSeries"]:
        with open(filename,'w') as f:

            headerFormater="{{: <{}}}, ".format(prec_total)
            headerString=(headerFormater*13)[:-2]+"\n"

            unitsTime="m" if units["time"]=="minutes" else "s"
            unitsStress="MPa" if units["stress"]=="MPa" else "GPa"
            
            f.write(headerString.format(
                "Time ({})".format(unitsTime),
                "Stress_xx ({})".format(unitsStress),
                "Stress_yy ({})".format(unitsStress),
                "Stress_zz ({})".format(unitsStress),
                "Stress_xy ({})".format(unitsStress),
                "Stress_xz ({})".format(unitsStress),
                "Stress_yz ({})".format(unitsStress),
                
                "Strain_xx".format(unitsStress),
                "Strain_yy".format(unitsStress),
                "Strain_zz".format(unitsStress),
                "Strain_xy".format(unitsStress),
                "Strain_xz".format(unitsStress),
                "Strain_yz".format(unitsStress),
                )
            )

            outputString=(prec*13)[:-2]+"\n"
            for iTimeStep in range(len(tensor4[0])):
                stress=tensor4[0][iTimeStep]
                strain=tensor4[1][iTimeStep]
                                
                f.write(outputString.format(time[iTimeStep],*stress,*strain))
            
            return (False,False,False,False,False)

    elif loadingType == "elasticity":
        if modifyConventionBool:
            C=modifyConvention(tensor4) #Amitex suit la convention 11 22 33 12 13 23, remettre en 11 22 33 23 13 12
        else: 
            C=tensor4

        with open(filename,'w') as f:
            f.write("Resulting homogenized stiffness tensor\n\n")

            for i in range(6):
                f.write(precString.format(*tensor4[i]))
            if modifyConventionBool:
                f.write('\n/!\ This tensor is in notation 11 22 33 12 13 23 /!\ ')
            else:
                f.write('\n Usual convention')

            f.write("\n\nIsotropic parameters: \n")
            C_iso,alphaIso,betaIso,E,nu = isotropic_projector_Facu(C)
            outputString="\nalpha\t= \t"+prec+"\nbeta\t= \t"+prec+"\nE\t\t= \t"+prec+"\nnu\t\t= \t"+prec+"\n"
            f.write(outputString.format(alphaIso,betaIso,E,nu))
            f.write("\nthe following analysis is considering an alignment of fibers alows z axis (axis=2)\n\n")
            axis=2

            alpha, beta, gamma, gamma_prime, delta, delta_prime=extract_trans_iso_parameters_from_S(C ,axis)		

            f.write("alpha\t\t={: >10.3f}\nbeta\t\t={: >10.3f}\ngamma\t\t={: >10.3f}\ndelta\t\t={: >10.3f}\ndelta_prime\t={: >10.3f}\nalpha*beta-gamma^2\t={: >10.3f}\n".\
                format(alpha,beta,gamma,delta,delta_prime,alpha*beta-gamma**2))
            f.write('\n')
            E_l,E_t,nu_l,nu_t,G_l=transverse_isotropic_projector(C,axis)[-5:]
            f.write(
                "E_l\t\t={: >10.4f}\nE_t\t\t={: >10.4f}\nnu_l\t={: >10.4f}\nnu_t\t={: >10.4f}\nG_l\t\t={: >10.6f}\n".\
                format(E_l,E_t,nu_l,nu_t,G_l)
            )

            C_trans_iso=generate_trans_isoC_from_E_nu_G(E_l,E_t,nu_l,nu_t,G_l,axis)

            f.write("\nOriginal tensor in 11 22 33 23 13 12 notation: \n")
            for iRows in range(6):
                f.write(precString.format(*C[iRows]))

            f.write("\nTransverse isotropic projection: \n")
            for iRows in range(6):
                f.write(precString.format(*C_trans_iso[iRows]))

            distance_trans=matrix_distance(C,C_trans_iso)
            f.write("\nDistance (Frobenius norm), to transverse isotropic space:\n{: >10.3f}\n".\
                format(distance_trans))

            f.write("\nIsotropic projection: \n")
            for iRows in range(6):
                f.write(precString.format(*C_iso[iRows]))

            distance_iso=matrix_distance(C,C_iso)
            f.write("\nDistance (Frobenius norm), to isotropic space:\n{: >10.3f}\n".format(distance_iso))

            if material is not None:
                f.write("\n ############################################################")
                f.write("\n\n\tInput data\n")

                f.write("\nMaterial tag: {:>15}\n".format(material_tag))

                ### matrix

                f.write("\nMatrix:\n")

                if material["matrix"]["behavior"]=="iso":
                    for key,val in material["matrix"].items():
                        f.write("\n{:<10}:\t{:>6}".format(key,val))
                elif material["matrix"]["behavior"]=="orthotropic":
                    for iRows in range(6):
                        f.write(precString.format(*material["matrix"]["C"][iRows]))
                
                elif material["matrix"]["behavior"]=="viscoelas_maxwell":
                    f.write("\nviscoelastic properties:")
                    f.write("\nkappa_0:\t{:>6}".format(material["matrix"]["kappa_0"]))
                    f.write("\nmu_0   :\t{:>6}".format(material["matrix"]["mu_0"]))
                    f.write("\n chains: (kappa_i,nu_i,tau_i):")
                    
                    for chain in material["matrix"]["chains"]:
                        f.write("\n\t{:>6}\t{:>6}\t{:>6}".format(*chain))


                f.write("\n\nFiber:")
                if material["fiber"]["behavior"] in ["iso","trans_iso"]:
                    for key,val in material["fiber"].items():
                        f.write("\n{:<10}:\t{:>6}".format(key,val))
                elif material["fiber"]["behavior"]=="none":
                    f.write("\nNo fibers present")

        return (E_l,E_t,nu_l,nu_t,G_l)

        
        


######################################################################################

###    Sanity check



# # test transverse isotropic
# A11=10.
# A12=34.
# A13=53.
# A33=36.
# A44=25.

# #isotrope transverse selon e3
# AVoigt3=[[A11, A12, A13, 0,   0,   0],\
#          [A12, A11, A13, 0,   0,   0],\
#          [A13, A13, A33, 0,   0,   0],\
#          [0,   0,   0,   A44, 0,   0],\
#          [0,   0,   0,   0,   A44, 0],\
#          [0,   0,   0,   0,   0,   A11-A12]]

# #A=voigt4_to_tensor4(AVoigt3)
# axis=2
# alpha, beta, gamma, gamma_prime, delta, delta_prime = extract_trans_iso_parameters_from_S( AVoigt3 ,axis)


# print("alpha=\t\t{}\nbeta=\t\t{}\ngamma=\t\t{}\ngamma_prime=\t{}\ndelta=\t\t{}\ndelta_prime=\t{}\n".format(alpha, beta, gamma, gamma_prime, delta, delta_prime))


# test_A=generate_trans_isoS_from_params(alpha, beta, gamma, gamma_prime, delta, delta_prime,2)

# print('A=')
# printVoigt4(AVoigt3)

# print('test_A=')
# printVoigt4(test_A)

# print('isotropic distance= ',matrix_distance(array(AVoigt3),array(test_A)))

# print('###################################################')
# axis=0
# C=generate_trans_isoC_from_E_nu_G(180, 10, 0.28, 0.4, 7, axis )

# print('\ntransverse iso C=')
# printVoigt4(C)


# trans_iso_C, alpha_S,beta_S,gamma_S,gamma_prime_S,delta_S,delta_prime_S, E_l,E_t,nu_l,nu_t,G_l = \
#             transverse_isotropic_projector(C,axis)

# # sanity_check_trans_iso(C, trans_iso_C,alpha_S, beta_S,gamma_S,gamma_prime_S,\
# #                delta_S,delta_prime_S, E_l,E_t,nu_l,nu_t,G_l,axis )

# #
# C_iso=generate_isoC_from_E_nu(80,0.3)
# print('C_iso=')
# printVoigt4(C_iso)

# axis=0

# pseudo_trans_iso_C, alpha_S,beta_S,gamma_S,gamma_prime_S,delta_S,delta_prime_S, E_l,E_t,nu_l,nu_t,G_l = \
#             transverse_isotropic_projector(C_iso,axis)

# print("pseudo_trans_iso_C=")
# printVoigt4(pseudo_trans_iso_C)

# sanity_check_trans_iso(C_iso, pseudo_trans_iso_C,alpha_S, beta_S,gamma_S,gamma_prime_S,\
#                 delta_S,delta_prime_S, E_l,E_t,nu_l,nu_t,G_l,axis )



