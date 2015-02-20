import numpy as np
cimport numpy as np
from solver import solver
from diffusion import diffusion
ctypedef np.float64_t dtype_t
ctypedef np.int64_t itype_t

def step(np.ndarray[dtype_t, ndim=1] S, np.ndarray[dtype_t, ndim=1] B, np.ndarray[dtype_t, ndim=1] b_dot, float dt, int N, float t, int nx, int ny, float dx, METHOD):
	# A_tilde = np.empty
	# if A_tilde.size == 0:
		# A_tilde,C_tilde,nm_half,npl,mm_half,ml = isempty_A_tilde(A_GLEN,RHO,g,n_GLEN,dx,C_SLIDE,m_SLIDE)
	# t_n = t + dt
	# S_out = np.genfromtxt('S_out.txt')
	# return S_out,t_n
	cdef float t_n, OMEGA
	cdef np.ndarray[itype_t,ndim=1] ic_jc, ip_jc, im_jc, ic_jp, ic_jm, im_jm, ip_jm, im_jp, ip_jp
	cdef np.ndarray[itype_t,ndim=1] row, col
	cdef np.ndarray[dtype_t, ndim=1] val, C, S_out, D_IC_jc, D_ic_JC, D_IP_jc, D_ic_JP, D_sum

	ic_jc,im_jc,ip_jc,ic_jm,ic_jp,im_jm,ip_jm,im_jp,ip_jp = solver.SetupIndexArrays(nx,ny) 
	D_IC_jc, D_IP_jc, D_ic_JC, D_ic_JP = diffusion.diffusion_gl(S,B,nx,ny,dx)
	D_sum = D_IC_jc + D_IP_jc + D_ic_JC + D_ic_JP
	
	row = np.int64([[ic_jc],[ic_jc],[ic_jc],[ic_jc],[ic_jc]]).flatten()
	# row = row.T.reshape(row.size,1).T
	col = np.int64([[im_jc],[ip_jc],[ic_jm],[ic_jp],[ic_jc]]).flatten()
	# col = col.T.reshape(col.size,1).T
	val = np.array([[-OMEGA*D_IC_jc],[-OMEGA*D_IP_jc],[-OMEGA*D_ic_JC],[-OMEGA*D_ic_JP],[1/dt+OMEGA*D_sum]]).flatten()
	# row = row - 1
	# col = col - 1
	# A = csr_matrix( (val,(row,col)), shape=(N,N)).todense()   ### matrix A is symmetric positive definite
	# A = csr_matrix( (val,(row,col)), shape=(N,N))
	C = (1 - OMEGA) * ((D_IC_jc * S[im_jc]) + D_IP_jc * S[ip_jc] + D_ic_JC * S[ic_jm] + D_ic_JP * S[ic_jp]) + (1/dt - (1 - OMEGA) * D_sum) * S[ic_jc] + b_dot 
	C = C.flatten()

	print 'starting to solve'
	
	# S_out = solver(row,col,val,N,C,METHOD) 
	S_out = solver.solver(row,col,val,N,C,METHOD)
	print 'solved'
	# print S_out[0:10]
	S_out = solver.compareSB(S_out,B)
	# S_out[S_out<B] = B[B>S_out]
	print S_out[0:10]
	
	# H_out = S_out - B
	t_n = t + dt
	
	# D_max = np.max(D_IC_jc+D_IP_jc+D_ic_JC+D_ic_JP)
	# k_LAMBDA_max = np.argmax(D_IC_jc+D_IP_jc+D_ic_JC+D_ic_JP)
	# # D_max,k_LAMBDA_max = np.max(D_IC_jc+D_IP_jc+D_ic_JC+D_ic_JP)  ######### @matlab
	# LAMBDA_max = 0.25 * dt * D_max
	
	return S_out,t_n