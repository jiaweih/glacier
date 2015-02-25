from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
ctypedef np.int64_t itype_t

def step(np.ndarray[dtype_t, ndim=1] S, np.ndarray[dtype_t, ndim=1] B, np.ndarray[dtype_t, ndim=1] b_dot, float dt, int N, float t, int nx, int ny, float dx, METHOD):
	cdef float t_n, OMEGA
	cdef np.ndarray[itype_t,ndim=1] ic_jc, ip_jc, im_jc, ic_jp, ic_jm, im_jm, ip_jm, im_jp, ip_jp
	cdef np.ndarray[itype_t,ndim=1] row, col
	cdef np.ndarray[dtype_t, ndim=1] val, C, S_out, D_IC_jc, D_ic_JC, D_IP_jc, D_ic_JP, D_sum

	ic_jc,im_jc,ip_jc,ic_jm,ic_jp,im_jm,ip_jm,im_jp,ip_jp = SetupIndexArrays(nx,ny) 
	D_IC_jc, D_IP_jc, D_ic_JC, D_ic_JP = diffusion_gl(S,B,nx,ny,dx)
	D_sum = D_IC_jc + D_IP_jc + D_ic_JC + D_ic_JP
	
	row = np.int64([[ic_jc],[ic_jc],[ic_jc],[ic_jc],[ic_jc]]).flatten()
	col = np.int64([[im_jc],[ip_jc],[ic_jm],[ic_jp],[ic_jc]]).flatten()
	val = np.array([[-OMEGA*D_IC_jc],[-OMEGA*D_IP_jc],[-OMEGA*D_ic_JC],[-OMEGA*D_ic_JP],[1/dt+OMEGA*D_sum]]).flatten()
	C = (1 - OMEGA) * ((D_IC_jc * S[im_jc]) + D_IP_jc * S[ip_jc] + D_ic_JC * S[ic_jm] + D_ic_JP * S[ic_jp]) + (1/dt - (1 - OMEGA) * D_sum) * S[ic_jc] + b_dot 
	C = C.flatten()	
	S_out = solver(row,col,val,N,C,METHOD)
	S_out = compareSB(S_out,B)
	t_n = t + dt
	
	return S_out,t_n

cdef diffusion_gl(np.ndarray[dtype_t, ndim=1] S, np.ndarray[dtype_t, ndim=1] B, int nx, int ny, float dx):
	cdef float A_tilde, C_tilde, nm_half, npl, ml, K_eps, A_GLEN, g, OMEGA, n_GLEN
	cdef int m_SLIDE, C_SLIDE, RHO
	cdef np.ndarray[dtype_t, ndim=1] SB, H
	cdef np.ndarray[itype_t, ndim=1] ic_jc,im_jc,ip_jc,ic_jm,ic_jp,im_jm,ip_jm,im_jp,ip_jp
	cdef np.ndarray[dtype_t, ndim=1] H_IC_jc, H_ic_JC, H_IC_jc_up, H_ic_JC_up, dS_dx_IC_jc, dS_dy_IC_jc, dS_dx_ic_JC, dS_dy_ic_JC
	cdef np.ndarray[dtype_t, ndim=1] S2_IC_jc, S2_ic_JC, D_IC_jc, D_ic_JC, D_IP_jc, D_ic_JP

	A_tilde = 2*A_GLEN*(RHO*g)**n_GLEN/(n_GLEN+2)/(dx**2)
	C_tilde = C_SLIDE*(RHO*g)**m_SLIDE/(dx**2)
	nm_half = (n_GLEN-1)/2
	npl = n_GLEN+1
	mm_half = (m_SLIDE-1)/2
	ml = m_SLIDE
	
	SB = S - B
	SB[SB<0] = 0
	H = SB

	ic_jc,im_jc,ip_jc,ic_jm,ic_jp,im_jm,ip_jm,im_jp,ip_jp = SetupIndexArrays(nx,ny) 
	
	H_IC_jc = 0.5*(H[ic_jc] + H[im_jc])
	H_ic_JC = 0.5*(H[ic_jc] + H[ic_jm])
	
	H_IC_jc_up = H[im_jc]
	H_ic_JC_up = H[ic_jm]
	
	ix = (S[ic_jc]>S[im_jc]).reshape(-1)
	H_IC_jc_up[S[ic_jc]>S[im_jc]] = H[ic_jc[ix]].reshape(-1)

	ix = (S[ic_jc]>S[ic_jm]).reshape(-1)
	H_ic_JC_up[S[ic_jc]>S[ic_jm]] = H[ic_jc[ix]].reshape(-1)
	
	dS_dx_IC_jc = (S[ic_jc]-S[im_jc])/dx
	dS_dy_IC_jc = (S[ic_jp]+S[im_jp]-S[ic_jm]-S[im_jm])/(4*dx)
	dS_dx_ic_JC = (S[ip_jc]+S[ip_jm]-S[im_jc]-S[im_jm])/(4*dx)
	dS_dy_ic_JC = (S[ic_jc]-S[ic_jm])/dx
	
	S2_IC_jc = np.square(dS_dx_IC_jc) + np.square(dS_dy_IC_jc) + K_eps
	S2_ic_JC = np.square(dS_dx_ic_JC) + np.square(dS_dy_ic_JC) + K_eps
	
	if C_tilde == 0:    ### No sliding case
		D_IC_jc = A_tilde*H_IC_jc_up*np.power(H_IC_jc,npl)*np.power(S2_IC_jc,nm_half)
		D_ic_JC = A_tilde*H_ic_JC_up*np.power(H_ic_JC,npl)*np.power(S2_ic_JC,nm_half)
	elif C_tilde > 0:    ### Sliding case
		D_IC_jc = A_tilde*H_IC_jc_up*np.power(H_IC_jc,npl)*np.power(S2_IC_jc,nm_half) \
				+ C_tilde*H_IC_jc_up*np.power(H_IC_jc,ml)*np.power(S2_IC_jc,mm_half)
		D_ic_JC = A_tilde*H_ic_JC_up*np.power(H_ic_JC,npl)*np.power(S2_ic_JC,nm_half) \
				+ C_tilde*H_ic_JC_up*np.power(H_ic_JC,ml)*np.power(S2_ic_JC,mm_half)
	else:
		print 'diffusion(): C_tilde is undefined or incorrectly defined'
		
	D_IP_jc  = D_IC_jc[ip_jc]
	D_ic_JP  = D_ic_JC[ic_jp]
	
	return D_IC_jc,D_IP_jc,D_ic_JC,D_ic_JP

cdef np.ndarray[dtype_t, ndim=1] solver(np.ndarray[itype_t, ndim=1] row, np.ndarray[itype_t, ndim=1] col, np.ndarray[dtype_t, ndim=1] val, int N, np.ndarray[dtype_t, ndim=1] C, METHOD):
	cdef np.ndarray[dtype_t, ndim=1] S_out
	upper = str.upper(METHOD)
	A = csr_matrix( (val,(row,col)), shape=(N,N))
	if upper == 'BACKSLASH':
		S_out = linalg.spsolve(A,C)
	# elif upper == 'PCG':
	# 	tol = 1.0e-06       ### 1.0e-09;
	# 	MAXIT = 100
	# 	S_out = np.empty(N*N)
	# 	info, itern, relres = krylov.pcg(A, C, S_out, tol, MAXIT)
	# 	if info != 0:
	# 		if info == -1:
	# 			print 'step(): pcg did not converge after MAXIT iterations at t={:f} yr'.format(t)
	# 		elif info == -2:
	# 			print 'step(): pcg preconditioner is ill-conditioned at t={:f} yr'.format(t)
	# 		elif info == -5:
	# 			print 'step(): pcg stagnated at t={:f} yr'.format(t)
	# 		elif info == -6:
	# 			print 'step(): one of the scalar quantities in pcg became too small or too large at t={:f} yr'.format(t)
	# 		else:
	# 			print 'step(): Unknown pcg flag raised, flat={}'.format(info)
	return S_out

cdef np.ndarray[dtype_t,ndim=1] compareSB(np.ndarray[dtype_t,ndim=1] S_out, np.ndarray[dtype_t,ndim=1] B):
	cdef int size, i
	size = len(S_out)
	for i in np.arange(size):
		if S_out[i] < B[i]:
			S_out[i] = B[i]
	return S_out

def find_max(np.ndarray[dtype_t,ndim=1] S, np.ndarray[dtype_t,ndim=1] B):
	cdef int k_H_max, k_S_max
	cdef float H_max, S_max
	cdef np.ndarray[dtype_t, ndim=1] SB
	SB = S - B
	H_max = np.max(SB)
	k_H_max = np.argmax(SB)
	S_max = np.max(S)
	k_S_max = np.argmax(S)
	return H_max,k_H_max,S_max,k_S_max

cdef np.ndarray[itype_t,ndim=1] setupArrays(np.ndarray[itype_t,ndim=1] a, np.ndarray[itype_t,ndim=1] b, np.ndarray[itype_t,ndim=2] ic_jc):
	cdef np.ndarray[itype_t,ndim=2] x,y
	x,y = np.meshgrid(b,a)
	array = []
	for l in zip(y.ravel(),x.ravel()):
		array.append(ic_jc[l])
	array = np.array(array)
	return array

cdef SetupIndexArrays(int nx, int ny):
	cdef int N
	cdef np.ndarray[itype_t,ndim=1] ic_jc_t, ic, ip, im, jc, jp, jm, ip_jc, im_jc, ic_jp, ic_jm, im_jm, ip_jm, im_jp, ip_jp
	N = nx * ny

	ic_jc_t = np.arange(1,N+1)  
	ic_jc = ic_jc_t.reshape(nx,ny)

	ic = np.arange(nx)
	ip = np.append(np.array([np.arange(1,nx)]),nx - 1)
	im = np.append(0,np.array([np.arange(nx - 1)]))

	jc = np.arange(ny)
	jp = np.append(0,np.array([np.arange(ny - 1)]))
	jm = np.append(np.array([np.arange(1,ny)]),ny - 1)

	ip_jc = setupArrays(ip,jc,ic_jc) - 1
	im_jc = setupArrays(im,jc,ic_jc) - 1
	ic_jp = setupArrays(ic,jp,ic_jc) - 1
	ic_jm = setupArrays(ic,jm,ic_jc) - 1

	im_jm = setupArrays(im,jm,ic_jc) - 1
	ip_jm = setupArrays(ip,jm,ic_jc) - 1
	im_jp = setupArrays(im,jp,ic_jc) - 1
	ip_jp = setupArrays(ip,jp,ic_jc) - 1

	ic_jc = ic_jc.reshape(-1) - 1

	return ic_jc,im_jc,ip_jc,ic_jm,ic_jp,im_jm,ip_jm,im_jp,ip_jp
