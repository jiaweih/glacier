from solver import solver
import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
ctypedef np.int64_t itype_t

# n_GLEN = 3          # Glen's flow law exponent
# A_GLEN = 7.5738e-17 #6.05904e-18; Monthly #7.5738e-17 Cuffey & Paterson (4th ed) Glen's law parameter in Pa^{-3} yr^{-1} units (same as A_GLEN=2.4e-24 Pa^{-3} s^{-1})

# m_SLIDE = 2        # Sliding law exponent
# C_SLIDE = 0    # 1.0e-08;  # 1.0e-06;  # Sliding coefficient in Pa, metre,(Year units)

# RHO = 900   # Density (SI units)
# g = 9.80    # Gravity (SI units, rho*g has units of Pa)
# K_eps = 1.0e-12
# OMEGA = 1.5  # 1.6

def diffusion_gl(np.ndarray[dtype_t, ndim=1] S, np.ndarray[dtype_t, ndim=1] B, int nx, int ny, float dx):
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
	
	SB = S
	SB[SB<0] = 0
	H = SB

	ic_jc,im_jc,ip_jc,ic_jm,ic_jp,im_jm,ip_jm,im_jp,ip_jp = solver.SetupIndexArrays(nx,ny) 
	
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