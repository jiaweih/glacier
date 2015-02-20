from scipy.sparse import csr_matrix
import numpy as np
cimport numpy as np
from scipy.sparse import linalg

ctypedef np.float64_t dtype_t
# def solver(np.ndarray[np.int, ndim=1] row, np.ndarray[np.int, ndim=1] col, np.ndarray[np.float64_t, ndim=1] val, int N, np.ndarray[np.float64_t, ndim=1] C,METHOD):
def solver(np.ndarray[np.int32_t, ndim=1] row, np.ndarray[np.int32_t, ndim=1] col, np.ndarray[dtype_t, ndim=1] val, int N, np.ndarray[dtype_t, ndim=1] C, METHOD):
	np.ndarray[dtype_t, ndim=2] S_out
	upper = str.upper(METHOD)
	A = csr_matrix( (val,(row,col)), shape=(N,N))
	if upper == 'BACKSLASH':
		# matlab = matlab_wrapper.MatlabSession()
		# matlab.put('A',A)
		# matlab.put('C',C)
		# matlab.eval('x = A\C')
		# S_out = matlab.get('x')
		S_out = linalg.spsolve(A,C)
		#S_out = np.linalg.lstsq(A,C)
		#S_out = linalg.lsqr(A,C)
		#S_out = np.linalg.solve(A,C)
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
	return S_out.flatten()

def compareSB(np.ndarray[dtype_t,ndim=1] S_out, np.ndarray[dtype_t,ndim=1] B):
	cdef int size, i
	size = len(S_out)
	for i in np.arange(size):
		if S_out[i] < B[i]:
			S_out[i] = B[i]
	# S_out[S_out<B] = B[B>S_out]
	return S_out

def find_max(np.ndarray[dtype_t,ndim=1] S, np.ndarray[dtype_t,ndim=1] B):
	cdef int k_H_max, k_S_max
	cdef flat H_max, S_max
	cdef np.ndarray[dtype_t, ndim=1] SB
	SB = S - B
	# print 'Starting to calculate max'
	H_max = np.max(SB)
	# print 'H_max done'
	k_H_max = np.argmax(SB)
	# print 'k_H_max: {}'.format(k_H_max)
	S_max = np.max(S)
	# print 'S_max'
	k_S_max = np.argmax(S)
	# print 'k_S_max:  {}'.format(k_S_max)
	return H_max,k_H_max,S_max,k_S_max