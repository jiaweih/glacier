!#/usr/bin/env python

from pysparse.itsolvers import krylov
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime as datetime
import scipy

global      ic_jc ip_jc im_jc ic_jp ic_jm ip_jm im_jm im_jp   #### where are they


mat = scipy.io.loadmat(infile)
B = mat['B']
b_dot = mat['b_dot']
dx = mat['dx']
dy = mat['dy']
i = mat['i'][0,0] ##
j = mat['j'][0,0] ##
nx = mat['nx'][0,0]
ny = mat['ny'][0,0]
N = SetupIndexArrays(nx,ny) ###### put in global


'''
Define physical parameters (here assuming EISMINT-1 values)
'''

n_GLEN = 3          # Glen's flow law exponent
A_GLEN = 7.5738e-17 #6.05904e-18; Monthly #7.5738e-17 Cuffey & Paterson (4th ed) Glen's law parameter in Pa^{-3} yr^{-1} units (same as A_GLEN=2.4e-24 Pa^{-3} s^{-1})

m_SLIDE = 2        # Sliding law exponent
C_SLIDE = 0    # 1.0e-08;  # 1.0e-06;  # Sliding coefficient in Pa, metre,(Year units)

RHO = 900   # Density (SI units)
g = 9.80    # Gravity (SI units, rho*g has units of Pa)
K_eps = 1.0e-12
OMEGA = 1.5  # 1.6

MODEL = 3
METHOD = 'BACKSLASH'   ### 'PCG'
B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt = model_params(MODEL)
    
def main():
	print_out(METHOD,OMEGA,dt,A_GLEN,C_SLIDE,nx,ny)
	timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

	B = B.reshape(N,1)
	B[np.isnan(B)] = 0

	b_dot = b_dot.reshape(N,1)
	S = B

	### Display some diagnostics before proceeding with model integration
	t = 0
	t_SAVE = dt_SAVE
	tm = time.clock()

	while 1:
	    S,t,LAMBDA_max,k_LAMBDA_max = step(S, B, b_dot, dt, N, t, METHOD)
	    H_max, k_H_max = np.max(S - B)
	    S_max, k_S_max = np.max(S)
	    ALPHA_I = 100*np.sum(S>B)/N
	    
	    print 'BKS: At t={:8.2f} yr ALPHA_I={:.2f}%% and maxima are: H({:d}) = {:f} \
	            S({:d})={:f} LAMBDA({:D}) = {:f}\n'.format(t, ALPHA_I, k_H_max, H_max, k_S_max, S_max, k_LAMBDA_max, LAMBDA_max)

	    if t > t_STOP:
	        break
	    
	    I = np.zeros(ny,nx)
	    I[S>B] = 1
	    
	    plt.imshow(I)
	    plt.title('Location: {:s}; t = {:.2f} yr'.format(run_str, t))
		file_str = print '{:s}_{:05d}'.format(run_str, round(t))

	    S_map = S.reshape(ny,nx)
	    B_map = B.reshape(ny,nx)
	    I_map = I.reshape(ny,nx)
	    now = datetime.datetime.now().strftime('%H:%M:%S')
	    print 'main(): Output stored in file "{:s}" at time {:s} \n'.format('{}.mat'.format(file_str),now)
	    
	    t_SAVE = t_SAVE + dt_SAVE

	e = time.clock() - tm
	print 'ALL DONE: Forward modelling'


def model_params(MODEL):
	case = {0: case_0, 1: case_1, 2: case_2, 3: case_3, 4: case_4}
    if MODEL in range(5):
        B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt = case[MODEL]()
    else:
        print 'main_forward(): Unprogrammed MODEL'
    return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt

def read_mat(infile):
    mat = scipy.io.loadmat(infile)
    B = mat['B']
    b_dot = mat['b_dot']
    dx = mat['dx']
    dy = mat['dy']
    i = mat['i'][0,0] ##
    j = mat['j'][0,0] ##
    nx = mat['nx'][0,0]
    ny = mat['ny'][0,0]
    return B,b_dot,dx,dy,ny,nx

def case_0():
    run_str = 'Toy model'
    dt = 1000
    dt_SAVE = 5 * dt
    t_STOP = 25000
    
    ny = 11
    nx = 11
    N = nx*ny
    
    dx = 100    ### 200
    dy = 100    ### 200
    
    x = np.linspace(0, dx*(nx - 1), nx)
    y = np.linspace(dx*(ny - 1), 0, ny)
    
    L_x = dx*(nx - 1)
    L_y = dy*(ny - 1)
    
    R0 = 0.5*L_x
    
    x_c = 0.5*L_x
    y_c = 0.5*L_y
    
    X, Y = np.meshgrid(x,y)
    
    Z0 = 2000
    sigma_x = x_c
    sigma_y = y_c
    R2 = np.square(X-x_c) + np.square(Y-y_c)
    B = Z0*np.exp(-R2/R0^2)  ##
    
    B_min = np.min(B)   ##
    B_max = np.max(B)   ##
    b_dot_melt = -2 + 2*(B - B_min)/(B_max - B_min)
    b_dot_ppt = 1
    b_dot = b_dot_melt + b_dot_ppt
    
    B[5,6] = B[5,6] - 100
    B[6,6] = B[6,6] + 100
    B[7,6] = B[7,6] - 100
    
    B[6,5] = B[6,5] + 200
    B[6,7] = B[6,7] + 200
    
    B = B + 5000
    return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt

def case_1():
    run_str = 'problem_1'
    file_dat = '{}.mat'.format(run_str)
    B,b_dot,dx,dy,ny,nx = read_mat(file_dat)
    
    t_STOP = 500
    dt_SAVE = 5*t_STOP
    dt = 1
    return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt

def case_2():
    run_str = 'problem_1 - ascii'
    file_dat = os.path.join('/', 'data', 'LettenmaierCoupledModel', 'Onestep', 'v_200', 'data', 'problem_1.dat')
 ##   [B, b_dot, dx, dy, ny, nx] = LoadAsciiData(file_dat);
 	#B,b_dot,dx,dy,ny,nx = read_mat(file_dat)
    t_STOP = 500
    dt_SAVE = 5*t_STOP
    dt = 1
    return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt

def case_3():
    run_str = 'mb4_spin1'
    #file_dat = os.path.join('M:', 'DHSVM', 'washington', 'cascade','spin','{}.mat'.format(run_str)) #####
    file_dat = '{}.mat'.format(run_str)
    B,b_dot,dx,dy,ny,nx = read_mat(file_dat)
    
    t_STOP = 1000
    dt_SAVE = 5*t_STOP
    dt = 0.08333
    return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt

def case_4():
    run_str = 'manipulate9_mth'
    file_dat = os.path.join('M:', 'DHSVM', 'bolivia', 'spinup', 'spin_up','{}.mat'.format(run_str))
    B,b_dot,dx,dy,ny,nx = read_mat(file_dat)
    
    t_STOP = 12000
    dt_SAVE = 5*t_STOP
    dt = 1.0
    return B,b_dot,dx,dy,ny,nx,t_STOP,dt_SAVE,dt

def print_out(METHOD,OMEGA,dt,A_GLEN,C_SLIDE,nx,ny):
    print '=================================================================================='
    print 'LAUNCHING GLACIER SIMULATION MODEL - Ver 5.01 using the {:s} solver\n\n'.format(METHOD)
    print '  OMEGA      = {:.2f}\n'.format(OMEGA)
    print '  dt         = {:.2f} yr\n'.format(dt)
    print '  A_GLEN     = {:e}\n'.format(A_GLEN)
    print '  C_SLIDE    = {:e}\n'.format(C_SLIDE)
    print '  nx         = {:d}\n'.format(nx)
    print '  ny         = {:d}\n'.format(ny)
    if METHOD.find('ADI') != -1:
        print '  ADI METHOD = {%s}\n'.format(ADI_METHOD)
    print '=================================================================================='


def step(S, B, b_dot, dt, N, t, METHOD):
    if A_tilde.size == 0:
        A_tilde,C_tilde,nm_half,npl,mm_half,ml = isempty_A_tilde(A_GLEN,RHO,g,n_GLEN,dx,C_SLIDE,m_SLIDE)
        
    D_IC_jc, D_IP_jc, D_ic_JC, D_ic_JP = diffusion_gl(S, B)
    D_sum = D_IC_jc + D_IP_jc + D_ic_JC + D_ic_JP
    
    row = np.array([ic_jc,ic_jc,ic_jc,ic_jc,ic_jc]).reshape(5,1)
    col = np.array([im_jc,ip_jc,ic_jm,ic_jp,ic_jc]).reshape(5,1)
    val = np.array([-OMEGA*D_IC_jc, -OMEGA*D_IP_jc, -OMEGA*D_ic_JC, -OMEGA*D_ic_JP,1/dt+OMEGA*D_sum]).reshape(5,1)
    
    row = row - 1
    col = col - 1
    A = csr_matrix( (val,(row,col)), shape=(N,N))   ### matrix A is symmetric positive definite
    C = (1 - OMEGA) * ((D_IC_jc * S[im_jc]) + D_IP_jc * S[ip_jc] + D_ic_JC * S[ic_jm] + D_ic_JP * S[ic_jp]) \
        + (1/dt - (1 - OMEGA) * D_sum) * S[ic_jc] + b_dot 
        
    S_out = solver(A,C,METHOD)
    S_out[S_out<B] = B[B>S_out]
    
    H_out = S_out - B
    t_n = t + dt
    
    D_max,k_LAMBDA_max = np.max(D_IC_jc+D_IP_jc+D_ic_JC+D_ic_JP)  #########
    LAMBDA_max = 0.25 * dt * D_max
    
    return S,t,LAMBDA_max,k_LAMBDA_max

def solver(A,C,METHOD):
    upper = str.upper(METHOD)
    if upper == 'BACKSLASH':
        S_out = np.linalg.solve(A,C)
    elif upper == 'PCG':
        tol = 1.0e-06       ### 1.0e-09;
        MAXIT = 100
        S_out = np.empty(N*N)
        info, itern, relres = krylov.pcg(A, C, S_out, tol, MAXIT)
        if info != 0:
            if info == -1:
                print 'step(): pcg did not converge after MAXIT iterations at t={:f} yr'.format(t)
            elif info == -2:
                print 'step(): pcg preconditioner is ill-conditioned at t={:f} yr'.format(t)
            elif info == -5:
                print 'step(): pcg stagnated at t={:f} yr'.format(t)
            elif info == -6:
                print 'step(): one of the scalar quantities in pcg became too small or too large at t={:f} yr'.format(t)
            else:
                print 'step(): Unknown pcg flag raised, flat={}'.format(info)
    return S_out

### if A_tilde.size == 0, then execute the following function
def isempty_A_tilde(A_GLEN,RHO,g,n_GLEN,dx,C_SLIDE,m_SLIDE):
    A_tilde = 2*A_GLEN*(RHO*g)**n_GLEN/((n_GLEN+2)*dx**2)
    C_tilde = C_SLIDE*(RHO*g)**m_SLIDE/dx**2
    nm_half = (n_GLEN-1)/2
    npl = n_GLEN+1
    mm_half = (m_SLIDE-1)/2
    ml = m_SLIDE
    return A_tilde,C_tilde,nm_half,npl,mm_half,ml

if __name__ == "__main__":
	main()