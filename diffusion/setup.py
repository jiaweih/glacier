from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext  =  [Extension( "diffusion", sources=["diffusion_gl.pyx"] )]

setup(
   name = "diffusion", 
   cmdclass={'build_ext' : build_ext}, 
   include_dirs = [np.get_include()],   
   ext_modules=ext
   )