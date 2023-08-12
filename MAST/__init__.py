import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NCCL_P2P_LEVEL'] = '2'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'  # make pybullet render with GPU
# print("MAST.__init__: Setting OMP and MKL num threads to 1, NCCL_P2P_LEVEL=2, MESA version to 330")
