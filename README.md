# anisotropic_mesh_laplacian
A repo to test a simple laplacian problem on an anisotropic mesh. The script computes the l2 error on a problem case with various anisotropic 2D meshes.

# Requirements 
`firedrake` should be installed : https://www.firedrakeproject.org 
`mmg` should be used for the meshing part : http://www.mmgtools.org
Look at the basic python modules required in the python script (csv, gmsh, etc), they are either already in firedrake environment, either can be installed with a one-liner using `pip`

# In order to run the code 
1. Activate the firedrake environment:
   ```
   source $FIREDRAKE_PATH/firedrake/bin/activate
   ```
2. Export a variable in your terminal that you call `MMG_PATH`, such that executables, in particular `mmg2d_O3` are located there, and :
   ```
   $MMG_PATH/mmg2d_O3
   ```
   would call the mmg 2d machinery.
3. Run the python script:
   ```
   python3 project_aniso_l2.py
   ```
