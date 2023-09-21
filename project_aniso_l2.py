import gmsh
import numpy as np
import firedrake as fd
import shutil
import os
import subprocess
from firedrake import inner, grad, dx
from ufl import tanh
import csv

physical_tag = 1
boundary_tag = 5
geo_filename = "rectangle.geo_unrolled"
msh_filename = geo_filename.replace(".geo_unrolled", ".msh")
aniso_msh_filename = geo_filename.replace(".geo_unrolled", "_aniso.msh")


def get_boundary_of_unitbox():
    """Get the border of the uniform rectangle, i.e the lines located at x=0,1 and y=0,1.

    Returns:
        boundary_tags(list): list of the number of the lines that are on the border of the rectangle.
    """
    boundary_tags = []
    lines = gmsh.model.occ.getEntities(dim=1)
    for line_dim, line_tag in lines:
        center_of_mass = gmsh.model.occ.getCenterOfMass(line_dim, line_tag)
        if np.isclose(center_of_mass[0], 0.0):
            boundary_tags.append(line_tag)
        elif np.isclose(center_of_mass[0], 1.0):
            boundary_tags.append(line_tag)
        elif np.isclose(center_of_mass[1], 0.0):
            boundary_tags.append(line_tag)
        elif np.isclose(center_of_mass[1], 1.0):
            boundary_tags.append(line_tag)
    return boundary_tags


def make_unit_box_mesh():
    """Make a unit box mesh with gmsh, domain [0,1]x[0,1] where the surface is reference with physical_tag and all the border with the same boundary_tag. The output is a .msh compatible with firedrake and a geo file to check the geometry."""
    gmsh.initialize()
    gmsh.model.add("simple_rectangle")
    gmsh.model.occ.add_rectangle(0, 0, 0, 1, 1)
    surfaces = gmsh.model.occ.getEntities(dim=2)
    gmsh.model.occ.synchronize()
    fluid_surfaces = [surface for (_, surface) in surfaces]
    gmsh.model.addPhysicalGroup(2, fluid_surfaces, physical_tag)
    gmsh.model.setPhysicalName(2, physical_tag, "Fluid volume")
    boundary_tags = get_boundary_of_unitbox()
    gmsh.model.addPhysicalGroup(1, boundary_tags, boundary_tag)
    gmsh.model.setPhysicalName(1, boundary_tag, "Boundary")
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Format", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.write(geo_filename)
    gmsh.model.mesh.generate(2)
    gmsh.write(msh_filename)
    gmsh.finalize()


def write_metric(metric, output_mesh_file: str = aniso_msh_filename):
    """Adapt a .msh mesh generated from gmsh  with constant metric, that can be meshed with mmg. (using it as input for mmg)

    Args:
        metric (_type_): the constant metric
        output_mesh_file (str, optional): the mesh file. Defaults to aniso_msh_filename.
    """
    final_file = f"tmp_{output_mesh_file}"
    End_infile = False
    N_nodes = 0
    with open(final_file, "w") as outFile:
        with open(output_mesh_file) as inFile:
            while not (End_infile):
                read_line = inFile.readline()
                outFile.write(read_line)
                if read_line == "$Nodes\n":
                    read_line = inFile.readline()
                    outFile.write(read_line)
                    N_nodes = int(read_line[:-1])
                if read_line == "$EndElements\n":
                    End_infile = True
                    inFile.close()
        # Now write the extra lines due to the metric
        outFile.write("$NodeData\n")
        outFile.write("1\n")
        outFile.write(""""special_emile_metric:metric"\n""")
        outFile.write("1\n")
        outFile.write("0\n")
        outFile.write("3\n")
        outFile.write("0\n")
        outFile.write("9\n")
        outFile.write(str(N_nodes) + "\n")
        # write a metric for each node
        for gg in range(N_nodes):
            sentence = f"{str(gg+1)} "
            for jj in range(9):
                if abs(metric[jj]) > 1e-30:
                    sentence += f"{np.around(metric[jj],decimals=25)} "
                else:
                    sentence += f"{0} "
            outFile.write(sentence + "\n")
        # Final line
        outFile.write("$EndNodeData\n")
    outFile.close()
    # Finally delete the tmp file
    shutil.copy(final_file, output_mesh_file)
    os.remove(final_file)


def adapt_channels_mmg_2d(
    constant_metric,
    input_mesh_file: str = msh_filename,
    output_mesh_file: str = aniso_msh_filename,
):
    """This script suppose that a bash variable MMG_PATH is set to the bin directory of your mmg3d executable
    Example : echo $MMG_PATH
              /Users/emilesoutter/ExtraSoftwares/mmg/bin/
             where the mmg exists there  /Users/emilesoutter/ExtraSoftwares/mmg/bin/mmg3d_O3

    Args:
        InputFile (_type_): _description_
        Outputfile (str, optional): _description_. Defaults to "Adapted_aniso.msh".
        folder (_type_, optional): _description_. Defaults to None.
        CL (int, optional): Characteristic length. Defaults to 1.
        N_adapt (int, optional): number of remeshing iterations. Defaults to 1.
        appctx (_type_, optional): Additional change to the metrics. example : attractors as lines or planes Defaults to None.
    """
    shutil.copy(input_mesh_file, output_mesh_file)
    write_metric(metric=constant_metric, output_mesh_file=output_mesh_file)
    # At this stage the Outputfile includes the metric that we want the only thing left is to call mmg
    mmg_path = os.environ.get("MMG_PATH")
    mmg2d = f"{mmg_path}mmg2d_O3"
    print(f"The mmg executable path used is currently {mmg2d}")
    subprocess.run(
        [
            mmg2d,
            output_mesh_file,
        ]
    )  #  option to tweal : -hgrad 5.0 -hausd 0.001, -opnbdy --> conserve better the boundary
    output_mesh_file2 = f"{output_mesh_file[:-3]}o.{output_mesh_file[-3:]}"
    shutil.copy(output_mesh_file2, output_mesh_file)


def solve_poisson_dubuis_problem(
    input_mesh_file: str = aniso_msh_filename,
    problem_tanh_amplitude_c: float = 50.0,
    save_paraview_solution: bool = False,
):
    """Solve on a given mesh the problem specified by Samuel Dubuis in a simple Laplacian case.

    Args:
        input_mesh_file (str, optional): the (anisotropic) input mesh. Defaults to aniso_msh_filename.
        problem_tanh_amplitude_c (float, optional): Constant that regulates the sharpness of the tanh function. Defaults to 50.0.
        save_paraview_solution (bool, optional): Save the solution in a paraview file. Defaults to False.
    """
    mesh = fd.Mesh(input_mesh_file)
    # Function spaces, test and trial functions
    V = fd.FunctionSpace(mesh, "CG", 1)
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    f = fd.Function(V)
    x, _ = fd.SpatialCoordinate(mesh)
    # rhs and weak formulations
    f.interpolate(
        2
        * problem_tanh_amplitude_c**2
        * tanh(problem_tanh_amplitude_c * (x - 0.5))
        * (1.0 - (tanh(problem_tanh_amplitude_c * (x - 0.5))) ** 2)
    )
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx
    u_solution = fd.Function(V)
    u_solution_exact = fd.Function(V)
    u_solution_expression = tanh(problem_tanh_amplitude_c * (x - 0.5))
    u_solution_exact.interpolate(u_solution_expression)
    fd.solve(a == L, u_solution)
    # By default compute the error norm between the 2 solutions, either the expression can be compared in itself, either the solution already projected on the function space
    # error = fd.errornorm(u_solution_expression, u_solution, degree_rise=3)
    error = fd.errornorm(u_solution_exact, u_solution)
    print(
        f"For the case with input file {input_mesh_file} the error is {error}, and the number of unkowns was {V.dim()}"
    )
    if save_paraview_solution:
        u_solution.rename("numerical_solution")
        u_solution_exact.rename("exact_solution")
        difference = fd.Function(V)
        difference.assign(u_solution_exact - u_solution)
        difference.rename("difference: u_h-u_exact")
        fd.File(f"numerical_case_{input_mesh_file[:-3]}.pvd").write(
            u_solution, u_solution_exact, difference
        )
    return error


def run_given_problem_case(
    aniso_output_mesh_file_name: str = aniso_msh_filename,
    mesh_size_x: float = 0.01,
    mesh_size_y: float = 0.1,
    input_mesh_file: str = msh_filename,
    save_paraview_solution: bool = False,
):
    """Given the specific of the input, compute the "Dubuis" example problem, i.e mesh the anisotropic mesh with mmg, based on an input gmsh .msh mesh format, then use the anisotropic mesh as input to solve the Laplace/poisson problem and compute the error. An option allows to save the solution in paraview, for debugging.

    Args:
        aniso_output_mesh_file_name (str, optional): Name of the anisotropic mesh. Defaults to aniso_msh_filename.
        mesh_size_x (float, optional): the mesh discretization along x. Defaults to 0.01.
        mesh_size_y (float, optional): the mesh discretization along y. Defaults to 0.1.
        input_mesh_file (str, optional): the input mesh of the domain, meshed with a classical gmsh tool. Defaults to msh_filename.
        save_paraview_solution (bool, optional): option to save or not the solution in paraview. Defaults to False.

    Returns:
        float: the L2 error computed on the mesh with analytical solution
    """
    problem_metric = np.zeros(9)
    problem_metric[0] = 1.0 / (mesh_size_x) ** 2
    problem_metric[4] = 1.0 / (mesh_size_y) ** 2
    adapt_channels_mmg_2d(
        input_mesh_file=input_mesh_file,
        output_mesh_file=aniso_output_mesh_file_name,
        constant_metric=problem_metric,
    )
    problem_error = solve_poisson_dubuis_problem(
        input_mesh_file=aniso_output_mesh_file_name,
        save_paraview_solution=save_paraview_solution,
    )
    return problem_error


if __name__ == "__main__":
    # make the default gmsh mesh
    make_unit_box_mesh()
    # input parameters sweep
    mesh_size_x_list = [
        0.01,
        0.005,
        0.0025,
        0.00125,
        0.000625,
        0.01,
        0.005,
        0.0025,
        0.00125,
        0.000625,
        0.005,
        0.0025,
        0.00125,
        0.000625,
        0.0003125,
    ]
    mesh_size_y_list = [
        0.1,
        0.05,
        0.025,
        0.0125,
        0.00625,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
    ]
    mesh_name_list = [
        f"aniso_h1_{hx}_h2_{hy}.msh"
        for hx, hy in zip(mesh_size_x_list, mesh_size_y_list)
    ]
    error_list = []
    for hx, hy, mesh_name in zip(mesh_size_x_list, mesh_size_y_list, mesh_name_list):
        error = run_given_problem_case(
            mesh_size_x=hx,
            mesh_size_y=hy,
            aniso_output_mesh_file_name=mesh_name,
        )
        error_list.append(error)
    # save the final results in your favourite format, here csv
    output_headers = ["h_x", "h_y", "error l2", "mesh file name"]
    # Specify the file path
    csv_file_path = "dubuis_problem_output.csv"
    # Combine the headers and parameter lists
    data = [output_headers]  # The first row is the header
    data.extend(zip(mesh_size_x_list, mesh_size_y_list, error_list, mesh_name_list))

    # Write the data to the CSV file
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file with headers has been created: {csv_file_path}")
