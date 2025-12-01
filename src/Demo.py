"""
PURPOSE OF THE SCRIPT
Analyze configuration of interest
Using the Veriational FE formulationin its NON-dimensional formulation (see "models/adimensional.py").
Dimensionless parameters:
beta := R/h >> 1
gamma := p0/E << 1
f := gamma beta**4

ARTICLE RELATED SECTION
"""

import json
import logging
import os
import pdb
import sys
from pathlib import Path
import importlib.resources as pkg_resources
import shutil

import dolfinx
import dolfinx.plot
from dolfinx import log
import dolfinx.io
from dolfinx.io import XDMFFile, gmshio
import dolfinx.mesh
from dolfinx.fem import Constant, dirichletbc
from dolfinx.fem.petsc import (assemble_matrix, create_vector, create_matrix, assemble_vector)

import ufl
from ufl import (CellDiameter, FacetNormal, dx)

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import petsc4py

import numpy as np

import yaml
import warnings
import basix

import matplotlib
from matplotlib.ticker import PercentFormatter, ScalarFormatter, MaxNLocator
matplotlib.use('WebAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyvista

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm

import adios4dolfinx as adios

from models.adimensional import A_NonlinearPlateFVK
from meshes import mesh_bounding_box
from meshes.primitives import mesh_circle_gmshapi
#from disclinations.utils import Logging
from utils.la import compute_cell_contributions, compute_disclination_loads
from utils.viz import plot_scalar, plot_profile, plot_mesh
from utils.sample_function import sample_function, interpolate_sample
from solvers import SNESSolver, SNESProblem
from visuals import visuals
visuals.matplotlibdefaults(useTex=False)

logging.basicConfig(level=logging.INFO)

REQUIRED_VERSION = "0.8.0"

if dolfinx.__version__ != REQUIRED_VERSION:
    warnings.warn(f"We need dolfinx version {REQUIRED_VERSION}, but found version {dolfinx.__version__}. Exiting.")
    sys.exit(1)


petsc4py.init(sys.argv)
log.set_log_level(log.LogLevel.WARNING)

comm = MPI.COMM_WORLD

def monitor(snes, it, norm):
    logging.info(f"Iteration {it}, residual {norm}")
    print(f"Iteration {it}, residual {norm}")
    return PETSc.SNES.ConvergedReason.ITERATING

# Set output directory
OUTDIR = os.path.join("../output")

if comm.rank == 0:
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

print("OUTDIR: ", OUTDIR)
X_COORD = 0
Y_COORD = 1
AIRY = 0
TRANSVERSE = 1
ABS_TOLLERANCE = 1e-11
REL_TOLLERANCE = 1e-11
SOL_TOLLERANCE = 1e-11

def hessian(u): return ufl.grad(ufl.grad(u))
def sigma(u):
    J = ufl.as_matrix([[0, -1], [1, 0]])
    return J.T*( hessian(u) ) * J
def mongeAmpere(u1, u2): return ufl.inner( sigma(u1), hessian(u2))

def ouward_unit_normal_x(x): return x[X_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)
def ouward_unit_normal_y(x): return x[Y_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)
def counterclock_tangent_x(x): return -x[Y_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)
def counterclock_tangent_y(x): return x[X_COORD] / np.sqrt(x[X_COORD]**2 + x[Y_COORD]**2)

# READ PARAMETERS FILE

with open('Data.yml') as f:
    parameters = yaml.load(f, Loader=yaml.FullLoader)


Eyoung =  parameters["model"]["E"]
nu = parameters["model"]["nu"]
R = parameters["geometry"]["radius"]
thickness = parameters["model"]["thickness"]
mesh_size = parameters["geometry"]["mesh_size"]
IP = parameters["model"]["alpha_penalty"]
disclination_points_list = [[d[0], d[1], 0.0] for d in parameters["positions_list"]]
disclination_power_list = parameters["frank_angle"]

# SAVE THE CURRENT SCRIPT INTO THE OUTPUT FOLDER
current_script = os.path.abspath(sys.argv[0]) # Get the current script file path
shutil.copy(current_script, os.path.join(OUTDIR, os.path.basename(current_script))) # Copy the script into the output folder

# REDIRECTING STD OUTPUT TO OUTPUT.TXT
#tee = Logging.Tee(os.path.join(OUTDIR, "stdout.txt"), os.path.join(OUTDIR, "stderr.txt"))
#sys.stdout = type("StdoutTee", (object,), {"write": tee.write_stdout, "flush": tee.flush})()
#sys.stderr = type("StderrTee", (object,), {"write": tee.write_stderr, "flush": tee.flush})()

# COMPUTE DIMENSIONLESS PARAMETERS
beta = R / thickness
rho_g = 1e4 # Density of the material times g-accelleration
N = 1 # N-times plate's own weight
p0 = rho_g * thickness
gamma = N * p0 / Eyoung
f0 = (beta**4) * gamma

print(10*"*")
print("Dimensionless parameters: ")
print("β := R/h = ", beta)
print("ɣ = ", gamma)
print("f := β^4 * ɣ = ", f0)
print(10*"*")

# LOAD MESH
parameters["geometry"]["geom_type"] = "circle"
model_rank = 0
tdim = 2

gmsh_model, tdim = mesh_circle_gmshapi( parameters["geometry"]["geom_type"], 1, mesh_size, tdim )
mesh, mts, fts = gmshio.model_to_mesh(gmsh_model, comm, model_rank, tdim)

h = CellDiameter(mesh)
n = FacetNormal(mesh)

# FUNCTION SPACES
fe_space = basix.ufl.element("P", str(mesh.ufl_cell()), parameters["model"]["order"])
mixedFE_space = dolfinx.fem.functionspace(mesh, basix.ufl.mixed_element([fe_space, fe_space]))

q = dolfinx.fem.Function(mixedFE_space)
v, w = ufl.split(q)
state = {"v": v, "w": w}

# SET DIRICHLET BC
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
bndry_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
dofs_v = dolfinx.fem.locate_dofs_topological(V=mixedFE_space.sub(AIRY), entity_dim=1, entities=bndry_facets)
dofs_w = dolfinx.fem.locate_dofs_topological(V=mixedFE_space.sub(TRANSVERSE), entity_dim=1, entities=bndry_facets)
bcs_v = dirichletbc( np.array(0, dtype=PETSc.ScalarType), dofs_v, mixedFE_space.sub(AIRY) )
bcs_w = dirichletbc( np.array(0, dtype=PETSc.ScalarType), dofs_w, mixedFE_space.sub(TRANSVERSE) )
_bcs = {AIRY: bcs_v, TRANSVERSE: bcs_w}
bcs = list(_bcs.values())


# DEFINE THE VARIATIONAL PROBLEM
model = A_NonlinearPlateFVK(mesh, parameters["model"])
energy = model.energy(state)[0]

# Volume load
def volume_load(x): return - f0 * (1 + 0*x[0]**2 + 0*x[1]**2)
f = dolfinx.fem.Function(mixedFE_space.sub(TRANSVERSE).collapse()[0])
f.interpolate(volume_load)
dx = ufl.Measure("dx")
W_ext = f * w * dx # CFe: external work
penalisation = model.penalisation(state)

# Disclinations
if mesh.comm.rank == 0:
    disclinations = []
    for dp in disclination_points_list: disclinations.append(np.array([dp], dtype=mesh.geometry.x.dtype))
else:
    for dp in disclination_points_list: disclinations.append(np.zeros((0, 3), dtype=mesh.geometry.x.dtype))

# Functional
L = energy - W_ext + penalisation

# DEFINE THE FEM (WEAK) PROBLEM
F = ufl.derivative(L, q, ufl.TestFunction(mixedFE_space))

MixedFE_space_v, MixedFE_space_v_to_MixedFE_space_dofs = mixedFE_space.sub(AIRY).collapse()

dp_list = [element*(beta**2) for element in disclination_power_list]

b = compute_disclination_loads(disclinations, dp_list, mixedFE_space, V_sub_to_V_dofs=MixedFE_space_v_to_MixedFE_space_dofs, V_sub=MixedFE_space_v)

solver_parameters = {
        "snes_type": "newtonls",  # Solver type: NGMRES (Nonlinear GMRES)
        "snes_max_it": 50,  # Maximum number of iterations
        "snes_rtol": REL_TOLLERANCE ,  # Relative tolerance for convergence
        "snes_atol": ABS_TOLLERANCE,  # Absolute tolerance for convergence
        "snes_stol": SOL_TOLLERANCE,  # Tolerance for the change in solution norm
        "snes_monitor": None,  # Function for monitoring convergence (optional)
        "snes_linesearch_type": "basic",  # Type of line search
    }

solver = SNESSolver(
    F_form=F,
    u=q,
    bcs=bcs,
    petsc_options=solver_parameters, #parameters.get("solvers").get("elasticity").get("snes"),
    prefix='plate_configuration',
    b0=b.vector,
    monitor=monitor,
)

solver.solve()

# DISPLAY COMPUTED ENERGY VALUES
energy_scale = Eyoung * (thickness**3) / (beta**2)
energy_components = {
    "bending": energy_scale*model.energy(state)[1],
    "membrane": energy_scale*model.energy(state)[2],
    "coupling": energy_scale*model.energy(state)[3],
    "external_work": p0* thickness * (R**2) *W_ext
    }

computed_energy_terms = {label: comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(energy_term)), op=MPI.SUM) for
                         label, energy_term in energy_components.items()}

print("Dimensional energy values: ", computed_energy_terms)


# -- STORE FIELDS ---
#chkpfilename = os.path.join(OUTDIR, "function_checkpoint.bp")
#adios.write_mesh(chkpfilename, mesh)
#adios.write_function(chkpfilename, q, time=0.1, name="vw")

# DEFINE AIRY AND TRANSVERSE DISPLACEMENT FOR POST-PROCESSING
v_pp, w_pp = q.split()
V_v, dofs_v = mixedFE_space.sub(AIRY).collapse()
V_w, dofs_w = mixedFE_space.sub(TRANSVERSE).collapse()

# COMPUTE STRESSES
sigma_xx = dolfinx.fem.Function(V_v)
sigma_xy = dolfinx.fem.Function(V_v)
sigma_yy = dolfinx.fem.Function(V_v)

sigma_xx_expr = dolfinx.fem.Expression( hessian(v_pp)[Y_COORD, Y_COORD], V_v.element.interpolation_points() )
sigma_xy_expr = dolfinx.fem.Expression( - hessian(v_pp)[X_COORD, Y_COORD], V_v.element.interpolation_points() )
sigma_yy_expr = dolfinx.fem.Expression( hessian(v_pp)[X_COORD, X_COORD], V_v.element.interpolation_points() )

sigma_xx.interpolate(sigma_xx_expr)
sigma_xy.interpolate(sigma_xy_expr)
sigma_yy.interpolate(sigma_yy_expr)

n_x = dolfinx.fem.Function(V_v)
n_y = dolfinx.fem.Function(V_v)
t_x = dolfinx.fem.Function(V_v)
t_y = dolfinx.fem.Function(V_v)
n_x.interpolate(ouward_unit_normal_x)
n_y.interpolate(ouward_unit_normal_y)
t_x.interpolate(counterclock_tangent_x)
t_y.interpolate(counterclock_tangent_y)

sigma_n_x = dolfinx.fem.Function(V_v)
sigma_n_y = dolfinx.fem.Function(V_v)
sigma_nx_expr = dolfinx.fem.Expression( sigma_xx*n_x + sigma_xy*n_y, V_v.element.interpolation_points() )
sigma_ny_expr = dolfinx.fem.Expression( sigma_xy*n_x + sigma_yy*n_y, V_v.element.interpolation_points() )

sigma_n_x.interpolate(sigma_nx_expr)
sigma_n_y.interpolate(sigma_ny_expr)
sigma_n = np.column_stack((sigma_n_x.x.array.real, sigma_n_y.x.array.real, np.zeros_like(sigma_n_x.x.array.real)))

sigma_nn = dolfinx.fem.Function(V_v)
sigma_nn_expr = dolfinx.fem.Expression( sigma_n_x*n_x + sigma_n_y*n_y , V_v.element.interpolation_points() )
sigma_nn.interpolate(sigma_nn_expr)

# COMPUTE MONGE-AMPERE BRACKET
ma_w = dolfinx.fem.Function(V_v)
ma_w_expr = dolfinx.fem.Expression( mongeAmpere(w_pp, w_pp), V_v.element.interpolation_points() )
ma_w.interpolate(ma_w_expr)



# PLOT MESH
plt.figure()
ax = plot_mesh(mesh)
fig = ax.get_figure()
fig.savefig(f"{OUTDIR}/mesh.png")

# PLOT WITH PYVISTA
IMG_WIDTH = 2200
IMG_HEIGHT = 1500
PNG_SCALE = 2.0
LINEWIDTH = 5
FONTSIZE = 30
CLINEWIDTH = 4
if pyvista.OFF_SCREEN: pyvista.start_xvfb(wait=0.1)
transparent = False
figsize = 800

scalar_bar_args = {
    "title_font_size": 35,  # Font size for the title
    "label_font_size": 35,  # Font size for the scalar bar labels <------------------------------------
    "vertical": False,  # Vertical orientation of the scalar bar
    "position_y": 0.025,  # Adjust vertical position
    "position_x": 0.15,  # Adjust horizontal position
    "height": 0.05,  # Height of the scalar bar
    "width": 0.7,  # Width of the scalar bar
    "n_labels": 4,  # Adjust number of ticks
    "fmt": "%.2e",  # Format for two decimal places
}

topology, cells, geometry = dolfinx.plot.vtk_mesh(MixedFE_space_v)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)



# PLOT FEM AND ANALYTICAL SOLUTIONS

# Airy, countour plot
subplotter = pyvista.Plotter(shape=(1, 1))
grid.point_data["v"] = v_pp.x.array.real[dofs_v]
grid.set_active_scalars("v")
subplotter.subplot(0, 0)
scalar_bar_args["title"] = "v"
grid.set_active_scalars("v")
subplotter.subplot(0, 0)
subplotter.add_text("v", position="upper_edge", font_size=14, color="black")
scalar_bar_args["title"] = "v"
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1/max(np.abs(v_pp.x.array.real[dofs_v] )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="viridis")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.export_html(f"{OUTDIR}/Airy.html")

# Transverse displacement, countour
subplotter = pyvista.Plotter(shape=(1, 1)) #
grid.point_data["w"] = w_pp.x.array.real[dofs_w]
grid.set_active_scalars("w")
scalar_bar_args["title"] = "w"

# Transverse displacement, 3D view
grid.set_active_scalars("w")
subplotter.subplot(0, 0)
subplotter.add_text("w", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh(
   grid.warp_by_scalar( scale_factor = 1/max(np.abs(w_pp.x.array.real[dofs_w] )) ),
   show_edges=False,
   edge_color="white",
   show_scalar_bar=True,
   scalar_bar_args=scalar_bar_args,
   cmap="plasma")
#subplotter.show_grid(xlabel="X-axis", ylabel="Y-axis", zlabel="Height (u)")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.export_html(f"{OUTDIR}/transverseDisplacement.html")


# PLOT SIGMA NN
subplotter = pyvista.Plotter(shape=(1, 1))
grid["sigma_nn"] = sigma_nn.x.array.real
grid.set_active_scalars("sigma_nn")
subplotter.subplot(0, 0)
scalar_bar_args["title"] = "sigma_rr"
subplotter.add_text("sigma_rr", position="upper_edge", font_size=14, color="black")
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(sigma_nn.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/visualization_sigma_rr.png", scale = PNG_SCALE)
subplotter.export_html(f"{OUTDIR}/visualization_sigma_rr.html")


# PLOT MONGE-AMPERE W
subplotter = pyvista.Plotter(shape=(1, 1))
grid["ma_w"] = ma_w.x.array.real
grid.set_active_scalars("ma_w")
subplotter.subplot(0, 0)
subplotter.add_text("ma_w", position="upper_edge", font_size=14, color="black")
scalar_bar_args["title"] = "[w, w]"
subplotter.add_mesh( grid.warp_by_scalar( scale_factor = 1 / max(np.abs(ma_w.x.array.real )) ), show_edges=False, edge_color="white", show_scalar_bar=True, scalar_bar_args=scalar_bar_args, cmap="coolwarm")
subplotter.window_size = (IMG_WIDTH, IMG_HEIGHT)
subplotter.screenshot(f"{OUTDIR}/viz_[w,w].png", scale = PNG_SCALE)
subplotter.export_html(f"{OUTDIR}/viz_[w,w].html")
