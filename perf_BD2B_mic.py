# This example implements 1st-level homogenization of Biot-Darcy-Brinkman model of flow in deformable
# double porous media.
# The mathematical model is described in:
#
#ROHAN E., TURJANICOVA J., LUKES V.
#Multiscale modelling and simulations of tissue perfusion using the Biot-Darcy-Brinkman model.
# Computers & Structures, 2020,
#
# Run calculation of homogenized coefficients:
#
#   ./homogen.py example_perfusion_BD2B/perf_BD2B_mic.py
#
# The results are stored in `example_perfusion_BD2B/results/micro` directory.
#

import numpy as nm
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.homogenization.utils import define_box_regions
from sfepy.discrete.fem.mesh import Mesh
import sfepy.discrete.fem.periodic as per
import sfepy.homogenization.coefs_base as cb
import os.path as osp

data_dir = 'example_perfusion_BD2B'


#Definition of periodic boundary conditions
def get_periodic_bc(var_tab, dim=3, dim_tab=None):
    if dim_tab is None:
        dim_tab = {'x': ['left', 'right'],
                   'z': ['bottom', 'top'],
                   'y': ['near', 'far']}

    periodic = {}
    epbcs = {}

    for ivar, reg in var_tab:
        periodic['per_%s' % ivar] = pers = []
        for idim in 'xyz'[0:dim]:
            key = 'per_%s_%s' % (ivar, idim)
            regs = ['%s_%s' % (reg, ii) for ii in dim_tab[idim]]
            epbcs[key] = (regs, {'%s.all' % ivar: '%s.all' % ivar},
                          'match_%s_plane' % idim)
            pers.append(key)

    return epbcs, periodic


# define homogenized coefficients and subproblems for correctors
def define(filename_mesh=None):
    eps0 = 0.01  # given scale parameter 

    if filename_mesh is None:
        filename_mesh = osp.join(data_dir, 'micro_perf_puc.vtk')

    mesh = Mesh.from_file(filename_mesh)

    dim = 3
    sym = (dim + 1) * dim // 2
    sym_eye = 'nm.array([1,1,0])' if dim == 2 else 'nm.array([1,1,1,0,0,0])'

    bbox = mesh.get_bounding_box()
    regions = define_box_regions(mesh.dim, bbox[0], bbox[1], eps=1e-3)

    regions.update({
        'Y': 'all',
        'Gamma_Y': ('vertices of surface', 'facet'),
        # solid matrix
        'Ys': 'cells of group 1',
        'Ys_left': ('r.Ys *v r.Left', 'vertex'),
        'Ys_right': ('r.Ys *v r.Right', 'vertex'),
        'Ys_bottom': ('r.Ys *v r.Bottom', 'vertex'),
        'Ys_top': ('r.Ys *v r.Top', 'vertex'),
        'Gamma_Ysf': ('r.Ys *v r.Yf', 'facet', 'Ys'),
        # channel
        'Yf': 'cells of group 2',
        'Yf0': ('r.Yf -v r.Gamma_Yfs', 'vertex'),
        'Yf_left': ('r.Yf0 *v r.Left', 'vertex'),
        'Yf_right': ('r.Yf0 *v r.Right', 'vertex'),
        'Yf_bottom': ('r.Yf0 *v r.Bottom', 'vertex'),
        'Yf_top': ('r.Yf0 *v r.Top', 'vertex'),
        'Gamma_Yfs': ('r.Ys *v r.Yf', 'facet', 'Yf'),
    })

    if dim == 3:
        regions.update({
                'Ys_far': ('r.Ys *v r.Far', 'vertex'),
                'Ys_near': ('r.Ys *v r.Near', 'vertex'),
                'Yf_far': ('r.Yf0 *v r.Far', 'vertex'),
                'Yf_near': ('r.Yf0 *v r.Near', 'vertex'),
        })


    fields = {
        'volume': ('real', 'scalar', 'Y', 1),
        'displacement': ('real', 'vector', 'Ys', 1),
        'pressure': ('real', 'scalar', 'Yf', 1),
        'velocity': ('real', 'vector', 'Yf', 2),
    }

    variables = {
        # displacement
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'Pi_u': ('parameter field', 'displacement', 'u'),
        'U1': ('parameter field', 'displacement', '(set-to-None)'),
        'U2': ('parameter field', 'displacement', '(set-to-None)'),
        # velocity
        'w': ('unknown field', 'velocity'),
        'z': ('test field', 'velocity', 'w'),
        'Pi_w': ('parameter field', 'velocity', 'w'),
        'W1': ('parameter field', 'velocity', '(set-to-None)'),
        'W2': ('parameter field', 'velocity', '(set-to-None)'),
        # pressure
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        # volume
        'volume': ('parameter field', 'volume', '(set-to-None)'),
    }

    functions = {
        'match_x_plane': (per.match_x_plane,),
        'match_y_plane': (per.match_y_plane,),
        'match_z_plane': (per.match_z_plane,),
    }

    materials = {
        'matrix': ({'D': stiffness_from_youngpoisson(dim, 1e3, 0.49)},),#Soft tissue
        'fluid': ({
                   'eta_p': 3.6e-3 / eps0**2,#Rescaled blood viscosity
                   'aux_compress': 1e-18},),#Auxillary compressibility
    }

    ebcs = {
        'fixed_u': ('Corners', {'u.all': 0.0}),
        'fixed_w': ('Gamma_Yfs', {'w.all': 0.0}),
    }

    epbcs, periodic = get_periodic_bc([('u', 'Ys'), ('p', 'Yf'), ('w', 'Yf')])

    integrals = {
        'i': 4,
    }

    options = {
        'coefs': 'coefs',
        'coefs_filename': 'coefs_micro',
        'requirements': 'requirements',
        'volume': {
            'variables': ['u', 'p'],
            'expression': 'd_volume.i.Ys(u) + d_volume.i.Yf(p)',
        },
        'output_dir': data_dir+'/results/micro',
        'ls': 'ls',
        'file_per_var': True,
        'absolute_mesh_path': True,
        'multiprocessing': True,
        'output_prefix': 'micro:',

    }
    #Definition of used solvers
    solvers = {
        'ls': ('ls.mumps', {}),

        'ns_em9': ('nls.newton', {
                    'i_max': 1,
                    'eps_a': 1e-9,
                    'eps_r': 1e-3,
                    'problem': 'nonlinear'}),
        'ns_em12': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-12,
            'eps_r': 1e-3,
            'problem': 'nonlinear'}),
    }
    #Definition of homogenized coefficients, see (22) and (23)
    coefs = {
        #Elasticity coefficient
        'A': {
            'requires': ['pis_u', 'corrs_omega_ij'],
            'expression': 'dw_lin_elastic.i.Ys(matrix.D, U1, U2)',
            'set_variables': [('U1', ('corrs_omega_ij', 'pis_u'), 'u'),
                              ('U2', ('corrs_omega_ij', 'pis_u'), 'u')],
            'class': cb.CoefSymSym,
        },
        #Biot coefficient
        'hat_B': {
            'status': 'auxiliary',
            'requires': ['corrs_omega_ij'],
            'expression': '- ev_div.i.Ys(U1)',
            'set_variables': [('U1', 'corrs_omega_ij', 'u')],
            'class': cb.CoefSym,
        },
        'B': {
            'requires': ['c.phi_f', 'c.hat_B'],
            'expression': 'c.hat_B + c.phi_f * %s' % sym_eye,
            'class': cb.CoefEval,
        },

        'M': {
            'requires': ['corrs_omega_p'],
            'expression': 'dw_lin_elastic.i.Ys(matrix.D, U1, U2)',
            'set_variables': [('U1', 'corrs_omega_p', 'u'),
                              ('U2', 'corrs_omega_p', 'u')],
            'class': cb.CoefOne,
        },
        #Permeability
        'K': {
            'requires': ['corrs_psi_i'],
            'expression': 'dw_div_grad.i.Yf(W1, W2)',  # !!!
            'set_variables': [('W1', 'corrs_psi_i', 'w'),
                              ('W2', 'corrs_psi_i', 'w')],
            'class': cb.CoefDimDim,
        },
        #Volume fraction of fluid part
        'phi_f': {
            'requires': ['c.vol'],
            'expression': 'c.vol["fraction_Yf"]',
            'class': cb.CoefEval,
        },
        #Coefficient for storing viscosity
        'eta_p': {
            'expression': '%e' % materials['fluid'][0]['eta_p'],
            'class': cb.CoefEval,
        },
        #Volume fractions
        'vol': {
            'regions': ['Ys', 'Yf'],
            'expression': 'd_volume.i.%s(volume)',
            'class': cb.VolumeFractions,
        },
        #Surface volume fractions
        'surf_vol': {
            'regions': ['Ys', 'Yf'],
            'expression': 'd_surface.i.%s(volume)',
            'class': cb.VolumeFractions,
        },
        'filenames': {},
    }
    #Definition of microscopic corrector problems
    requirements = {
        #Definition of \Pi^{ij}_k
        'pis_u': {
            'variables': ['u'],
            'class': cb.ShapeDimDim,
        },
        #Correcotr like class returning ones
        'pis_w': {
            'variables': ['w'],
            'class': cb.OnesDim,
        },
        #Corrector problem related to elasticity, see (17)
        'corrs_omega_ij': {
            'requires': ['pis_u'],
            'ebcs': ['fixed_u'],
            'epbcs': periodic['per_u'],
            'is_linear': True,
            'equations': {
                'balance_of_forces':
                    """dw_lin_elastic.i.Ys(matrix.D, v, u)
                   = - dw_lin_elastic.i.Ys(matrix.D, v, Pi_u)"""
            },
            'set_variables': [('Pi_u', 'pis_u', 'u')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_omega_ij',
            'dump_variables': ['u'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em9'},
        },
        # Corrector problem related to elasticity, see (18)
        'corrs_omega_p': {
            'requires': [],
            'ebcs': ['fixed_u'],
            'epbcs': periodic['per_u'],
            'equations': {
                'balance_of_forces':
                    """dw_lin_elastic.i.Ys(matrix.D, v, u)
                     = -dw_surface_ltr.i.Gamma_Ysf(v)"""
            },
            'class': cb.CorrOne,
            'save_name': 'corrs_omega_p',
            'dump_variables': ['u'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em3'},
        },
        #Corrector problem related to velocity, see (19)
        'corrs_psi_i': {
            'requires': ['pis_w'],
            'ebcs': ['fixed_w'],
             'epbcs': periodic['per_w'] + periodic['per_p'],
            'is_linear': True,
            'equations': {
                'balance_of_forces':  # !!!
                    """dw_div_grad.i.Yf(fluid.eta_p,z, w)
                     - dw_stokes.i.Yf(z, p)
                     = dw_volume_dot.i.Yf(z, Pi_w)""",
                'incompressibility':
                    """dw_stokes.i.Yf(w, q)
                    + dw_volume_dot.i.Yf(fluid.aux_compress, q, p)
                     = 0""",#
            },
            'set_variables': [('Pi_w', 'pis_w', 'w')],
            'class': cb.CorrDim,
            'save_name': 'corrs_psi_i',
            'dump_variables': ['w', 'p'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em12'},
        },
    }

    return locals()
