# This example implements 2nd-level homogenization of Biot-Darcy-Brinkman model of flow in deformable
# double porous media.
# The mathematical model is described in:
#
#ROHAN E., TURJANICOVA J., LUKES V.
#Multiscale modelling and simulations of tissue perfusion using the Biot-Darcy-Brinkman model.
# Computers & Structures, 2020,
#
# Run calculation of homogenized coefficients:
#
#   ./homogen.py example_perfusion_BD2B/perf_BD2B_mes.py
#
# The results are stored in `example_perfusion_BD2B/results/meso` directory.
#

import numpy as nm
from sfepy import data_dir

import os.path as osp
from sfepy.discrete.fem.mesh import Mesh
import sfepy.discrete.fem.periodic as per
from sfepy.mechanics.tensors import dim2sym
from sfepy.homogenization.utils import define_box_regions
import sfepy.homogenization.coefs_base as cb
from sfepy.homogenization.micmac import get_homog_coefs_linear


data_dir = 'example_perfusion_BD2B'

def coefs2qp(coefs, nqp):
    out = {}
    for k, v in coefs.items():
        if type(v) not in [nm.ndarray, float]:
            continue

        if type(v) is nm.ndarray:
            if len(v.shape) >= 3:
                out[k] = v

        out[k] = nm.tile(v, (nqp, 1, 1))

    return out

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
            if key in ["per_pc1_y","per_pc1_z","per_w1_y","per_w1_z","per_pc2_y","per_pc2_z","per_w2_y","per_w2_z"]:
                continue
            regs = ['%s_%s' % (reg, ii) for ii in dim_tab[idim]]
            epbcs[key] = (regs, {'%s.all' % ivar: '%s.all' % ivar},
                          'match_%s_plane' % idim)
            pers.append(key)

    return epbcs, periodic



# get homogenized coefficients, recalculate them if necessary
def get_homog(coors, mode, pb,micro_filename, **kwargs):
    if not (mode == 'qp'):
        return
    nqp = coors.shape[0]
    coefs_filename = 'coefs_micro'
    coefs_filename = osp.join(pb.conf.options.get('output_dir', '.'),
                              coefs_filename) + '.h5'
    coefs = get_homog_coefs_linear(0, 0, None,
                                   micro_filename=micro_filename,coefs_filename = coefs_filename    )
    coefs['B'] = coefs['B'][:, nm.newaxis]
    for k in coefs.keys():
        v = coefs[k]
        if type(v) is nm.ndarray:
            if len(v.shape) == 0:
                coefs[k] = v.reshape((1, 1))
            elif len(v.shape) == 1:
                coefs[k] = v[:, nm.newaxis]
        elif isinstance(v, float):
            coefs[k] = nm.array([[v]])

    out = coefs2qp(coefs, nqp)

    return out

def define(filename_mesh=None):

    eta = 3.6e-3
    if filename_mesh is None:
        filename_mesh = osp.join(data_dir, 'meso_perf_2ch_puc.vtk')
        # filename_mesh = osp.join(data_dir, 'perf_mic_2chbx.vtk')

    mesh = Mesh.from_file(filename_mesh)

    poroela_micro_file = osp.join(data_dir, 'perf_BD2B_mic.py')
    dim = 3
    sym = (dim + 1) * dim // 2
    sym_eye = 'nm.array([1,1,0])' if dim == 2 else 'nm.array([1,1,1,0,0,0])'

    bbox = mesh.get_bounding_box()
    regions = define_box_regions(mesh.dim, bbox[0], bbox[1], eps=1e-3)

    regions.update({
        'Z': 'all',
        'Gamma_Z': ('vertices of surface', 'facet'),
        # matrix
        'Zm': 'cells of group 3',
        'Zm_left': ('r.Zm *v r.Left', 'vertex'),
        'Zm_right': ('r.Zm *v r.Right', 'vertex'),
        'Zm_bottom': ('r.Zm *v r.Bottom', 'vertex'),
        'Zm_top': ('r.Zm *v r.Top', 'vertex'),
        "Gamma_Zm": ('(r.Zm *v r.Zc1) +v (r.Zm *v r.Zc2)', 'facet', 'Zm'),
        'Gamma_Zm1': ('r.Zm *v r.Zc1', 'facet', 'Zm'),
        'Gamma_Zm2': ('r.Zm *v r.Zc2', 'facet', 'Zm'),
        # first canal
        'Zc1': 'cells of group 1',
        'Zc01': ('r.Zc1 -v r.Gamma_Zc1', 'vertex'),
        'Zc1_left': ('r.Zc01 *v r.Left', 'vertex'),
        'Zc1_right': ('r.Zc01 *v r.Right', 'vertex'),
        # 'Zc1_bottom': ('r.Zc01 *v r.Bottom', 'vertex'),
        # 'Zc1_top': ('r.Zc01 *v r.Top', 'vertex'),
        'Gamma_Zc1': ('r.Zm *v r.Zc1', 'facet', 'Zc1'),
        'Center_c1': ('vertex 973', 'vertex'),  # canal center
        # 'Center_c1': ('vertex 1200', 'vertex'),  # canal center

        # second canal
        'Zc2': 'cells of group 2',
        'Zc02': ('r.Zc2 -v r.Gamma_Zc2', 'vertex'),
        'Zc2_left': ('r.Zc02 *v r.Left', 'vertex'),
        'Zc2_right': ('r.Zc02 *v r.Right', 'vertex'),
        # 'Zc2_bottom': ('r.Zc02 *v r.Bottom', 'vertex'),
        # 'Zc2_top': ('r.Zc02 *v r.Top', 'vertex'),
        'Gamma_Zc2': ('r.Zm *v r.Zc2', 'facet', 'Zc2'),
        'Center_c2': ('vertex 487', 'vertex'),  # canal center
        # 'Center_c2': ('vertex 1254', 'vertex'),  # canal center

    })

    if dim == 3:
        regions.update({
            'Zm_far': ('r.Zm *v r.Far', 'vertex'),
            'Zm_near': ('r.Zm *v r.Near', 'vertex'),
    #         'Zc1_far': ('r.Zc01 *v r.Far', 'vertex'),
    #         'Zc1_near': ('r.Zc01 *v r.Near', 'vertex'),
    #         'Zc2_far': ('r.Zc02 *v r.Far', 'vertex'),
    #         'Zc2_near': ('r.Zc02 *v r.Near', 'vertex'),
        })


    fields = {
        'one': ('real', 'scalar', 'Z', 1),
        'displacement': ('real', 'vector', 'Zm', 1),
        'pressure_m': ('real', 'scalar', 'Zm', 1),
        'pressure_c1': ('real', 'scalar', 'Zc1', 1),
        'displacement_c1': ('real', 'vector', 'Zc1', 1),
        'velocity1': ('real', 'vector', 'Zc1', 2),
        'pressure_c2': ('real', 'scalar', 'Zc2', 1),
        'displacement_c2': ('real', 'vector', 'Zc2', 1),
        'velocity2': ('real', 'vector', 'Zc2', 2),
    }

    variables = {
        # displacement
        'u': ('unknown field', 'displacement', 0),
        'v': ('test field', 'displacement', 'u'),
        'Pi_u': ('parameter field', 'displacement', 'u'),
        'U1': ('parameter field', 'displacement', '(set-to-None)'),
        'U2': ('parameter field', 'displacement', '(set-to-None)'),

        'uc1': ('unknown field', 'displacement_c1', 6),
        'vc1': ('test field', 'displacement_c1', 'uc1'),
        'uc2': ('unknown field', 'displacement_c2', 7),
        'vc2': ('test field', 'displacement_c2', 'uc2'),
        # velocity in canal 1
        'w1': ('unknown field', 'velocity1', 1),
        'z1': ('test field', 'velocity1', 'w1'),
        'Pi_w1': ('parameter field', 'velocity1', 'w1'),
        'par1_w1': ('parameter field', 'velocity1', '(set-to-None)'),
        'par2_w1': ('parameter field', 'velocity1', '(set-to-None)'),
        # velocity in canal 2
        'w2': ('unknown field', 'velocity2', 2),
        'z2': ('test field', 'velocity2', 'w2'),
        'Pi_w2': ('parameter field', 'velocity2', 'w2'),
        'par1_w2': ('parameter field', 'velocity2', '(set-to-None)'),
        'par2_w2': ('parameter field', 'velocity2', '(set-to-None)'),
        # pressure
        'pm': ('unknown field', 'pressure_m', 3),
        'qm': ('test field', 'pressure_m', 'pm'),
        'Pm1': ('parameter field', 'pressure_m', '(set-to-None)'),
        'Pm2': ('parameter field', 'pressure_m', '(set-to-None)'),
        'Pi_pm': ('parameter field', 'pressure_m', 'pm'),
        'pc1': ('unknown field', 'pressure_c1', 4),
        'qc1': ('test field', 'pressure_c1', 'pc1'),
        'par1_pc1': ('parameter field', 'pressure_c1', '(set-to-None)'),
        'par2_pc1': ('parameter field', 'pressure_c1', '(set-to-None)'),
        'pc2': ('unknown field', 'pressure_c2', 5),
        'qc2': ('test field', 'pressure_c2', 'pc2'),
        'par1_pc2': ('parameter field', 'pressure_c2', '(set-to-None)'),
        'par2_pc2': ('parameter field', 'pressure_c2', '(set-to-None)'),
        # one
        'one': ('parameter field', 'one', '(set-to-None)'),
    }

    functions = {
        'match_x_plane': (per.match_x_plane,),
        'match_y_plane': (per.match_y_plane,),
        'match_z_plane': (per.match_z_plane,),
        'get_homog': (lambda ts, coors, mode=None, problem=None, **kwargs:\
            get_homog(coors, mode, problem, poroela_micro_file, **kwargs),),
    }
    materials = {

        'hmatrix': 'get_homog',
        'fluid': ({
                    'eta_c': eta* nm.eye(dim2sym(dim)),
                  },),
        'mat': ({
                    'k1': nm.array([[1, 0, 0]]).T,
                    'k2': nm.array([[0, 1, 0]]).T,
                    'k3': nm.array([[0, 0, 1]]).T,
                },),
    }

    ebcs = {
        'fixed_u': ('Corners', {'um.all': 0.0}),
        'fixed_pm': ('Corners', {'p.0': 0.0}),
        'fixed_w1': ('Center_c1', {'w1.all': 0.0}),
        'fixed_w2': ('Center_c2', {'w2.all': 0.0}),

    }

    epbcs, periodic = get_periodic_bc([('u', 'Zm'), ('pm', 'Zm'),
                                        ('pc1', 'Zc1'), ('w1', 'Zc1'),
                                        ('pc2', 'Zc2'), ('w2', 'Zc2')])

    all_periodic1 = periodic['per_w1'] + periodic['per_pc1']
    all_periodic2 = periodic['per_w2'] + periodic['per_pc2']

    integrals = {
        'i': 4,
    }

    options = {
        'coefs': 'coefs',
        'coefs_filename': 'coefs_meso',

        'requirements': 'requirements',
        'volume': {
            'variables': ['u', 'pc1', 'pc2'],
            'expression': 'd_volume.i.Zm(u) + d_volume.i.Zc1(pc1)+d_volume.i.Zc2(pc2)',
        },
        'output_dir': data_dir + '/results/meso',
        'file_per_var': True,
        'save_format': 'vtk',  # Global setting.
        'dump_format': 'h5',  # Global setting.
        'absolute_mesh_path': True,
        'multiprocessing': False,
        'ls': 'ls',
        'nls': 'ns_m15',
        'output_prefix': 'Meso:',

    }

    solvers = {
        'ls': ('ls.mumps', {}),
        'ls_s1': ('ls.schur_mumps',
              {'schur_variables': ['pc1'],
                'fallback': 'ls'}),
        'ls_s2': ('ls.schur_mumps',
                  {'schur_variables': ['pc2'],
                   'fallback': 'ls'}),
        'ns': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-14,
            'eps_r': 1e-3,
            'problem': 'nonlinear'}),
        'ns_em15': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-15,
            'eps_r': 1e-3,
            'problem': 'nonlinear'}),
        'ns_em12': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-12,
            'eps_r': 1e-3,
            'problem': 'nonlinear'}),
        'ns_em9': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-9,
            'eps_r': 1e-3,
            'problem': 'nonlinear'}),
        'ns_em6': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-4,
            'eps_r': 1e-3,
            'problem': 'nonlinear'}),
    }

    #Definition of homogenized coefficients, see (33)-(35)
    coefs = {
        'A': {
            'requires': ['pis_u', 'corrs_omega_ij'],
            'expression': 'dw_lin_elastic.i.Zm(hmatrix.A, U1, U2)',
            'set_variables': [('U1', ('corrs_omega_ij', 'pis_u'), 'u'),
                              ('U2', ('corrs_omega_ij', 'pis_u'), 'u')],
            'class': cb.CoefSymSym,
        },
        'B_aux1': {
            'status': 'auxiliary',
            'requires': ['corrs_omega_ij'],
            'expression': '- dw_surface_ltr.i.Gamma_Zm(U1)',  # !!! -
            'set_variables': [('U1', 'corrs_omega_ij', 'u')],
            'class': cb.CoefSym,
        },
        'B_aux2': {
            'status': 'auxiliary',
            'requires': ['corrs_omega_ij', 'pis_u', 'corr_one'],
            'expression': 'dw_biot.i.Zm(hmatrix.B, U1, one)',
            'set_variables': [('U1', ('corrs_omega_ij', 'pis_u'), 'u'),
                              ('one', 'corr_one', 'one')],
            'class': cb.CoefSym,
        },
        'B': {
            'requires': ['c.B_aux1', 'c.B_aux2', 'c.vol_c'],
            'expression': 'c.B_aux1 + c.B_aux2 + c.vol_c* %s' % sym_eye,
            'class': cb.CoefEval,
        },
        'H11': {
            'requires': ['corrs_phi_k_1'],
            'expression': 'dw_diffusion.i.Zm(hmatrix.K, Pm1, Pm2)',
            'set_variables': [('Pm1', 'corrs_phi_k_1', 'pm'),
                              ('Pm2', 'corrs_phi_k_1', 'pm')],
            'class': cb.CoefDimDim,
        },
        'H12': {
            'requires': ['corrs_phi_k_1','corrs_phi_k_2'],
            'expression': 'dw_diffusion.i.Zm(hmatrix.K, Pm1, Pm2)',
            'set_variables': [('Pm1', 'corrs_phi_k_1', 'pm'),
                              ('Pm2', 'corrs_phi_k_2', 'pm')],
            'class': cb.CoefDimDim,
        },
        'H21': {
            'requires': ['corrs_phi_k_1', 'corrs_phi_k_2'],
            'expression': 'dw_diffusion.i.Zm(hmatrix.K, Pm1, Pm2)',
            'set_variables': [('Pm1', 'corrs_phi_k_2', 'pm'),
                              ('Pm2', 'corrs_phi_k_1', 'pm')],
            'class': cb.CoefDimDim,
        },
        'H22': {
            'requires': [ 'corrs_phi_k_2'],
            'expression': 'dw_diffusion.i.Zm(hmatrix.K, Pm1, Pm2)',
            'set_variables': [('Pm1', 'corrs_phi_k_2', 'pm'),
                              ('Pm2', 'corrs_phi_k_2', 'pm')],
            'class': cb.CoefDimDim,
        },
        'K': {
             'requires': ['corrs_pi_k', 'pis_pm'],
             'expression': 'dw_diffusion.i.Zm(hmatrix.K, Pm1, Pm2)',
             'set_variables': [('Pm1', ('corrs_pi_k', 'pis_pm'), 'pm'),
                               ('Pm2', ('corrs_pi_k', 'pis_pm'), 'pm')],
             'class': cb.CoefDimDim,
         },
        'Q1': {
             'requires': ['corrs_phi_k_1', 'pis_pm'],
             'expression': 'dw_diffusion.i.Zm(hmatrix.K, Pm1, Pm2)',
             'set_variables': [('Pm1', 'pis_pm', 'pm'),
                               ('Pm2', 'corrs_phi_k_1', 'pm')],
             'class': cb.CoefDimDim,
         },
        'P1': {
            'requires': ['c.Q1', 'c.vol'],
            'expression': 'c.vol["fraction_Zc1"] * nm.eye(%d) - c.Q1' % dim,
            'class': cb.CoefEval,
        },
        'P1T': {
            'requires': ['c.P1'],
            'expression': 'c.P1.T',
            'class': cb.CoefEval,
        },
        'Q2': {
            'requires': ['corrs_phi_k_2', 'pis_pm'],
            'expression': 'dw_diffusion.i.Zm(hmatrix.K, Pm1, Pm2)',
            'set_variables': [('Pm1', 'pis_pm', 'pm'),
                              ('Pm2', 'corrs_phi_k_2', 'pm')],
            'class': cb.CoefDimDim,
        },
        'P2': {
            'requires': ['c.Q2', 'c.vol'],
            'expression': 'c.vol["fraction_Zc2"] * nm.eye(%d) - c.Q2' % dim,
            'class': cb.CoefEval,
        },
        'P2T': {
            'requires': ['c.P2'],
            'expression': 'c.P2.T',
            'class': cb.CoefEval,
        },
        'M_aux1': {
            'status': 'auxiliary',
            'requires': [],
            'expression': 'ev_volume_integrate_mat.i.Zm(hmatrix.M, one)',
            'set_variables': [],
            'class': cb.CoefOne,
        },
        'M_aux2': {
            'status': 'auxiliary',
            'requires': ['corrs_omega_p', 'corr_one'],
            'expression': 'dw_biot.i.Zm(hmatrix.B, U1, one)',
            'set_variables': [('U1', 'corrs_omega_p', 'u'),
                              ('one', 'corr_one', 'one')],
            'class': cb.CoefOne,
        },
        'M_aux3': {
            'status': 'auxiliary',
            'requires': ['corrs_omega_p'],
            'expression': ' dw_surface_ltr.i.Gamma_Zm(U1)',
            'set_variables': [('U1', 'corrs_omega_p', 'u')],
            'class': cb.CoefOne,
        },
        'M': {
            'requires': ['c.M_aux1', 'c.M_aux2', 'c.M_aux3'],
            'expression': 'c.M_aux1 + c.M_aux2 -c.M_aux3 ',
            'class': cb.CoefEval,
        },
        'S1': {
            'requires': ['corrs_psi_ij_1', 'pis_w1'],
            'expression': 'dw_lin_elastic.i.Zc1(fluid.eta_c, par1_w1, par2_w1)',
            'set_variables': [('par1_w1', ('corrs_psi_ij_1', 'pis_w1'), 'w1'),
                              ('par2_w1', ('corrs_psi_ij_1', 'pis_w1'), 'w1')],
            'class': cb.CoefSymSym,
        },
        'S2': {
            'requires': ['corrs_psi_ij_2', 'pis_w2'],
            'expression': 'dw_lin_elastic.i.Zc2(fluid.eta_c, par1_w2, par2_w2)',
            'set_variables': [('par1_w2', ('corrs_psi_ij_2', 'pis_w2'), 'w2'),
                              ('par2_w2', ('corrs_psi_ij_2', 'pis_w2'), 'w2')],
            'class': cb.CoefSymSym,
        },
        'vol': {
            'regions': ['Zm', 'Zc1', 'Zc2'],
            'expression': 'd_volume.i.%s(one)',
            'class': cb.VolumeFractions,
        },
        'surf_vol': {
            'regions': ['Zm', 'Zc1', 'Zc2'],
            'expression': 'd_surface.i.%s(one)',
            'class': cb.VolumeFractions,
        },

        'surf_c': {
            'requires': ['c.surf_vol'],
            'expression': 'c.surf_vol["fraction_Zc1"]+c.surf_vol["fraction_Zc2"]',
            'class': cb.CoefEval,
        },
        'vol_c': {
            'requires': ['c.vol'],
            'expression': 'c.vol["fraction_Zc1"]+c.vol["fraction_Zc2"]',
            'class': cb.CoefEval,
        },
        'filenames': {},

    }
    #Definition of mesoscopic corrector problems
    requirements = {
        'corr_one': {
            'variable': 'one',
            'expression':
                "nm.ones((problem.fields['one'].n_vertex_dof, 1), dtype=nm.float64)",
            'class': cb.CorrEval,
        },

        'pis_u': {
            'variables': ['u'],
            'class': cb.ShapeDimDim,
            'save_name': 'corrs_pis_u',
            'dump_variables': ['u'],

        },
        'pis_pm': {
            'variables': ['pm'],
            'class': cb.ShapeDim,
        },

        'pis_w1': {
            'variables': ['w1'],
            'class': cb.ShapeDimDim,
        },
        'pis_w2': {
            'variables': ['w2'],
            'class': cb.ShapeDimDim,
        },
        # Corrector problem, see (31)_1
        'corrs_omega_ij': {
            'requires': ['pis_u'],
            'ebcs': ['fixed_u'],
            'epbcs': periodic['per_u'],
            'is_linear': True,
            'equations': {
                'balance_of_forces':
                    """dw_lin_elastic.i.Zm(hmatrix.A, v, u)
                   = - dw_lin_elastic.i.Zm(hmatrix.A, v, Pi_u)"""
            },
            'set_variables': [('Pi_u', 'pis_u', 'u')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_omega_ij',
            'dump_variables': ['u'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em6'},

            'is_linear': True,
        },
        # Corrector problem, see (31)_2
        'corrs_omega_p': {
            'requires': ['corr_one'],
            'ebcs': ['fixed_u'],
            'epbcs': periodic['per_u'],
            'equations': {
                'balance_of_forces':
                    """dw_lin_elastic.i.Zm(hmatrix.A, v, u)
                     = dw_biot.i.Zm(hmatrix.B, v, one)
                     - dw_surface_ltr.i.Gamma_Zm(v)""",
            },
            'set_variables': [('one', 'corr_one', 'one')],
            'class': cb.CorrOne,
            'save_name': 'corrs_omega_p',
            'dump_variables': ['u'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em9'},

        },
        # Corrector problem, see (31)_3
        'corrs_pi_k': {
            'requires': ['pis_pm'],
            'ebcs': [],  # ['fixed_pm'],
            'epbcs': periodic['per_pm'],
            'is_linear': True,
            'equations': {
                'eq':
                    """dw_diffusion.i.Zm(hmatrix.K, qm, pm)
                   = - dw_diffusion.i.Zm(hmatrix.K, qm, Pi_pm)""",
            },
            'set_variables': [('Pi_pm', 'pis_pm', 'pm')],
            'class': cb.CorrDim,
            'save_name': 'corrs_pi_k',
            'dump_variables': ['pm'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em12'},
        },
        # Corrector problem, see (31)_4
        'corrs_phi_k_1': {
            'requires': [],
            'ebcs': [],
            'epbcs': periodic['per_pm'],
            'equations': {
                'eq':
                    """dw_diffusion.i.Zm(hmatrix.K, qm, pm)
                   = - dw_surface_ndot.i.Gamma_Zm1(mat.k%d, qm)""",
            },
            'class': cb.CorrEqPar,
            'eq_pars': [(ii + 1) for ii in range(dim)],
            'save_name': 'corrs_phi_k_1',
            'dump_variables': ['pm'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em9'},
        },
        'corrs_phi_k_2': {
            'requires': [],
            'ebcs': [],
            'epbcs': periodic['per_pm'],
            'equations': {
                'eq':
                    """dw_diffusion.i.Zm(hmatrix.K, qm, pm)
                   = - dw_surface_ndot.i.Gamma_Zm2(mat.k%d, qm)""",
            },
            'class': cb.CorrEqPar,
            'eq_pars': [(ii + 1) for ii in range(dim)],
            'save_name': 'corrs_phi_k_2',
            'dump_variables': ['pm'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em9'},
        },
        # Corrector problem, see (32)
        'corrs_psi_ij_1': {
            'requires': ['pis_w1'],
            'ebcs': ['fixed_w1'],
            'epbcs': periodic['per_w1'] + periodic['per_pc1'],

            'equations': {
                'eq1':
                    """2*dw_lin_elastic.i.Zc1(fluid.eta_c, z1, w1)
                     - dw_stokes.i.Zc1(z1, pc1)
                   = - 2*dw_lin_elastic.i.Zc1(fluid.eta_c, z1, Pi_w1)""",
                'eq2':
                    """dw_stokes.i.Zc1(w1, qc1)
                   = - dw_stokes.i.Zc1(Pi_w1, qc1)"""
            },
            'set_variables': [('Pi_w1', 'pis_w1', 'w1')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_psi_ij_1',
            'dump_variables': ['w1', 'pc1'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em15'},
            # 'solvers': {'ls': 'ls_s1', 'nls': 'ns_em15'},

            'is_linear': True,
        },
        'corrs_psi_ij_2': {
            'requires': ['pis_w2'],
            'ebcs': ['fixed_w2'],
            'epbcs': periodic['per_w2'] + periodic['per_pc2'],

            'equations': {
                'eq1':
                    """2*dw_lin_elastic.i.Zc2(fluid.eta_c, z2, w2)
                     - dw_stokes.i.Zc2(z2, pc2)
                   = - 2*dw_lin_elastic.i.Zc2(fluid.eta_c, z2, Pi_w2)""",
                'eq2':
                    """dw_stokes.i.Zc2(w2, qc2)
                   = - dw_stokes.i.Zc2(Pi_w2, qc2)"""
            },
            'set_variables': [('Pi_w2', 'pis_w2', 'w2')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_psi_ij_1',
            'dump_variables': ['w2', 'pc2'],
            'solvers': {'ls': 'ls', 'nls': 'ns_em15'},
            # 'solvers': {'ls': 'ls_s2', 'nls': 'ns_em15'},

            'is_linear': True,
        },

    }

    return locals()