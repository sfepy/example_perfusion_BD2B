# This example implements macroscopic homogenized model of Biot-Darcy-Brinkman model of flow in deformable
# double porous media.
# The mathematical model is described in:
#
#ROHAN E., TURJANICOVA J., LUKES V.
#Multiscale modelling and simulations of tissue perfusion using the Biot-Darcy-Brinkman model.
# Computers & Structures, 2020,
#
# Run simulation:
#
#   ./simple.py example_perfusion_BD2B/perf_BD2B_mac.py
#
# The results are stored in `example_perfusion_BDB/results/macro` directory.
#

import numpy as nm
from sfepy.homogenization.micmac import get_homog_coefs_linear
from sfepy.homogenization.utils import define_box_regions
from sfepy.discrete.fem.mesh import Mesh
from sfepy.solvers.ts import TimeStepper
from sfepy.base.base import Struct, debug
from sfepy.discrete import Problem
import os.path as osp
from scipy.io import savemat
import tools


material_cache = {}
data_dir = 'example_perfusion_BD2B'
val_w1=1e-8

def mtx_hook(mtx, pb, call_mode=None):
    # MatrixMarket I/O Functions for Matlab

    if call_mode == 'basic':
        import scipy.io as sio

        dmtx = mtx.todense()
        out = {}
        lvars = pb.get_variables()
        for ivar in lvars.iter_state():
            aindx = lvars.adi.indx[ivar.name]
            if ivar.eq_map.n_eq > 0:
                vn = ivar.name
                out[vn] = (aindx.start, aindx.start + ivar.eq_map.n_eq)
                sl = slice(aindx.start, aindx.start + ivar.eq_map.n_eq)
                out['K_%s%s' % (vn, vn)] = dmtx[sl, sl]

        sio.savemat('0mtx', out)

    return mtx

#Returns coefficients in quadrature points
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

# Get raw homogenized coefficients, recalculate them if necessary
def get_raw_coefs(problem):
    if 'raw_coefs' not in material_cache:
        micro_filename = material_cache['meso_filename']
        coefs_filename = 'coefs_meso'
        coefs_filename = osp.join(problem.conf.options.get('output_dir', '.'),
                                  coefs_filename) + '.h5'
        coefs = get_homog_coefs_linear(0, 0, None,
                                       micro_filename=micro_filename, coefs_filename=coefs_filename)
        coefs['B'] = coefs['B'][:, nm.newaxis]
        material_cache['raw_coefs'] = coefs
    return material_cache['raw_coefs']

#Get homogenized coefficients in quadrature points
def get_homog(coors,pb, mode,  **kwargs):
    if not (mode == 'qp'):
        return
    nqp = coors.shape[0]

    coefs=get_raw_coefs(pb)
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

#Definition of dirichlet boundary conditions
def get_ebc( coors, amplitude,  cg1, cg2,const=False):
    """
    Define the essential boundary conditions as a function of coordinates
    `coors` of region nodes.
    """
    y = coors[:, 1] - cg1
    z = coors[:, 2] - cg2

    val = amplitude*((cg1**2 - (abs(y)**2))+(cg2**2 - (abs(z)**2)))
    if const:
        val=nm.ones_like(y) *amplitude

    return val

#Returns value of \phi_\alpha\bar{w}^{mes,\alpha} as a material function
def get_ebc_mat(  coors,pb, mode, amplitude,  cg1, cg2,konst=False):
    if mode == 'qp':
        val = get_ebc(  coors, amplitude,  cg1, cg2,konst)
        phic1 = get_raw_coefs(pb)['vol']["fraction_Zc1"]
        phic2 = get_raw_coefs(pb)['vol']["fraction_Zc2"]
        v_w1 = val[:, nm.newaxis, nm.newaxis]
        return {'val_w1': v_w1*phic1,'val_w2': v_w1*phic2}

#Returns time-dependent boundary condition of displacement as a material function
def get_u(ts, coors):
    nqp, dim = coors.shape
    ut=8.0e-4
    t_stop=0.5
    val = nm.tile(-ut*ts.time, (nqp, 1, 1))
    if ts.time>t_stop:
        val = nm.tile(-ut * t_stop, (nqp, 1, 1))
        print "---plato---"
    return val

def get_ebc_w_from_consistency(pb,coors, val_w1):
    """Dopocita konzistentni okrajovou podminu na w2"""
    nqp, dim = coors.shape
    phi1 = get_raw_coefs(pb)['vol']["fraction_Zc1"]
    phi2 = get_raw_coefs(pb)['vol']["fraction_Zc2"]
    pvars = pb.create_variables(['svar'])
    aux=nm.ones((pvars['svar'].n_nod,1))
    pvars['svar'].set_data(aux)
    # area_in=pb.evaluate('d_surface.is.In(svar)',mode="eval",var_dict={"svar":pvars['svar']})
    # area_out=pb.evaluate('d_surface.is.Out(svar)',mode="eval",var_dict={"svar":pvars['svar']})
    # val = nm.tile(val_w1*phi1*area_in/(area_out*phi2), (nqp, 1, 1))
    val = nm.tile(val_w1*phi1/(phi2), (nqp, 1, 1))
    return val

#Definition of boundary conditions for numerical example at http://sfepy.org/sfepy_examples/example_perfusion_BD2B/
def define_bc(cg1,cg2,is_steady=False, val_in=1e2, val_out=1e2):

    funs = {
        'w_in': (lambda  ts, coor, bc, problem, **kwargs:
                 get_ebc( coor, val_in, cg1, cg2),),
        'w_out': (lambda  ts, coor, bc, problem, **kwargs:
                  get_ebc(  coor, val_out,  cg1, cg2),),
        'w_in_mat': (lambda  ts,coor, problem, mode=None, **kwargs:
                     get_ebc_mat( coor, problem, mode, val_in,
                                  cg1, cg2),),
        'w_out_mat': (lambda  ts,coor, problem, mode=None, **kwargs:
                      get_ebc_mat(  coor, problem, mode, val_out,
                                    cg1, cg2),),
        'get_ebc_w_consistency': (
            lambda ts, coor, bc, problem, **kwargs: get_ebc_w_from_consistency(problem, coor, val_w1),),
    }
    mats = {
        'w_in': 'w_in_mat',
        'w_out': 'w_out_mat',
    }

    ebcs = {
        'w_in': ('In', {'w1.0': 'w_in','w1.[1,2]': 0.0, 'w2.0': 0.0}),
        'u_in': ('In', { 'u.0': 0.0, }),
        'w_out': ('Out', {'w1.0': 0.0, 'w2.[1,2]': 0.0 }),
        'B_dirichlet_w':('Bottom',{'w1.2' :0.0,'w2.2' :0.0}),
        'T_dirichlet_w':('Top',{'w1.2' :0.0,'w2.2' :0.0}),
        'N_dirichlet_w': ('Near', {'w1.1': 0.0, 'w2.1': 0.0}),
        'F_dirichlet_w': ('Far', {'w1.1': 0.0, 'w2.1': 0.0}),
        'N_dirichlet_u': ('Near', {'u.1': 0.0}),
        'F_dirichlet_u': ('Far', {'u.1': 0.0}),
        'B_dirichlet_u': ('Bottom', { 'u.all': 0.0}),

    }
    lcbcs = {
               'imv': ('Omega', {'ls.all' : None}, None, 'integral_mean_value')
            }
    if is_steady:
        ebcs.update({
            'T_dirichlet_u_steady': ('Top', {'u.all': 0.0}),
        })
    else:
        ebcs.update({  # x
            'T_dirichlet_u_time': ('Top', {'u.[0,1]': 0.0,'u.2': "get_u"}),
        })
    return ebcs, funs, mats, lcbcs

#Definition of macroscopic equation for steady state problem, see
def define_steady_state_equations():
    phook = 'steady_state'

    equations = {
        'eq1': """
            dw_lin_elastic.i.Omega(hom.A, v, u)
          - dw_biot.i.Omega(hom.B, v, p)
          - dw_v_dot_grad_s.i.Omega(hom.P1T, v, p)
          - dw_v_dot_grad_s.i.Omega(hom.P2T, v, p)
          - dw_volume_dot.i.Omega(hom.H11, v, w1)
          - dw_volume_dot.i.Omega(hom.H12, v, w1)
          - dw_volume_dot.i.Omega(hom.H21, v, w2)
          - dw_volume_dot.i.Omega(hom.H22, v, w2)

          = 0""",

        'eq2': """
            dw_diffusion.i.Omega(hom.K, q, p)
          - dw_v_dot_grad_s.i.Omega(hom.P1, w1, q)
          - dw_v_dot_grad_s.i.Omega(hom.P2, w2, q)
          + dw_volume_dot.i.Omega( q,ls )
        = + dw_surface_integrate.is.In(w_in.val_w1, q) 
          - dw_surface_integrate.is.Out(w_out.val_w2, q)
          """,

        'eq3': """
            dw_lin_elastic.i.Omega(hom.S1, z1, w1)
          + dw_volume_dot.i.Omega(hom.H11, z1, w1)
          + dw_volume_dot.i.Omega(hom.H12, z1, w1)
          + dw_v_dot_grad_s.i.Omega(hom.P1T, z1, p)
          = 0""",

        'eq4': """
            dw_lin_elastic.i.Omega(hom.S2, z2, w2)
          + dw_volume_dot.i.Omega(hom.H21, z2, w2)
          + dw_volume_dot.i.Omega(hom.H22, z2, w2)
          + dw_v_dot_grad_s.i.Omega(hom.P2T, z2, p)
          = 0""",
        'eq_imv': 'dw_volume_dot.i.Omega( lv, p ) = 0',
    }
    return equations, phook

#Definition of macroscopic equation for time dependent problem, see
def define_time_equations(tstep):

    phook = 'time_evolution'
    dtime = (tstep.t1 - tstep.t0) / float(tstep.n_step - 1)
    dt = tstep.dt
    assert (dtime == dt)

    equations = {
        'eq1': """
            dw_lin_elastic.i.Omega(hom.A, v, u)
          - dw_biot.i.Omega(hom.B, v, p)
          - dw_v_dot_grad_s.i.Omega(hom.P1T, v, p)
          - dw_v_dot_grad_s.i.Omega(hom.P2T, v, p)
          - dw_volume_dot.i.Omega(hom.H11, v, w1)
          - dw_volume_dot.i.Omega(hom.H12, v, w1)
          - dw_volume_dot.i.Omega(hom.H21, v, w2)
          - dw_volume_dot.i.Omega(hom.H22, v, w2)
          = 0""",

        'eq2': """
            %e * dw_diffusion.i.Omega(hom.K, q, p)
          - %e * dw_v_dot_grad_s.i.Omega(hom.P1, w1, q)
          - %e * dw_v_dot_grad_s.i.Omega(hom.P2, w2, q)
          + %e * dw_volume_dot.i.Omega( q,ls )
          + dw_biot.i.Omega(hom.B, u, q)
          + dw_volume_dot.i.Omega(hom.M, q, p)
          = 
          + dw_biot.i.Omega(hom.B, U, q)
          + dw_volume_dot.i.Omega(hom.M, q, P)
          + %e * dw_surface_integrate.is.In(w_in.val_w1, q) 
          - %e * dw_surface_integrate.is.Out(w_out.val_w2, q)
          """%(dt,dt,dt,dt,dt,dt),

        'eq3': """
            %e * dw_lin_elastic.i.Omega(hom.S1, z1, w1)
          + dw_lin_elastic.i.Omega(hom.S1, z1, u)
          + %e * dw_volume_dot.i.Omega(hom.H11, z1, w1)
          + %e * dw_volume_dot.i.Omega(hom.H12, z1, w1)
          + %e * dw_v_dot_grad_s.i.Omega(hom.P1T, z1, p)
          = 
          + dw_lin_elastic.i.Omega(hom.S1, z1, U)
          """%(dt,dt,dt,dt),

        'eq4': """
            %e * dw_lin_elastic.i.Omega(hom.S2, z2, w2)
          + dw_lin_elastic.i.Omega(hom.S2, z2, u)
          + %e * dw_volume_dot.i.Omega(hom.H21, z2, w2)
          + %e * dw_volume_dot.i.Omega(hom.H22, z2, w2)
          + %e * dw_v_dot_grad_s.i.Omega(hom.P2T, z2, p)
          = 
          + dw_lin_elastic.i.Omega(hom.S2, z2, U)
          """%(dt,dt,dt,dt),
        'eq_imv': 'dw_volume_dot.i.Omega( lv, p ) = 0',
    }
    return equations, phook



#method for solving steady state problem
def steady_state(pb):
    print """
    ##################### steady state ##################
    """
    pb.flag = 'linear'

    outbase = tools.get_output_base(pb)

    conf = pb.conf.copy()
    conf.equations,_= define_steady_state_equations()
    ebcs_steady,_,_,_= define_bc(conf.cg1,conf.cg2,is_steady=True,val_in=conf.val_w1,val_out=conf.val_w2)

    conf.edit('ebcs', ebcs_steady)
    lpb = Problem.from_conf(conf)
    lpb.time_update()
    lpb.init_solvers(nls_conf=lpb.solver_confs['newton'])
    lpb.set_linear(False)
    state = lpb.solve().get_parts()
    state_data = {ii: state[ii] for ii in lpb.conf.state_vars}

    eval_data = tools.eval_macro(lpb, state_data, is_steady=True)
    tools.write_macro_fields(outbase + '_steady.vtk', eval_data,
                                  lpb.domain.mesh)
    return state_data

#method for solving time dependent problem using FDM numerical method
def time_evolution(pb):
    pb.flag = 'linear'
    out = []

    out_keys = ['E', 'p_e', 'A', 'B', 'M', 'K']
    outbase = tools.get_output_base(pb)
    out_data = tools.init_out_data(pb.domain.regions, pb.conf.tstep,
                                        keys=out_keys)

    update_vars = [('U',"u"),('P',"p")]
    mvars = pb.get_variables()

    # initial conditions obtained as solution of steady state problem
    init_state=steady_state(pb)

    u0=init_state['u'].reshape((mvars["U"].n_dof,))
    p0=init_state['p'].reshape((mvars["P"].n_dof,))
    state_data = {"U":u0,"P":p0}

    for step, time in pb.conf.tstep:
        print('##################################################')
        print('  step: %d' % step)
        print('##################################################')

        for ii,_ in update_vars:
            mvars[ii].set_data(state_data[ii])

        yield pb, out

        state = out[-1][1].get_parts()

        for ii, jj in update_vars:
            state_data[ii] = state[jj]

        state_data["W1"] = state["w1"]
        state_data["W2"] = state["w2"]

        eval_data = tools.eval_macro(pb, state_data)
        tools.write_macro_fields(outbase + '_%03d.vtk' % step, eval_data,
                           pb.domain.mesh)

        eval_data.update(tools.get_material_prop(pb.evaluate))
        tools.append_out_data(out_data, eval_data)

        state_data["U"]=state["u"]
        state_data["P"]=state["p"]
        yield None

    savemat(outbase + '.mat', out_data)



#Definition of macroscopic problem
def define(filename_mesh=None,cg1=None, cg2=None):
    if filename_mesh is None:
        filename_mesh = osp.join(data_dir, 'macro_perf.vtk')
        cg1, cg2 = 0.0015, 0.0015  # y and z coordinates of center of gravity

    mesh = Mesh.from_file(filename_mesh)
    poroela_mezo_file = osp.join(data_dir,'perf_BD2B_mes.py')
    material_cache['meso_filename']=poroela_mezo_file

    bbox = mesh.get_bounding_box()
    regions = define_box_regions(mesh.dim, bbox[0], bbox[1], eps=1e-6)

    regions.update({
        'Omega': 'all',
        'Wall': ('r.Top +v r.Bottom +v r.Far +v r.Near', 'facet'),
        # 'In': ('r.Left -v r.Wall', 'facet'),
        # 'Out': ('r.Right -v r.Wall', 'facet'),
        'In': ('copy r.Left', 'facet'),
        'Out': ('copy r.Right ', 'facet'),
        'Out_u': ('r.Out -v (r.Top +v r.Bottom)', 'facet'),

    })
    val_w1=5e3
    val_w2=5e3
    ebcs, bc_funs, mats, lcbcs = define_bc(cg1,cg2,is_steady=False,val_in=val_w1,val_out=val_w2)

    fields = {
        'displacement': ('real', 'vector', 'Omega', 1),
        'pressure': ('real', 'scalar', 'Omega', 1),
        'velocity1': ('real', 'vector', 'Omega', 1),
        'velocity2': ('real', 'vector', 'Omega', 1),
        'sfield': ('real', "scalar", 'Omega', 1),

    }

    variables = {
        #Displacement
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        #Pressure
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        'ls': ('unknown field', 'pressure'),
        'lv': ('test field', 'pressure', 'ls'),
        #Velocity
        'w1': ('unknown field', 'velocity1'),
        'z1': ('test field', 'velocity1', 'w1'),
        'w2': ('unknown field', 'velocity2'),
        'z2': ('test field', 'velocity2', 'w2'),
        'U': ('parameter field', 'displacement', 'u'),
        'P': ('parameter field', 'pressure', 'p'),
        'W1': ('parameter field', 'velocity1', 'w1'),
        'W2': ('parameter field', 'velocity2', 'w2'),
        'svar': ('parameter field', 'sfield', '(set-to-none)'),

    }
    state_vars = ['p','u','w1','w2']

    functions = {
        'get_homog': (lambda ts, coors, problem, mode=None, **kwargs: \
                          get_homog(coors,problem, mode, **kwargs),),
        'get_u': (lambda ts, coor, mode=None, problem=None, **kwargs:
                  get_u(tstep, coor),),


    }
    functions.update(bc_funs)

    materials = {
        'hom': 'get_homog',
    }
    materials.update(mats)

    #Definition of integrals
    integrals = {
        'i': 5,
        "is": ("s", 5),
    }
    #Definition of solvers
    solvers = {
        'ls': ('ls.mumps', {}),
        'newton': ('nls.newton',
                   {'i_max': 1,
                    'eps_a': 1e-10,
                    'eps_r': 1e-3,
                    'problem': 'nonlinear',
                    })
    }

    options = {
        'output_dir': data_dir + '/results/macro',
        'ls': 'ls',
        'nls': 'newton',
        'micro_filename' : poroela_mezo_file,
        'absolute_mesh_path': True,
        'output_prefix': 'Macro:',
        'matrix_hook': 'mtx_hook',
    }
    #Definition of time solver and equations for steady state and time evolution cases
    tstep =  TimeStepper(0.0, 1.0, n_step=20)
    equations,phook =  define_time_equations(tstep)

    options.update({'parametric_hook': phook})

    return locals()