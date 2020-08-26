import numpy as nm
from sfepy.base.base import Struct, debug
import os.path as osp


def get_output_base(pb):
    return osp.join(pb.conf.options.output_dir,
                    osp.split(pb.domain.name)[-1])

def get_material_prop(ev, keys=['K', 'A', 'B', 'M'], mode=''):
    out = {}
    for k in keys:
        out[k] = ev('ev_volume_integrate_mat.i.Omega(hom.%s%s, p)' % (mode, k),
                    mode='el_avg')

    return out

def get_recovery_regions(regs):
    eregs = []
    for reg in regs:
        if reg.name[:6] == 'el_out':
            eregs.append(reg.name)

    return eregs

def init_out_data(regs, tstep, keys=['E', 'w', 'p_e', 'A', 'B', 'M', 'Ke']):
    data_out = {}
    el_out = []

    for jj, reg in enumerate(get_recovery_regions(regs)):
        cid = regs[reg].get_cells(0)
        el_out.append(cid)
        for k in keys:
            data_out['%s_%d' % (k, cid)] = []

    el_out.sort()
    data_out['el_idxs'] = el_out
    data_out['n_iter'] = []
    data_out['time'] = tstep.times
    return data_out


def append_out_data(out, data, niter=0,
                    keys=['E', 'w', 'p_e', 'A', 'B', 'M', 'K']):
    for el in out['el_idxs']:
        ekey = '_%d' % el
        for k in keys:
            val = data[k][0] if type(data[k]) is tuple else data[k]
            n = nm.prod(val.shape)
            out[k + ekey].append(val.reshape(n,))

    out['n_iter'].append(niter)

def write_macro_fields(fname, data, mesh):
    vtkout = {}
    for k in data.keys():
        val, vn, flag = data[k]
        if len(val.shape) > 3 and (val.shape[3] > val.shape[2]):
            val = val.transpose((0, 1, 3, 2))

        if flag == 'v' and val.shape[0] > mesh.n_nod:
            val = val[:mesh.n_nod, ...]

        vtkout[k] = Struct(name='output_data',
                           mode='cell' if flag == 'c' else 'vertex',
                           dofs=None, var_name=vn, data=val)
    # debug()
    print("----STEP SAVED----")
    mesh.write(fname, out=vtkout)

#Method for evaluating macroscopic quantities
def eval_macro(pb, args, is_steady=False):
    mode = 'el_avg'
    vs = pb.create_variables(args.keys())
    for ii in args.keys():
        vs[ii].set_data(args[ii])
    ev = pb.evaluate
    out = {}
    if  not is_steady:
        out['u'] = (args['U'].reshape((vs['U'].n_nod, vs['U'].n_components)),
                    None, 'v')
        out['p'] = (args['P'].reshape((vs['P'].n_nod, 1)), None, 'v')
        out['w1'] = (args['W1'].reshape((vs['W1'].n_nod, vs['W1'].n_components)),
                     None, 'v')
        out['w2'] = (args['W2'].reshape((vs['W2'].n_nod, vs['W2'].n_components)),
                     None, 'v')

        # evaluate cauchy strain tensor in cells
        out['E'] = (ev('ev_cauchy_strain.i.Omega(U)',
                       var_dict={'U': vs['U']}, mode=mode), None, 'c')

        # evaluate pressure field in cells
        out['p_e'] = (ev('ev_volume_integrate.i.Omega(P)',
                         var_dict={'P': vs['P']}, mode=mode), None, 'c')

        # evaluate pressure gradient in cells
        out['grad_p'] = (ev('ev_grad.i.Omega(P)',
                            var_dict={'P': vs['P']}, mode=mode), None, 'c')
    else:
        out['w1'] = (args['w1'].reshape((vs['w1'].n_nod, vs['w1'].n_components)),
                     None, 'v')
        out['w2'] = (args['w2'].reshape((vs['w2'].n_nod, vs['w2'].n_components)),
                     None, 'v')
        out['u'] = (args['u'].reshape((vs['u'].n_nod, vs['u'].n_components)),
                    None, 'v')
        out['p'] = (args['p'].reshape((vs['p'].n_nod, 1)), None, 'v')
        nm.save(pb.conf.options.output_dir + "/p0", args['p'].reshape((vs['p'].n_nod, 1)))
        nm.save(pb.conf.options.output_dir + "/u0", args['u'].reshape((vs['u'].n_nod, vs['u'].n_components)))

    if hasattr(pb.conf.options, 'poroela_eval_macro_hook'):
        hook = pb.conf.options.poroela_eval_macro_hook
        out.update(hook(pb, args))

    return out