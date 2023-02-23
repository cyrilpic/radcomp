import bz2
import pickle
import time

import click
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dask_mpi import initialize
from distributed.worker import thread_state
from dask.distributed import Client


from radcomp.compressor import Compressor
from radcomp.condition import OperatingCondition
from radcomp.geometry import Geometry
from radcomp.thermo import CoolPropFluid


out_meta = {
    'cond_id': pd.Series(dtype=np.int64),
    'geom_id': pd.Series(dtype=np.int64),
    'fluid': pd.Series(dtype=str),
    'calc_tip_speed': pd.Series(dtype=np.double),
    'calc_n_rot': pd.Series(dtype=np.double),
    'calc_m_f': pd.Series(dtype=np.double),
    'comp_valid': pd.Series(dtype=bool),
    'comp_error': pd.Series(dtype=str),
    'comp_eta_tt': pd.Series(dtype=np.double),
    'comp_pr': pd.Series(dtype=np.double),
    'comp_m_in': pd.Series(dtype=np.double),
    'comp_n_rot_corr': pd.Series(dtype=np.double),
    'comp_flow': pd.Series(dtype=np.double),
    'comp_head': pd.Series(dtype=np.double),
    'comp_power': pd.Series(dtype=np.double),
    'dtime': pd.Series(dtype=np.double)
}

out_thermo_meta = out_meta.copy()
out_thermo_meta.update({
    'in0_mu': pd.Series(dtype=np.double),
    'in0_rho': pd.Series(dtype=np.double),
    'in0_A': pd.Series(dtype=np.double)
})

out_meta = pd.DataFrame(out_meta)
out_thermo_meta = pd.DataFrame(out_thermo_meta)


def simulate(df, geom_file=None, add_thermo=False):
    df = df.reset_index(drop=False)
    geom_t = pq.read_table(geom_file,
                           filters=[('geom_id', '>=', df.geom_id.min()),
                                    ('geom_id', '<=', df.geom_id.max())]).to_pandas()

    out_m = out_thermo_meta if add_thermo else out_meta
    out = {n: np.empty(len(df), dtype=t) for n, t in out_m.dtypes.items()}

    out['cond_id'] = df.cond_id
    out['geom_id'] = df.geom_id
    out['fluid'] = df.fluid

    for row in df.itertuples():
        fld = CoolPropFluid(row.fluid)
        geom = Geometry.from_dict(geom_t.loc[row.geom_id, :].to_dict())
        in0 = fld.thermo_prop('PT', row.in_P, row.in_T)
        if add_thermo:
            out['in0_mu'][row.Index] = in0.V
            out['in0_rho'][row.Index] = in0.D
            out['in0_A'][row.Index] = in0.A
        m_f = row.in_m_in0 * in0.A * in0.D * geom.A2_eff
        tip_speed = row.in_mach_tip * in0.A
        n_rot = tip_speed / geom.r4
        out['calc_m_f'][row.Index] = m_f
        out['calc_n_rot'][row.Index] = n_rot
        out['calc_tip_speed'][row.Index] = tip_speed

        t0 = time.perf_counter()
        op = OperatingCondition(in0=in0, fld=fld, m=m_f, n_rot=n_rot)
        comp = Compressor(geom, op)
        try:
            valid = comp.calculate()
            dtime = time.perf_counter() - t0
        except Exception as e:
            dtime = time.perf_counter() - t0
            out['comp_error'][row.Index] = repr(e)
            out['comp_valid'][row.Index] = False
        else:
            out['comp_valid'][row.Index] = valid
            out['comp_eta_tt'][row.Index] = comp.eff
            out['comp_pr'][row.Index] = comp.PR
            out['comp_m_in'][row.Index] = comp.m_in
            out['comp_n_rot_corr'][row.Index] = comp.n_rot_corr
            out['comp_flow'][row.Index] = comp.flow
            out['comp_head'][row.Index] = comp.head
            out['comp_power'][row.Index] = comp.power
        out['dtime'][row.Index] = dtime

    out_df = pd.DataFrame(out, columns=out_m.columns)
    return out_df.set_index('cond_id')


# End general setup
initialize()


@click.command()
@click.option('--geometries', '-g', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--conditions', '-c', type=click.Path(exists=True, file_okay=False), required=True)
@click.option('--output-npartitions', default=20)
@click.option('--thermo/--no-thermo', default=False)
def main(geometries, conditions, output_npartitions, thermo):
    out_m = out_thermo_meta if thermo else out_meta
    c = Client()
    output_name = conditions.replace('_tabular', '_output')
    tab = dd.read_parquet(conditions, index='cond_id', split_row_groups=True)
    out = tab.map_partitions(simulate, geom_file=geometries, add_thermo=thermo, meta=out_m.set_index('cond_id'))
    out.repartition(npartitions=output_npartitions).to_parquet(output_name)
    c.close()


if __name__ == '__main__':
    main()
