import os
import pathlib

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from radcomp import thermo


parameters = {
    'T_range': [150, 400],  # K
    'Pr': [1, 120],
    'm_in': [5e-2, 0.7],
    'mach_tip': [5e-2, 2.5]
}


@click.command()
@click.option('--geometries', '-g', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', type=click.Path(file_okay=False), default='data', help="Folder where dataset subfolders will be created")
@click.option('--fluid', type=str, required=True)
@click.option('--fluid-type', type=click.Choice(['coolprop', 'refprop']), default='coolprop')
@click.option('--batch-size', '-b', default=100000)
@click.option('--n-points', '-no', default=10, help="Number of operating points (Mf, Nrot)")
@click.option('--n-inlet', '-ni', default=10, help="Number of inlet conditions (Pin, Tin)")
def main(geometries, output, fluid, fluid_type, batch_size, n_points, n_inlet):
    """Sample conditions for the provided geometries"""
    # Prepare fluid and load to check if it exists
    if fluid_type == 'coolprop':
        fld = thermo.CoolPropFluid(fluid)
    elif fluid_type == 'refprop':
        fld = thermo.RefpropFluid(fluid)
        fld.activate()

    # Prepare rng
    rng = np.random.default_rng()

    geometries = pathlib.Path(geometries)
    output_subfolder = pathlib.Path(output) / (os.path.splitext(geometries.parts[-1])[0] + "_tabular")
    output_subfolder.mkdir(exist_ok=True)
    output_subfolder /= f'fluid={fluid}'
    output_subfolder.mkdir(exist_ok=True)

    geom_ds = ds.dataset(geometries)

    T_range = np.array([max(fld.T_triple+10, parameters['T_range'][0]),
                        min(fld.T_max-50, parameters['T_range'][1])])

    for b in geom_ds.to_batches(batch_size=batch_size, columns=['geom_id']):
        geom_idx = b.to_pandas().index
        n_geom = len(geom_idx)

        Teff = rng.uniform(low=T_range[0], high=T_range[1], size=n_geom*n_inlet)

        Pmax = fld.P_crit * (Teff > fld.T_crit)
        Pmax[Pmax == 0] = [fld.thermo_prop('TQ', t, 1).P for t in Teff[Pmax == 0]]

        Pmax[Pmax > fld.P_crit/3] = fld.P_crit/3

        Peff_r = parameters['Pr'][0] + (1-rng.power(5, len(Teff))) * (parameters['Pr'][1]-parameters['Pr'][0])
        Peff = Pmax/Peff_r

        m_in = rng.uniform(low=parameters['m_in'][0], high=parameters['m_in'][1], size=n_geom*n_inlet*n_points)
        mach_tip = rng.uniform(low=parameters['mach_tip'][0], high=parameters['mach_tip'][1], size=n_geom*n_inlet*n_points)

        df = pd.DataFrame({
            'geom_id': geom_idx.repeat(n_inlet*n_points),
            'in_T': Teff.repeat(n_points),
            'in_P': Peff.repeat(n_points),
            'in_m_in0': m_in,
            'in_mach_tip': mach_tip,
        }).set_index('geom_id')

        table = pa.Table.from_pandas(df)
        table_name = output_subfolder / f'data_{geom_idx.min()}-{geom_idx.max()}.parquet'
        pq.write_table(table, table_name, row_group_size=n_geom*n_inlet)


if __name__ == '__main__':
    main()
