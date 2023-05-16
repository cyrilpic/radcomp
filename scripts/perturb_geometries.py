import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

relevant_columns = ['r4', 'beta4', 'beta2', 'n_blades', 'blade_e',
                    'b4', 'r2h', 'r2s', 'r1', 'r5', 'b5', 'l_ind',
                    'clearance', 'backface', 'beta2s']

independent_parameters = {
    'r4': [0.005, 0.25],
    'beta4': [-60., 0.],
    'beta2': [-70., -35.],
    'n_blades': [5, 20],
    'blade_e': [1.e-4, 3.e-3]
}


@click.command()
@click.option('--geometries', '-g', type=click.Path(exists=True, dir_okay=False))
@click.option('--references', '-r', type=click.Path(exists=True, dir_okay=False))
@click.option('--method', '-m', type=click.Choice(['normal', 'pm', 'interp']), required=True)
@click.option('--output', '-o', type=str)
@click.option('--npoints', '-n', default=100)
def perturb_geometries(geometries, references, method, npoints, output):
    rng = np.random.default_rng()

    ref_t = pd.read_parquet(references)
    bounds = ref_t.loc[:, relevant_columns].agg(['min', 'max'])
    for k, v in independent_parameters.items():
        bounds.loc[:, k] = v
    lb, ub = bounds.to_numpy()
    scale = (ub - lb) * 1e-2

    geom_t = pd.read_parquet(geometries)
    n_parents = 2 if method == 'interp' else 1
    sampled = geom_t.sample(npoints*n_parents, replace=True).reset_index(drop=False).rename(columns={'geom_id': 'orig_geom_id'})
    sampled.index.name = 'geom_id'

    if method == 'normal':
        mask = np.full(npoints*n_parents, True)
        new_values = sampled.loc[:, relevant_columns]
        while mask.any():
            noise = rng.normal(scale=scale, size=(mask.sum(), len(scale)))
            new_values.loc[mask] = sampled.loc[mask, relevant_columns] + noise
            mask = (((new_values < lb) | (new_values > ub)).any(axis=1)
                    | (new_values.r4 <= new_values.r2s)
                    | (new_values.r2s <= new_values.r2h)).values

    sampled.loc[:, relevant_columns] = new_values

    sampled.loc[:, 'n_blades'] = sampled.loc[:, 'n_blades'].round()
    sampled.loc[:, 'n_splits'] = sampled.loc[:, 'n_blades']
    sampled.loc[sampled.n_splits > 11, 'n_splits'] = 0

    table = pa.Table.from_pandas(sampled, preserve_index=True)
    pq.write_table(table, output)


if __name__ == '__main__':
    perturb_geometries()
