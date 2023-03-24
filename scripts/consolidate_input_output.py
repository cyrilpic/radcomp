"""Merge tables"""
import pathlib
import os

import click
import pyarrow.parquet as pq


@click.command()
@click.option('--geometries', '-g', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--conditions', '-c', type=click.Path(exists=True, dir_okay=True))
def main(geometries, conditions):
    geometries = pathlib.Path(geometries)
    if conditions is None:
        conditions = geometries.parent / (os.path.splitext(geometries.parts[-1])[0] + "_tabular")
    input_subfolder = conditions
    output_subfolder = str(conditions).replace("_tabular", "_output")
    new_subfolder = str(conditions).replace("_tabular", "")
    geom_t = pq.read_table(geometries)
    inputs_t = pq.read_table(input_subfolder)
    outputs_t = pq.read_table(output_subfolder).sort_by([('geom_id', "ascending"), ('cond_id', "ascending")])
    g_i = geom_t.join(inputs_t, 'geom_id').combine_chunks().sort_by([('geom_id', "ascending"), ('cond_id', "ascending")])
    if g_i.column('cond_id') != outputs_t.column('cond_id'):
        raise ValueError('Tables not aligned')
    for k in outputs_t.column_names:
        if k not in g_i.column_names:
            g_i = g_i.append_column(k, outputs_t.column(k))
    pq.write_to_dataset(g_i, new_subfolder, use_legacy_dataset=False, existing_data_behavior='delete_matching')


if __name__ == '__main__':
    main()
