"""Merge tables"""
import pathlib
import os

import click
import pyarrow.parquet as pq


@click.command()
@click.option('--geometries', '-g', type=click.Path(exists=True, dir_okay=False), required=True)
def main(geometries):
    geometries = pathlib.Path(geometries)
    input_subfolder = geometries.parent / (os.path.splitext(geometries.parts[-1])[0] + "_tabular")
    output_subfolder = geometries.parent / (os.path.splitext(geometries.parts[-1])[0] + "_output")
    new_subfolder = geometries.parent / (os.path.splitext(geometries.parts[-1])[0])
    geom_t = pq.read_table(geometries)
    inputs_t = pq.read_table(input_subfolder)
    outputs_t = pq.read_table(output_subfolder)
    for c in outputs_t.column_names:
        if c not in inputs_t.column_names:
            inputs_t = inputs_t.append_column(c, outputs_t.column(c))
    full = inputs_t.join(geom_t, 'geom_id')
    pq.write_to_dataset(full, new_subfolder, partition_cols=['fluid'])


if __name__ == '__main__':
    main()
