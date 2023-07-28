import os
import pathlib

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from radcompressor import thermo


parameters = {"T_range": [170, 400], "Pr": [1, 100]}  # K

parameters_narrow = {"m_in": [0.15, 0.25], "mach_tip": [0.35, 0.7]}

parameters_wide = {"m_in": [5e-2, 0.7], "mach_tip": [5e-2, 2.5]}


@click.command()
@click.option("--geometries", "-g", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--output",
    type=click.Path(file_okay=False),
    default="data",
    help="Folder where dataset subfolders will be created",
)
@click.option("--output-row-size", default=5000)
@click.option("--fluid", "-f", "fluid_list", type=str, required=True, multiple=True)
@click.option(
    "--fluid-type", type=click.Choice(["coolprop", "refprop"]), default="coolprop"
)
@click.option("--batch-size", "-b", default=20000)
@click.option(
    "--n-points", "-no", default=10, help="Number of operating points (Mf, Nrot)"
)
@click.option(
    "--n-inlet", "-ni", default=10, help="Number of inlet conditions (Pin, Tin)"
)
@click.option("--narrow/--wide", default=False)
def main(
    geometries,
    output,
    output_row_size,
    fluid_list,
    fluid_type,
    batch_size,
    n_points,
    n_inlet,
    narrow,
):
    """Sample conditions for the provided geometries"""
    # Set bounds
    if narrow:
        parameters.update(parameters_narrow)
    else:
        parameters.update(parameters_wide)
    # Prepare fluid and load to check if it exists
    if fluid_type == "coolprop":
        fld = {f: thermo.CoolPropFluid(f) for f in fluid_list}
    elif fluid_type == "refprop":
        fld = {f: thermo.RefpropFluid(f) for f in fluid_list}

    # Prepare rng
    rng = np.random.default_rng()

    geometries = pathlib.Path(geometries)
    output_subfolder = pathlib.Path(output) / (
        os.path.splitext(geometries.parts[-1])[0] + "_tabular"
    )
    output_subfolder.mkdir(exist_ok=True)

    geom_ds = ds.dataset(geometries)

    idx_offset = 0

    for b in geom_ds.to_batches(batch_size=batch_size, columns=["geom_id"]):
        geom_idx = b.to_pandas().index
        n_geom = len(geom_idx)

        fluids = rng.choice(fluid_list, size=n_geom * n_inlet, replace=True)
        T_triple, T_max, T_crit, P_crit = np.array(
            [
                [fld[f].T_triple, fld[f].T_max, fld[f].T_crit, fld[f].P_crit]
                for f in fluids
            ]
        ).T
        T_low = np.maximum(T_triple + 30, parameters["T_range"][0])
        T_high = np.minimum(T_max - 50, parameters["T_range"][1])

        Teff = rng.uniform(low=T_low, high=T_high, size=n_geom * n_inlet)

        Pmax = P_crit * (Teff > T_crit)
        Pmax[Pmax == 0] = [
            fld[f].thermo_prop("TQ", t, 1).P - 1.1e-4
            for t, f in zip(Teff[Pmax == 0], fluids[Pmax == 0])
        ]
        # 1.1e-4 added to avoid Coolprop issues

        Pmax[Pmax > P_crit / 3] = P_crit[Pmax > P_crit / 3] / 3

        Peff_r = parameters["Pr"][0] + (1 - rng.power(5, len(Teff))) * (
            parameters["Pr"][1] - parameters["Pr"][0]
        )
        Peff = Pmax / Peff_r

        # Validate that each condition is valid
        [
            fld[f].thermo_prop("PT", p, t).P - 1.1e-4
            for t, f, p in zip(Teff, fluids, Peff)
        ]

        m_in = rng.uniform(
            low=parameters["m_in"][0],
            high=parameters["m_in"][1],
            size=n_geom * n_inlet * n_points,
        )
        mach_tip = rng.uniform(
            low=parameters["mach_tip"][0],
            high=parameters["mach_tip"][1],
            size=n_geom * n_inlet * n_points,
        )

        df = pd.DataFrame(
            {
                "geom_id": geom_idx.repeat(n_inlet * n_points),
                "fluid": fluids.repeat(n_points),
                "in_T": Teff.repeat(n_points),
                "in_P": Peff.repeat(n_points),
                "in_m_in0": m_in,
                "in_mach_tip": mach_tip,
            }
        )
        df.index.name = "cond_id"

        df.index += idx_offset
        idx_offset = df.index.max() + 1

        table = pa.Table.from_pandas(df, preserve_index=True)
        table_name = (
            output_subfolder / f"data_{geom_idx.min()}-{geom_idx.max()}.parquet"
        )
        pq.write_table(table, table_name, row_group_size=output_row_size)


if __name__ == "__main__":
    main()
