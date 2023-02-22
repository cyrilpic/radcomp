"""Script to generate new geometries"""
import ast
import warnings

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats.qmc import Sobol

# supported operators
operators = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
             ast.Div: np.true_divide, ast.USub: np.negative, ast.LtE: np.less_equal}


independent_parameters = {
    'r4': [0.005, 0.075],
    'beta4': [-60., 0.],
    'beta2': [-70., -35.],
    'n_blades': [5, 20],
    'blade_e': [1.e-4, 5.e-4],
    'blockage1': [0.8, 1.],
    'blockage2': [0.8, 1.],
    'blockage3': [0.8, 1.],
    'blockage4': [0.8, 1.],
    'blockage5': [0.8, 1.]
}

relative_parameters = {
    'b4': ['0.015*r4', '0.3*r4'],
    'r2h': ['0.1*r4', '0.3*r4'],
    'r2s': ['1.2*r2h', '0.7*r4'],
    'r1': ['r2s', 'r4'],
    'r5': ['1*r4', '1.5*r4'],
    'b5': ['0.5*b4', '1.5*b4'],
    'l_ind': ['1*r4', '4*r4'],
    'clearance': ['0.01*b4', '0.15*b4'],
    'backface': ['0.001*r4', '0.15*r4'],
    'beta2s': ['beta2-20', 'beta2'],
}

fixed_parameters = {
    'rug_imp': '1.2e-5',
    'rug_ind': '1.2e-5',
    'l_comp': '0.7*r4',
    'n_splits': '(n_blades<=11)*n_blades',
    'alpha2': '0.'
}


def eval_(node, iparams):
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.Name):
        return iparams[node.id]
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left, iparams), eval_(node.right, iparams))
    elif isinstance(node, ast.Compare) and len(node.ops) == 1:  # <left> <operator> <right> only single comparison supported
        return operators[type(node.ops[0])](eval_(node.left, iparams), eval_(node.comparators[0], iparams))
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand, iparams))
    else:
        raise TypeError(node)


@click.command()
@click.option('--method', '-m', type=click.Choice(['sobol', 'uniform']), required=True)
@click.option('--npoints', '-n', default=100)
@click.option('--output', '-o', type=str)
def sample_geometries(method, npoints, output):
    # Random part
    # Set seed
    #

    # Get random number
    n_parameters = len(independent_parameters) + len(relative_parameters)
    if method == 'sobol':
        sobol_engine = Sobol(n_parameters, scramble=True, bits=30)
        m = int(np.log2(npoints))
        if m != np.log2(npoints):
            warnings.warn(f'Number of points is not in the form 2^m, using m={m}.')
        X = sobol_engine.random_base2(m)
    elif method == 'uniform':
        X = np.random.random_sample((npoints, n_parameters))
    else:
        raise TypeError('method is unknown or undefined')

    # Scale
    lb, ub = np.array(list(independent_parameters.values())).T
    iX, rX = X[:, :len(lb)], X[:, len(lb):] 
    iX = lb + (ub - lb) * iX

    for i, bounds in enumerate(independent_parameters.values()):
        if isinstance(bounds[0], int):
            iX[:, i] = np.round(iX[:, i])

    iparams = {k: iX[:, i] for i, k in enumerate(independent_parameters.keys())}
    for i, (k, str_exprs) in enumerate(relative_parameters.items()):
        lb = eval_(ast.parse(str_exprs[0], mode='eval').body, iparams)
        ub = eval_(ast.parse(str_exprs[1], mode='eval').body, iparams)
        iparams[k] = lb + (ub - lb) * rX[:, i]

    for k, str_expr in fixed_parameters.items():
        iparams[k] = eval_(ast.parse(str_expr, mode='eval').body, iparams)

    df = pd.DataFrame(iparams)
    df.index.name = 'geom_id'

    if output.endswith('.json'):
        df.to_json(output, orient='records')
    elif output.endswith('.csv'):
        df.to_csv(output)
    elif output.endswith('.parquet'):
        table = pa.Table.from_pandas(df, preserve_index=True)
        pq.write_table(table, output)


if __name__ == '__main__':
    sample_geometries()
