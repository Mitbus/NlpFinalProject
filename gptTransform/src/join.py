import logging
import argparse
import pandas as pd

from constants import *
from utils.data_tools import solve_postproc


def main(succ_files: list[str], warn_files: list[str], output: str):
    succ_dfs = [pd.read_csv(path, sep=',') for path in succ_files if path != '']
    warn_dfs = [pd.read_csv(path, sep=',') for path in warn_files if path != '']
    total_df = pd.DataFrame(columns=[COL_QUESTION, COL_ANSWER])

    logging.info('Processing successful')
    for df in succ_dfs:
        df.drop(df.columns.difference([COL_QUESTION, COL_ANSWER]), axis=1, inplace=True)
        total_df = pd.concat([total_df, df]).reset_index(drop=True)

    logging.info('Processing warnings')
    for df in warn_dfs:
        new_cols = df.apply(lambda row: solve_postproc(row, force=True), axis=1)
        df = pd.DataFrame(new_cols.tolist(), columns=[COL_QUESTION, COL_ANSWER])
        total_df = pd.concat([total_df, df]).reset_index(drop=True)
    
    logging.info('Saving results')
    total_df.to_csv(output, sep=',', index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # parse args
    parser = argparse.ArgumentParser(description='This script join the results to one file with fixed schema.')
    parser.add_argument(
        'input',
        type=str,
        help='provide comma separated global paths for input files'
    )
    parser.add_argument(
        'output',
        type=str,
        help='provide the global path for the output file'
    )
    parser.add_argument(
        '--warnings',
        type=str,
        default='',
        help='provide comma separated global paths for input warning files. Warnings may converted incorrect'
    )
    logging.info('Parsing args')
    args = parser.parse_args()

    # validate inputs
    succ_files = args.input.split(',')
    warn_files = args.warnings.split(',')

    main(succ_files, warn_files, args.output)
