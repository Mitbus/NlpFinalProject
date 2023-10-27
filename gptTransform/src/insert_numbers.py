import logging
import argparse
import pandas as pd

from constants import *
from utils.data_tools import insert_numbers


def main(input: str, source: str, output: str):
    logging.info('Reading files')
    joined_df = pd.merge(pd.read_csv(input, sep=','), pd.read_csv(source, sep=','), on=COL_QUESTION, how='left')

    logging.info('Processing')
    joined_df[COL_QUESTION] = joined_df.apply(insert_numbers, axis=1)
    joined_df.drop(joined_df.columns.difference([COL_QUESTION, COL_ANSWER]), axis=1, inplace=True)
    
    logging.info('Saving results')
    joined_df.to_csv(output, sep=',', index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # parse args
    parser = argparse.ArgumentParser(description='This script insert numbers to question column by given source dataset.')
    parser.add_argument(
        'input',
        type=str,
        help='provide csv file with question column'
    )
    parser.add_argument(
        'source',
        type=str,
        help='provide the source file'
    )
    parser.add_argument(
        'output',
        type=str,
        help='provide the global path for the output file'
    )
    logging.info('Parsing args')
    args = parser.parse_args()

    main(args.input, args.source, args.output)
