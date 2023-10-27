import os
import time
import logging
import argparse
import pandas as pd
from collections.abc import Callable

from utils.gpt_tools import *
from utils.data_tools import *
from constants import *



def execute_task(
        chunk_df: pd.DataFrame,
        make_texts: Callable[[pd.Series], str],
        execute_batch: Callable[[list[str]], list[str | Exception]],
        postprocess_result: Callable[[pd.Series], tuple[str, str] | Exception]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    texts = chunk_df.apply(make_texts, axis=1).tolist()
    chunk_df[COL_RESULT] = execute_batch(texts)
    
    # split by errors and others
    chunk_df[COL_ERROR] = chunk_df[COL_RESULT].apply(lambda res: None if type(res) is str else str(res))
    chunk_valid, chunk_err = chunk_df[chunk_df[COL_ERROR].isnull()], chunk_df[chunk_df[COL_ERROR].notnull()]

    # apply postprocessing and put into _tmp successful array or None (in warning case)
    chunk_valid[COL_TMP] = chunk_valid.apply(
        lambda row: None if type(x := postprocess_result(row)) is Exception else x,
        axis=1)
    
    # split by warnings and successes
    chunk_warn, chunk_succ = chunk_valid[chunk_valid[COL_TMP].isnull()], chunk_valid[chunk_valid[COL_TMP].notnull()]
    if len(chunk_succ) != 0:
        chunk_succ = pd.DataFrame(chunk_succ[COL_TMP].tolist(), columns=[COL_QUESTION, COL_ANSWER])
    else:
        chunk_succ = pd.DataFrame(columns=[COL_QUESTION, COL_ANSWER])
    
    chunk_warn = chunk_warn.drop(columns=[COL_TMP, COL_ERROR])
    chunk_err = chunk_err.drop(columns=[COL_RESULT])

    return chunk_succ, chunk_warn, chunk_err


def main(src_path: str, out_path: str, succ_name: str, warn_name: str, err_name: str, lang: str, mode: Mode, 
         chunk: int, delay: float):
    df_succ, df_warn, df_err = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    with pd.read_csv(src_path, sep=',', chunksize=chunk) as reader:
        for i, chunk_df in enumerate(reader):
            logging.info(f'Processing chunk {i+1}')
            if mode == Mode.translate:
                chunk_succ, chunk_warn, chunk_err = execute_task(
                    chunk_df, join_question_answer, lambda texts: translate_batch(texts, lang), translate_postproc
                )
            elif mode == Mode.solve:
                chunk_succ, chunk_warn, chunk_err = execute_task(
                    chunk_df, insert_numbers, lambda texts: solve_batch(texts), solve_postproc
                )
            elif mode == Mode.formatting:
                chunk_succ, chunk_warn, chunk_err = execute_task(
                    chunk_df, get_from_existing, lambda texts: format_batch(texts), solve_postproc
                )

            # add temporary results
            df_succ = pd.concat([df_succ, chunk_succ]).reset_index(drop=True)
            df_warn = pd.concat([df_warn, chunk_warn]).reset_index(drop=True)
            df_err = pd.concat([df_err, chunk_err]).reset_index(drop=True)
            
            # save chunk
            logging.info(f'Saving chunk {i+1}: succesfull {len(chunk_succ)}, warnings {len(chunk_warn)}, errors {len(chunk_err)}')
            chunk_succ.to_csv(os.path.join(out_path, f'{succ_name}_{i+1}.csv'), sep=',', index=False)
            chunk_warn.to_csv(os.path.join(out_path, f'{warn_name}_{i+1}.csv'), sep=',', index=False)
            chunk_err.to_csv(os.path.join(out_path, f'{err_name}_{i+1}.csv'), sep=',', index=False)
            
            logging.info(f'Delay {delay} seconds after chunk {i+1}')
            time.sleep(delay)
    
    # save
    logging.info(f'Saving total results: succesfull {len(df_succ)}, warnings {len(df_warn)}, errors {len(df_err)}')
    df_succ.to_csv(os.path.join(out_path, f'{succ_name}.csv'), sep=',', index=False)
    df_warn.to_csv(os.path.join(out_path, f'{warn_name}.csv'), sep=',', index=False)
    df_err.to_csv(os.path.join(out_path, f'{err_name}.csv'), sep=',', index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # parse args
    parser = argparse.ArgumentParser(description='This script translates the dataset using GPT-3.5. As a result, the script will create three files: successful rows, potentially error rows, error rows.')
    parser.add_argument(
        'input',
        type=str,
        help='provide global path for input dataset file'
    )
    parser.add_argument(
        'output',
        type=str,
        help='provide the global path for the output directory. This directory must exist'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='translate',
        help='provide the mode of GPT request. Allowed: \ntranslate - translate question and answer, \nsolve - solve math task, \nformatting - formating existing result.'
    )
    parser.add_argument(
        '--out_files',
        type=str,
        default='successful,warning,error',
        help='specify 3 comma separeted file names: successful rows, potentially error rows, error rows (Default: successful,warning,error)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='russian',
        help='specify an string of full-named target language'
    )
    parser.add_argument(
        '--chunk',
        type=int,
        default=100,
        help='specify the chunk size of pandas operations'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.0,
        help='specify delay in seconds for avoid API errors (Default: 0)'
    )
    logging.info('Parsing args')
    args = parser.parse_args()

    # validate inputs
    out_files = args.out_files.split(',')
    assert len(out_files) == 3, 'out_files parameter must specify only 3 file names'
    assert args.mode in {'translate', 'solve', 'formatting'}, 'for the "mode" parameter allowed: translate, solve, formatting'
    mode = Mode(args.mode)

    main(args.input, args.output, *out_files, args.language, mode, args.chunk, args.delay)
