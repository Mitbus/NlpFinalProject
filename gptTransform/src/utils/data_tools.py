import re
import json
import logging
import pandas as pd

from constants import *


def translate_postproc(row: pd.Series, full_translate=False) -> tuple[str, str] | Exception:
    if type(row[COL_RESULT]) is not str:
        return Exception('Invalid output1')
    
    splited = row[COL_RESULT].split('\n@@@\n')
    if len(splited) != 2:
        return Exception('Invalid output2')
    
    question, answer = splited
    if any(c not in answer for c in '<>'):
        if full_translate and any(c in question.lower() for c in 'abcdefghijklmnopqrstuvwxyz\'"') or \
            any(c in answer.lower() for c in 'abcdefghijklmnopqrstuvwxyz\'"'):
                return Exception('Invalid output3')
    
    answer = answer.replace('> >', '>>').replace('< <', '<<')

    return question, answer


def solve_postproc(row: pd.Series, precision: int = 1e-6, force: bool = False) -> tuple[str, str] | Exception:
    if type(row[COL_RESULT]) is not str:
        if not force:
            return Exception('Invalid output1')
        else:
            row[COL_RESULT] = str(row[COL_RESULT])
    
    splited = row[COL_RESULT].split('\n####')
    if len(splited) != 2:
        if not force:
            return Exception('Invalid output2')
        else:
            splited = row[COL_RESULT].split('\n####', 1)
            if len(splited) == 1:
                splited.append(str(row[COL_DIGIT_ANSWER]))

    body, digit_answer = splited
    try:
        # fix popular mistake
        digit_answer = digit_answer.lower().split('answer:')[-1].rstrip('.')
        if abs(float(digit_answer) - float(row[COL_DIGIT_ANSWER])) >= precision and not force:
            return Exception('Invalid output3')
    except:
        if not force:
            return Exception('Invalid output4')
        else:
            digit_answer = str(row[COL_DIGIT_ANSWER])
    
    if '$' not in body and not force:
        return Exception('Invalid output5')
    
    # replace quotes to expected format
    while True:
        open_body = body.replace('$', '<<', 1)
        if open_body == body:
            break
        
        close_body = open_body.replace('$', '>>', 1)
        if close_body == open_body: 
            if not force:
                return Exception('Invalid output6')
            else:
                break
        
        body = close_body
    
    # convert tex format
    body = body.replace('\\times', '*')
    body = body.replace('×', '*')
    body = body.replace('÷', '/')
    body = body.replace('\\div', '/')
    pattern = r'\\frac\s*\{?([^\s{}]+?)\}?\s*\{?([^\s{}]+?)\}?'
    replacement = r'\1 / \2'
    body = re.sub(pattern, replacement, body)
    if '\\' in body or '{' in body or '}' in body:
        if not force:
            return Exception('Invalid output7')
        else:
            # remove patterns
            pattern = r'\\[a-zA-Z_]+([^{}]+?)'
            replacement = r'\1'
            body = re.sub(pattern, replacement, body)
            pattern = r'\{(.*?)\}'
            replacement = r'\1'
            body = re.sub(pattern, replacement, body)
    
    # duplicate calculations out of plugin
    pattern = r'<<(.*?)=(.*?)>>'
    replacement = r'\1= <<\1=\2>>\2'
    body = re.sub(pattern, replacement, body)
    body = body.replace('"""', '')
    row[COL_QUESTION] = insert_numbers(row)
    return row[COL_QUESTION], f'{body}\n#### {digit_answer.strip()}'


def join_question_answer(row: pd.Series) -> str:
    # NOTE: some agents ignoring << and >>
    answer = row[COL_ANSWER].replace('>>', '> >').replace('<<', '< <')
    return f'{row[COL_QUESTION]}\n@@@\n{answer}'


def insert_numbers(row: pd.Series) -> str:
    try:
        numbers = json.loads(row[COL_NUMBERS])
        text = row[COL_QUESTION]
        for i, num in enumerate(numbers):
            text = text.replace(f'number{i}', str(num))
        return text
    except:
        logging.warn(f'Failed to insert numbers into given text: "{row[COL_QUESTION]}"')
        return row[COL_QUESTION]

def get_from_existing(row: pd.Series) -> str:
    # Decided that result exists in series
    row[COL_RESULT] = row[COL_RESULT].replace('$', '') # fixes solve_postproc transformation: $ -> <<\>>
    return solve_postproc(row, force=True)[-1]