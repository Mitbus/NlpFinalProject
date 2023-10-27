from enum import Enum


class Mode(Enum):
    translate = 'translate' # translate question and answer
    solve = 'solve' # solve math task
    formatting = 'formatting' # formating existing result


class G4fStatus(Enum):
    ok = 'ok'
    error = 'error'
    wrong_answer = 'wrong_answer'


COL_RESULT = 'result'
COL_ERROR = 'error'
COL_TMP = '_tmp'
COL_QUESTION = 'question'
COL_NUMBERS = 'numbers'
COL_ANSWER = 'answer'
COL_DIGIT_ANSWER = 'digit_answer'
