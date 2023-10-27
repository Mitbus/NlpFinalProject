import g4f
from collections import defaultdict 
import asyncio
import time
import json
import logging
import random

from constants import *

g4f.logging = False
g4f.check_version = False


# GPT interaction


def translate_prompt(text: str, lang: str) -> str:
    return f'''You are a highly skilled AI trained in language translation. I would like you to translate the text delimited by triple quotes into {lang} language, ensuring that the translation is colloquial and authentic.
Translate all words, including names. Try to maintain the original punctuation. Only give me the output and nothing else.
"""
{text}
"""'''


def solve_task_prompt(text: str) -> str:
    return f'''You are a highly skilled AI trained in solving math problems. I would like you to answer the math question enclosed in triple quotes by reasoning step by step. 
Use calculations (digits only) with $ symbols only, for example $ 2 + 2 = 4 $ . No equations with "x" allowed. Use #### to separate the answer (it should be numbers only) from the rest of the text.
"""
{text}
"""'''

def formatting_task_prompt(text: str) -> str:
    return f'''You are a highly skilled AI trained in solving math problems. I would like you to formatting the math question enclosed in triple quotes according to the following requirements.
Any calculations must used with $ symbols only, for example $ 2 + 2 = 4 $ . Calculations should be made with numbers only. Use #### to separate the answer from the rest of the text.
"""
{text}
"""'''


def is_correct_answer(answer: str) -> bool:
    return not (
        answer.startswith('Hmm, I am not sure') or \
        answer.startswith('Hmm, I am sorry') or \
        answer.startswith('IP访问频率过高,请稍后再试') or \
        len(answer) == 0
    )


# NOTE: need to be updated: https://github.com/xtekky/gpt4free
_providers = [
    g4f.Provider.ChatgptAi,
    g4f.Provider.FreeGpt, # antispam
    g4f.Provider.GPTalk, # may be unstable
    g4f.Provider.GeekGpt,
    g4f.Provider.FakeGpt,
    g4f.Provider.Bing,
    g4f.Provider.Hashnode,
    g4f.Provider.ChatForAi,

    # developing
    # g4f.Provider.unfinished.ChatAiGpt,
    # g4f.Provider.unfinished.Komo,
    # g4f.Provider.unfinished.MikuChat,
    # g4f.Provider.unfinished.PerplexityAi,
    # g4f.Provider.unfinished.TalkAi,


    # has limits
    # g4f.Provider.AiAsk, # daily or forever ?
    # g4f.Provider.NoowAi, # daily or forever ?
    g4f.Provider.ChatgptX, # 50.000 tokens per minute

    # problems
    # g4f.Provider.AItianhuSpace, # too long prompt
    # g4f.Provider.Aichat, # too long prompt
    # g4f.Provider.AItianhu, # too long prompt
    g4f.Provider.ChatBase, # ~90% results are wrong

    # errors
    # g4f.Provider.GptForLove,
    # g4f.Provider.GptGo,
    # g4f.Provider.You,
    # g4f.Provider.Phind,
    # g4f.Provider.GptChatly,
    # g4f.Provider.Vercel,
    # g4f.Provider.Ylokh,
    # g4f.Provider.Liaobots,
    # g4f.Provider.Cromicle,
    # g4f.Provider.ChatgptLogin,
    # g4f.Provider.ChatgptFree,
    # g4f.Provider.ChatgptDuo,
    # g4f.Provider.ChatgptDemo,
    # g4f.Provider.ChatgptAi,
    # g4f.Provider.Chatgpt4Online,
    # g4f.Provider.Ails,
    # g4f.Provider.Aibn,
    # g4f.Provider.Acytoo,
]
_providers_stat = defaultdict(lambda: defaultdict(lambda: 0))
keep_last_n = 2
provider_timeout_range = 120, 300


def update_stats(provider: g4f.Provider.BaseProvider, status: G4fStatus):
    _providers_stat[str(provider)]['_last_n_stat'].pop(0)
    _providers_stat[str(provider)]['_last_n_stat'].append(status.value)
        
    _providers_stat[str(provider)][status.value] += 1


def get_provider() -> g4f.Provider.BaseProvider:
    valid_providers = []
    cur_time = int(time.time())
    
    for provider in _providers:
        if type(_providers_stat[str(provider)]['_last_n_stat']) is int:
            _providers_stat[str(provider)]['_last_n_stat'] = [G4fStatus.ok.value] * keep_last_n
        last_n = _providers_stat[str(provider)]['_last_n_stat']
        ok = sum(G4fStatus.ok.value == s for s in last_n)
        if ok > 0:
            valid_providers.append(provider)
            continue
        if cur_time >= _providers_stat[str(provider)]['_wakeup_time']:
            valid_providers.append(provider)
            continue
    
    if len(valid_providers) == 0:
        provider = random.choice(_providers)
    else:
        provider = random.choice(valid_providers)

    _providers_stat[str(provider)]['_wakeup_time'] = cur_time + random.randint(*provider_timeout_range)
    return provider


# async 
def run_provider(prompt: str, rec_limit: int = 3) -> str | Exception:
    provider = get_provider()
    for i in range(rec_limit):
        try:
            if i != 0: logging.warn(f'Trying to reconnect')
            # response = await g4f.ChatCompletion.create_async(
            response = g4f.ChatCompletion.create(
                model=g4f.models.gpt_35_turbo ,
                messages=[{"role": "user", "content": prompt}],
                provider=provider
            )
            if is_correct_answer(response):
                update_stats(provider, G4fStatus.ok)
                return response
            else:
                update_stats(provider, G4fStatus.wrong_answer)
                raise Exception('Incorrect answer', response)
        except Exception as ex:
            logging.warn(f'Got exception ({provider}): {ex}')
            update_stats(provider, G4fStatus.error)
            provider = get_provider()
            response = ex
            # time.sleep(0.2)
    logging.error(f'Reconnection limit exceeded')
    return response

# async 
def gpt_wrapper(prompts: list[str]) -> list[str | Exception]:
    calls = [run_provider(p) for p in prompts]
    return calls #await asyncio.gather(*calls)


# task tools


def translate_batch(texts: list[str], lang: str) -> list[str | Exception]:
    prompts = [translate_prompt(t, lang) for t in texts]
    result = gpt_wrapper(prompts) # asyncio.run(gpt_wrapper(prompts))
    logging.info(f'Providers statistics: {json.dumps(_providers_stat, indent=4)}')
    return result


def solve_batch(texts: list[str]) -> list[str | Exception]:
    prompts = [solve_task_prompt(t) for t in texts]
    result = gpt_wrapper(prompts) # asyncio.run(gpt_wrapper(prompts))
    logging.info(f'Providers statistics: {json.dumps(_providers_stat, indent=4)}')
    return result


def format_batch(texts: list[str]) -> list[str | Exception]:
    prompts = [formatting_task_prompt(t) if t != '' else Exception('Wrong input') for t in texts]
    result = gpt_wrapper(prompts) # asyncio.run(gpt_wrapper(prompts))
    logging.info(f'Providers statistics: {json.dumps(_providers_stat, indent=4)}')
    return result