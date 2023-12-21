"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str, verbose: bool = False):
        self._verbose = verbose
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        output: Union[None, str] = None,
    ) -> str:
        res = self.template["prompt_no_input"].format(
            instruction=instruction
        )
        if output:
            res = f"{res}{output}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"], 1)[1].strip()
