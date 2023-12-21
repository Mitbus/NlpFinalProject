import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter


def train(
    # model/data params
    base_model: str,
    train_data_path: str,
    test_data_path: str = "",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    eval_steps: int = 200,
    save_steps: int = 200,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"train_data_path: {train_data_path}\n"
            f"test_data_path: {test_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"logging_steps: {logging_steps}\n"
            f"eval_steps: {eval_steps}\n"
            f"save_steps: {save_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"],
            )  # NOTE: Where is data_point[output_key] ?
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    # tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    # NOTE: Use tokenizer from lora checkpoint
    assert resume_from_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(resume_from_checkpoint, use_fast=False)

    # NOTE: Let's check everything is OK with prompter and tokenizer
    # data_point = {
    #     'instruction': 'В апреле Наталья продала 48 клипов своим друзьям, а в мае продала вдвое меньше клипов. Сколько всего клипов продала Наталья в апреле и мае?',
    #     'output': 'Наталья продала 48 / 2 = <calculator>48 / 2</calculator>24 клипа в мае.\nВсего за апрель и май Наталья продала 48 + 24 = <calculator>48 + 24</calculator>72 клипа.\nОтвет: 72'
    # }
    # prompt_text = prompter.generate_prompt(
    #     data_point["instruction"],
    #     data_point["output"],
    # )
    # print(prompt_text)
    # prompt_tokens = generate_and_tokenize_prompt(data_point)
    # print(prompt_tokens)
    # assert False, 'Just check, stop here'

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,  # NOTE: Set False if loss becomes NaN
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = prepare_model_for_int8_training(model)

    if isinstance(lora_target_modules, str):
        lora_target_modules = lora_target_modules[1:-1].split(',')
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        resume_from_checkpoint = (
            False  # So the trainer won't try loading its state
        )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data = load_dataset("json", data_files=train_data_path)
    train_data = train_data["train"].shuffle().map(generate_and_tokenize_prompt)
    if test_data_path:
        val_data = load_dataset("json", data_files=test_data_path)
        val_data = val_data["train"].shuffle().map(generate_and_tokenize_prompt)
    else:
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps" if test_data_path else "no",
            save_strategy="steps",
            eval_steps=eval_steps if test_data_path else None,
            save_steps=save_steps,
            output_dir=output_dir,
            # save_total_limit=3,
            load_best_model_at_end=True if test_data_path else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else "tensorboard",
            run_name=wandb_run_name if use_wandb else output_dir.split("/")[-1],
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
