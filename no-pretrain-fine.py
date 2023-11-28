from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
import transformers
import torch
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
from accelerate import Accelerator
import os
import json
import jsonlines
import pandas as pd

from torch.utils.data import Dataset as Dataset2

from datasets import load_dataset, Dataset, Features, Value

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel, PeftConfig


def gen(x):
    gened = model.generate(
        **tokenizer(
            f"##질문:{x}\n##답변:",
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        max_new_tokens=256,
        no_repeat_ngram_size=5,
        # top_p=0.8,
        temperature=0.8,
        early_stopping=True,
        # num_return_sequences=5,
        do_sample=True,
        eos_token_id=2,
        pad_token_id=2
    )
    print(gened[0])
    print(tokenizer.decode(gened[0]))
    
    
model_name = "EleutherAI/polyglot-ko-1.3b"

config = LoraConfig(
    r=8,
    lora_alpha=32, # 32
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"":0})
model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"":0})

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)

model.eval()
model.config.use_cache = True  # silence the warnings. Please re-enable for inference!

while 1:
    ques = input('질문:')
    print(ques)
    if ques == '1': break
    gen(ques)
    