import os

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import warnings

from requests.exceptions import RequestsDependencyWarning
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Suppress only the RequestsDependencyWarning
warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

import numpy as np
import pandas as pd
import torch
from llm2vec import LLM2Vec
from peft.peft_model import PeftModel
from transformers import AutoConfig

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import transformers

device_map = "cuda:0"
# from vllm import LLM, SamplingParams

# Assuming 'dataset' is a pandas DataFrame and 'column_name' is the name of the column containing strings.
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, padding_side="left")
tokenizer.add_special_tokens({"pad_token": "<pad>"})

print("model")
model = AutoModelForCausalLM.from_pretrained(
    model_id, local_files_only=True, torch_dtype=torch.float16, device_map="auto"
)

model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))


print(model)


# model = LLM(model_id, dtype="half", tensor_parallel_size=4, gpu_memory_utilization=0.4)

# sampling_params = SamplingParams(
#     logprobs=True,  # Ensure that logits are returned
# )

# instruction = (
#     "Given the text feature named house of a tabular dataset, extract the corresponding value in the given example."
# )
# values = ["roma", "milano", "marsiglia"]
# queries = [[{"role": "system", "content": instruction}, {"role": "user", "content": f"house: {val}"}] for val in values]

# # input_ids = tokenizer.apply_chat_template(
# #     queries, add_generation_prompt=True, return_tensors="pt", padding=True, truncation=False
# # )


# outputs = model.generate(queries, sampling_params=sampling_params)


# with torch.no_grad():
#     out = model(input_ids.cuda())
# print(outs[-1]["logits"][0, -1].shape)
