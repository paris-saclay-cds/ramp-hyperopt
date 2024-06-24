import json
import os

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system("proxy")

from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs
from llm2vec import LLM2Vec
import torch
from peft.peft_model import PeftModel
from ramphy import Hyperparameter
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer

peft_model = Hyperparameter(
    dtype="str", default="mntp-supervised", values=["mntp-supervised", "mntp-unsup-simcse", "mntp"]
)
pooling_mode = Hyperparameter(
    dtype="str",
    default="mean",
    values=["mean", "eos_token", "weighted_mean", "bos_token"],
)
# positional is an experimental one that I am testing
encoding_mode = Hyperparameter(dtype="str", default="full", values=["full", "positional"])

PEFT_MODEL = str(peft_model)
POOLING_MODE = str(pooling_mode)
ENCODING_MODE = str(encoding_mode)


class DataPreprocessor(rs.BaseDataPreprocessor):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", device_map="auto")
        self.config = AutoConfig.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", trust_remote_code=True, device_map="auto"
        )
        self.model = AutoModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
            trust_remote_code=True,
            config=self.config,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.model = PeftModel.from_pretrained(self.model, "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", device_map="auto")
        self.model = self.model.merge_and_unload()
        self.model = PeftModel.from_pretrained(
            self.model, f"McGill-NLP/LLM2Vec-Sheared-LLaMA-{PEFT_MODEL}", device_map="auto"
        )
        self.l2v = LLM2Vec(self.model, self.tokenizer, pooling_mode=POOLING_MODE, max_length=512)

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        return

    def postprocess_encoding(self, raw_encoding: np.ndarray) -> np.ndarray:
        """This function postprocesses the raw encoding from the LLM

        Args:
            raw_encoding (_type_): Array of the raw encoding
        """
        if ENCODING_MODE == "full":
            # We just return the full raw encoding
            return raw_encoding
        elif ENCODING_MODE == "positional":
            # We select the max and return [normalized_max_position, max]
            normalized_idx = np.argmax(raw_encoding, axis=1) / raw_encoding.shape[1]
            max_values = np.max(raw_encoding, axis=1)
            encoding = np.array([max_values, normalized_idx]).swapaxes(1, 0)
            return encoding
        else:
            raise ValueError(f"Encoding {ENCODING_MODE} not implemented")
