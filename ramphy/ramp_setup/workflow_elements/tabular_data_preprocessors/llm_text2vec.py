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
encoding_mode = Hyperparameter(dtype="str", default="positional", values=["full", "positional"])

PEFT_MODEL = str(peft_model)
POOLING_MODE = str(pooling_mode)
ENCODING_MODE = str(encoding_mode)


class DataPreprocessor(rs.BaseDataPreprocessor):
    def __init__(self):
        device_map = "cuda:0"
        self.tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", device_map=device_map)
        self.config = AutoConfig.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", trust_remote_code=True, device_map=device_map
        )
        self.model = AutoModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
            trust_remote_code=True,
            config=self.config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )

        self.model = PeftModel.from_pretrained(
            self.model, "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp", device_map=device_map
        )
        self.model = self.model.merge_and_unload()
        self.model = PeftModel.from_pretrained(
            self.model, f"McGill-NLP/LLM2Vec-Sheared-LLaMA-{PEFT_MODEL}", device_map=device_map
        )
        self.l2v = LLM2Vec(self.model, self.tokenizer, pooling_mode=POOLING_MODE, max_length=512)

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        # Find text columns
        feature_types = metadata["data_description"]["feature_types"]
        text_columns = []
        for feature in feature_types:
            if feature_types[feature] == "text":
                text_columns.append(feature)

        # Encode columns
        new_feature_names = {}
        for feature in text_columns:
            encoded_columns = self.encode_column(column_name=feature, dataset=X_train)
            X_train = pd.concat([X_train, encoded_columns], ignore_index=True)
            new_feature_names[feature] = list(encoded_columns.columns)

            encoded_columns = self.encode_column(column_name=feature, dataset=X_test)
            X_test = pd.concat([X_test, encoded_columns], ignore_index=True)

        # Drop text columns from datasets
        X_train.drop(columns=text_columns)
        X_test.drop(columns=text_columns)

        # Update metadata
        metadata = deepcopy(metadata)
        for feat in text_columns:
            try:
                metadata["data_description"]["feature_values"].pop(feat)
            except KeyError:
                continue
            try:
                metadata["data_description"]["feature_types"].pop(feat)
            except KeyError:
                continue
            missing_count = None
            if feat in metadata["data_description"]["missing_data_count"]:
                missing_count = metadata["data_description"]["missing_data_count"][feat]
                metadata["data_description"]["missing_data_count"].pop(feat)

            for new_column in new_feature_names[feat]:
                metadata["data_description"]["feature_types"][new_column] = "num"
                if missing_count is not None:
                    metadata["data_description"]["missing_data_count"][new_column] = missing_count

        return X_train, y_train, X_test, metadata

    def encode_column(self, column_name: str, dataset: pd.DataFrame) -> pd.DataFrame:
        """This function encodes the given column from the given dataset.

        Args:
            column_name (str): _description_
            dataset (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: The dataframe of the encoded columns
        """
        instruction = f"Given the text feature named {column_name} of a tabular dataset, extract the corresponding value in the given example."
        values = dataset[column_name].values

        queries = [[instruction, f"{column_name}: {val}"] for val in values]
        raw_encoding = np.array(self.l2v.encode(queries, convert_to_numpy=True))
        encoding = self.postprocess_encoding(raw_encoding)
        new_col_names = [f"{column_name}_{idx}" for idx in range(encoding.shape[1])]
        encoded_columns = pd.DataFrame(data=encoding, columns=new_col_names)
        return encoded_columns

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
