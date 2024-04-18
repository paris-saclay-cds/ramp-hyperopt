import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class DataExtension(Enum):
    CSV = ".csv"
    PKL = ".pkl"
    TXT = ".txt"
    NPY = ".npy"


@dataclass
class DataDescription:
    features: List
    num_features: int
    target_name: List
    description: str
    feature_types: Dict[str, str]
    target_types: Dict[str, str]


@dataclass
class MetaData:
    title: str
    aux_data_names: List[str]
    aux_data_formats: List[DataExtension]
    raw_description: str
    task_description: str
    task_type: str
    data_description: DataDescription
    metric_path: Optional[str]
    score_name: Optional[str]
    metric_description: str
    positive_class_name: str
    id_name: str = "id"
    train_data_name: str = "train_X"
    train_data_format: DataExtension = DataExtension.CSV
    test_data_name: str = "test_X"
    test_data_format: DataExtension = DataExtension.CSV
    train_target_name: Optional[str] = "train_y"
    train_target_format: Optional[DataExtension] = DataExtension.CSV
    test_target_name: Optional[str] = "test_y"
    test_target_format: Optional[DataExtension] = DataExtension.CSV

    def __post_init__(self):
        """Used to force any format that is not an instance of DataExtension, into it"""
        if not isinstance(self.train_data_format, DataExtension):
            self.train_data_format = DataExtension(self.train_data_format)
        if not isinstance(self.train_target_format, DataExtension):
            self.train_target_format = DataExtension(self.train_target_format)
        if not isinstance(self.test_data_format, DataExtension):
            self.test_data_format = DataExtension(self.test_data_format)
        if not isinstance(self.test_target_format, DataExtension):
            self.test_target_format = DataExtension(self.test_target_format)

    def save(self, save_path: str | Path):
        """Save the metadata as a json

        Args:
            save_path (str | Path): save path
        """
        save_path = Path(save_path)
        metadata_dict = asdict(self)
        metadata_dict["data_description"] = asdict(metadata_dict["data_description"])
        for key in metadata_dict:
            if isinstance(metadata_dict[key], DataExtension):
                metadata_dict[key] = metadata_dict[key].value

        json.dump(
            obj=metadata_dict,
            indent=2,
            fp=open(
                save_path / "metadata.json",
                "w",
            ),
        )
