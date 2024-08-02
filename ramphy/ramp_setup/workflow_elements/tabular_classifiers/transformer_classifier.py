import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ramphy import Hyperparameter
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

num_layers = Hyperparameter(dtype="int", default=1, values=[1, 2, 3, 5])
num_heads = Hyperparameter(dtype="int", default=1, values=[1, 2, 5, 10])
ff_size = Hyperparameter(dtype="int", default=128, values=[128, 256, 512, 1024, 2056])
activation = Hyperparameter(dtype="str", default="gelu", values=["gelu", "relu"])
num_epochs = Hyperparameter(dtype="int", default=100, values=[50, 100, 300])  # Maybe not necessary
input_scaling = Hyperparameter(dtype="str", default="minmax", values=["standard", "minmax"])

INPUT_SCALING = str(input_scaling)
NUM_LAYERS = int(num_layers)
NUM_HEADS = int(num_heads)
FF_SIZE = int(ff_size)
ACTIVATION = str(activation)
NUM_EPOCHS = int(num_epochs)
LEARNING_RATE = 0.001
BATCH_SIZE = 10


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_size: int,
        num_heads: int,
        ff_size: int,
        output_size: int,
        softmax_out: bool = False,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if activation not in ["relu", "gelu"]:
            raise ValueError("Only use Relu or Gelu")

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=ff_size,
            batch_first=True,
            norm_first=True,  # Apparently this is better
            activation=activation,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(in_features=input_size, out_features=output_size)
        self.softmax_out = softmax_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_embeddings = self.transformer(x)
        output = self.output_layer(out_embeddings)
        if self.softmax_out:
            output = F.softmax(output)
        return output


class Classifier(BaseEstimator):
    def __init__(self, metadata: dict):
        self.metadata = metadata

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Prepare everything
        # ---------------------------
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        feat_size = X.shape[1]
        output_size = len(self.metadata["data_description"]["target_cols"])
        if output_size > 1:
            raise NotImplementedError("Multi-output classification is not yet supported.")
        self.criterion = nn.CrossEntropyLoss()
        softmax_out = False  # No softmax as from here https://jaykmody.com/blog/gpt-from-scratch

        writer = SummaryWriter(log_dir="./tensorboard")

        # Scale the features
        if INPUT_SCALING == "standard":
            self.feature_scaler = StandardScaler()
        elif INPUT_SCALING == "minmax":
            self.feature_scaler = MinMaxScaler()
        else:
            ValueError(f"Only minmax or standard scaling for features. {INPUT_SCALING} is not implemented")
        X = self.feature_scaler.fit_transform(X)

        X = torch.Tensor(X).to(self.device)  # type: ignore
        y = torch.Tensor(y).to(torch.int64).to(self.device)  # type: ignore

        self.transformer = Transformer(
            num_layers=NUM_LAYERS,
            input_size=feat_size,
            num_heads=NUM_HEADS,
            ff_size=FF_SIZE,
            activation=ACTIVATION,
            output_size=output_size,
            softmax_out=softmax_out,
        ).to(self.device)

        optimizer = optim.Adam(self.transformer.parameters(), lr=LEARNING_RATE)
        # ---------------------------

        # Train
        # ---------------------------
        training_steps = 0
        for epoch in tqdm(range(NUM_EPOCHS), desc="Training Epoch"):
            epoch_loss = 0
            input_ids = np.arange(len(X))
            np.random.shuffle(input_ids)
            batch_count = 0
            for batch_idx in range(0, len(X), BATCH_SIZE):
                batch_X = X[batch_idx : batch_idx + BATCH_SIZE]
                batch_y = y[batch_idx : batch_idx + BATCH_SIZE]

                optimizer.zero_grad()

                output = self.transformer(batch_X)
                loss = self.criterion(input=output, target=batch_y)
                loss.backward()
                optimizer.step()

                # Log
                epoch_loss += loss.detach().cpu().numpy()
                writer.add_scalar("Loss/Batch_loss", loss.item(), training_steps)
                training_steps += 1
                batch_count += 1
            epoch_loss = epoch_loss / batch_count
            writer.add_scalar("Loss/Training_loss", epoch_loss, epoch)
            print(f"Epoch Loss: {epoch_loss}")
        print()
        # ---------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.get_logits(X=X)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def get_logits(self, X: np.ndarray) -> np.ndarray:
        X = self.feature_scaler.transform(X)  # type: ignore
        with torch.no_grad():
            X = torch.Tensor(X).to(self.device)  # type: ignore
            y_pred = self.transformer(X).detach().cpu().numpy()
        return y_pred
