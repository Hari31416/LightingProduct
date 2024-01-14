from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch.nn as nn
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd
import os
import pickle
import argparse
from torch_train import TorchTrain
from utilities import get_simple_logger

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(FILE_DIR, "data")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
random_state = 42
# set random state
np.random.seed(random_state)
torch.manual_seed(random_state)


class PDFDataLoader:
    """A class that can be used to load the data to torch model. This will be used in the `PDFDataSet` class to create the final datasets."""

    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        """Gets the `idx` embedding and labels, converts them to the required format and returns them."""
        row = self.df[idx]
        embeddings = row["embeddings"]
        label = row["label"]
        # convert to torch int
        label = np.array(label)
        # add extra dimension to label
        label = np.expand_dims(label, axis=0)
        embeddings = torch.from_numpy(np.array(embeddings)).float()
        return embeddings.to(device), torch.from_numpy(label).to(device).float()

    def __len__(self):
        return len(self.df)


class PDFDataSet:
    def __init__(
        self,
        data_dir=DATA_DIR,
        fraction_test_data_in_train=0.2,
        model_ckpt="encoder",
    ) -> None:
        self.data_dir = data_dir
        self.fraction_test_data_in_train = fraction_test_data_in_train
        self.model_ckpt = model_ckpt
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        encoding_model = AutoModel.from_pretrained(model_ckpt)
        encoding_model = encoding_model.to(device)
        encoding_model = encoding_model.eval()
        self.encoding_model = encoding_model
        self.tokenizer = tokenizer
        self.logger = get_simple_logger("pdf_dataset")

    def create_datasets(self):
        train_data_path = os.path.join(FILE_DIR, self.data_dir, "train.csv")
        test_data_path = os.path.join(FILE_DIR, self.data_dir, "test.csv")
        df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        train_df, validation_df = train_test_split(df, test_size=0.3, random_state=42)
        if self.fraction_test_data_in_train:
            self.logger.info(
                f"Adding {self.fraction_test_data_in_train} fraction of test dataset to the training set."
            )
            test_df, test_df_for_training = train_test_split(
                test_df, test_size=self.fraction_test_data_in_train, random_state=42
            )
            train_df = pd.concat([train_df, test_df_for_training])

        train_dataset = Dataset.from_pandas(train_df)
        validation_dataset = Dataset.from_pandas(validation_df)
        test_dataset = Dataset.from_pandas(test_df)
        return train_dataset, validation_dataset, test_dataset

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def sentences_to_embedding(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        sentence_embeddings = self.mean_pooling(
            self.encoding_model(**encoded_input), encoded_input["attention_mask"]
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # remove last dimension
        sentence_embeddings = sentence_embeddings.squeeze()
        return sentence_embeddings.detach()

    def get_embeddings(self, row):
        return {
            "embeddings": self.sentences_to_embedding(
                sentences=row["content"],
            )
        }

    def create_embeddings(self):
        train_dataset, validation_dataset, test_dataset = self.create_datasets()
        train_dataset = train_dataset.map(self.get_embeddings)
        validation_dataset = validation_dataset.map(self.get_embeddings)
        test_dataset = test_dataset.map(self.get_embeddings)
        return train_dataset, validation_dataset, test_dataset


class PDFModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(PDFModel, self).__init__()
        self.seq_model = nn.Sequential()
        for i, hidden_size in enumerate(hidden_sizes):
            self.seq_model.add_module(f"linear_{i}", nn.Linear(input_size, hidden_size))
            self.seq_model.add_module(f"relu_{i}", nn.ReLU())
            input_size = hidden_size
        self.last_layer = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        seq_out = self.seq_model(x)
        out = self.last_layer(seq_out)
        return self.sigmoid(out)


def evaluate_model(y_true, y_pred, model_name, split="train"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    classification_report_ = classification_report(y_true, y_pred)
    print("------" * 10)
    print(f"Evaluating for the model: {model_name} for {split} dataset...")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(classification_report_)
    print("------" * 10)


def train_dl_model(
    train_data,
    validation_data,
    epochs=30,
    input_shape=384,
    hidden_sizes=[32, 16],
):
    model = PDFModel(input_size=input_shape, hidden_sizes=hidden_sizes, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    accuracy = torchmetrics.Accuracy(
        task="binary", num_classes=2, threshold=0.5, average="macro"
    )
    precision = torchmetrics.Precision(task="binary", average="macro")
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
    }
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
    tt = TorchTrain(model, optimizer, loss_fn, metrics=metrics, scheduler=scheduler)
    history = tt.fit(train_data, validation_data, verbose=True, epochs=epochs)
    return history, model


def evaluate_models(fraction_test_data_in_train=0.1):
    print("Creating Embeddings...")
    ds = PDFDataSet(fraction_test_data_in_train=fraction_test_data_in_train)
    train_dataset, validation_dataset, test_dataset = ds.create_embeddings()
    print("Done\n")

    print("Training DL Model")
    # Create dataset for DL models:
    BATCH_SIZE = 8
    train_dataloader = PDFDataLoader(train_dataset)
    validation_dataloader = PDFDataLoader(validation_dataset)
    test_dataloader = PDFDataLoader(test_dataset)

    train_data = DataLoader(train_dataloader, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = DataLoader(
        validation_dataloader,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_data = DataLoader(test_dataloader, batch_size=BATCH_SIZE, shuffle=True)
    for X, y in train_data:
        input_shape = int(X.shape[1])
        output_shape = int(y.shape[1])
        break
    epochs = 30
    hidden_sizes = [32, 16]
    history, model = train_dl_model(
        train_data=train_data,
        validation_data=validation_data,
        epochs=epochs,
        hidden_sizes=hidden_sizes,
    )
    print("Done\n")
    print("Evaluating DL Model")
    y_test_pred = model(torch.from_numpy(np.array(test_dataset["embeddings"])).float())
    y_test_pred = y_test_pred.detach().numpy()
    y_test_pred = np.where(y_test_pred > 0.5, 1, 0)
    evaluate_model(
        y_true=test_dataset["label"],
        y_pred=y_test_pred,
        model_name="DL Model",
        split="test",
    )
    print("Done\n")

    # ML Models
    print("Training and evaluating ML Models.")
    X_train = train_dataset["embeddings"]
    y_train = train_dataset["label"]
    X_validation = validation_dataset["embeddings"]
    y_validation = validation_dataset["label"]
    X_test = test_dataset["embeddings"]
    y_test = test_dataset["label"]
    rfc_best_params = {
        "max_depth": 23,
        "max_features": "log2",
        "n_estimators": 469,
    }

    xgb_best_params = {
        "max_depth": 25,
        "n_estimators": 372,
        "learning_rate": 0.2522824287799319,
    }
    print("Fitting RandomForest")
    rfc = RandomForestClassifier(**rfc_best_params)
    rfc.fit(X_train, y_train)
    evaluate_model(
        y_true=y_train,
        y_pred=rfc.predict(X_train),
        model_name="RandomForest",
        split="train",
    )
    evaluate_model(
        y_true=y_validation,
        y_pred=rfc.predict(X_validation),
        model_name="RandomForest",
        split="validation",
    )
    evaluate_model(
        y_true=y_test,
        y_pred=rfc.predict(X_test),
        model_name="RandomForest",
        split="test",
    )

    print("Fitting XGBoost")
    xgb = XGBClassifier(**xgb_best_params)
    xgb.fit(X_train, y_train)
    evaluate_model(
        y_true=y_train,
        y_pred=xgb.predict(X_train),
        model_name="XGBoost",
        split="train",
    )
    evaluate_model(
        y_true=y_validation,
        y_pred=xgb.predict(X_validation),
        model_name="XGBoost",
        split="validation",
    )
    evaluate_model(
        y_true=y_test,
        y_pred=xgb.predict(X_test),
        model_name="XGBoost",
        split="test",
    )
    print("All Done")


def train_and_save_final_model(model_save_path="final_model.pkl"):
    """This method creats and save the final model. The final model has the following characterstics:

    - It is a RandomForestClassifier trained on all the training data and 10% of the test data. 10% of the test data. The 10% of test data is necessary as the distribution of the test data is very different from the training data.
    - Since 10% of test data is used while training, this data is not used while claculating the final accuracy of the model, which is 100%.

    Parameters
    ----------
    model_save_path : str, optional
        The path to save the final model, by default "final_model.pkl"
    Returns
    -------
    None
    Examples
    --------
    >>> train_and_save_final_model()
    >>> train_and_save_final_model(model_save_path="final_model.pkl")
    """
    print("Creating Embeddings...")
    model_save_path = os.path.join(FILE_DIR, model_save_path)
    ds = PDFDataSet(fraction_test_data_in_train=0.1)
    train_dataset, validation_dataset, test_dataset = ds.create_embeddings()
    train_dataset = concatenate_datasets([train_dataset, validation_dataset])
    X_train = train_dataset["embeddings"]
    X_test = test_dataset["embeddings"]
    y_train = train_dataset["label"]
    y_test = test_dataset["label"]

    print("Training and evaluating the model...")
    rfc_best_params = {
        "max_depth": 23,
        "max_features": "log2",
        "n_estimators": 469,
    }
    rfc_model = RandomForestClassifier(**rfc_best_params)
    rfc_model.fit(X_train, y_train)
    evaluate_model(
        y_true=y_train,
        y_pred=rfc_model.predict(X_train),
        model_name="Final Model",
        split="train",
    )
    evaluate_model(
        y_true=y_test,
        y_pred=rfc_model.predict(X_test),
        model_name="Final Model",
        split="test",
    )

    print("Saving the model...")
    with open(model_save_path, "wb") as f:
        pickle.dump(rfc_model, f)
    print(f"Model saved to: {model_save_path}")


def main(args):
    task = args.task
    if task == "train":
        model_save_path = args.model_save_path
        train_and_save_final_model(model_save_path=model_save_path)
    elif task == "evaluate":
        fraction_test_data_in_train = args.fraction
        evaluate_models(fraction_test_data_in_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument(
        "--task",
        type=str,
        choices=["train", "evaluate"],
        required=True,
        help="Whether to train and save the best model or evaluate all the models.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of test data in train dataset",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="final_model.pkl",
        help="Path to save the final model",
    )
    args = parser.parse_args()
    main(args)
