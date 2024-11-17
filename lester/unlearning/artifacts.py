import pandas as pd
import numpy as np
import json
import torch


class Artifacts:
    def __init__(self, pipeline_name, run_id):
        self.artifact_path = f'.lester/{pipeline_name}/{run_id}'
        with open(f'{self.artifact_path}/column_provenance.json') as json_file:
            self.column_provenance = json.load(json_file)
        with open(f'{self.artifact_path}/matrix_column_provenance.json') as json_file:
            self.matrix_column_provenance = json.load(json_file)

    def load_relational_train_data(self):
        return pd.read_parquet(f'{self.artifact_path}/train.parquet')

    def load_relational_test_data(self):
        return pd.read_parquet(f'{self.artifact_path}/test.parquet')

    def provenance_column_name(self, source_name):
        return f"__lester_provenance_{source_name}"

    def load_train_provenance(self):
        return pd.read_parquet(f'{self.artifact_path}/row_provenance_X_train.parquet')

    def load_test_provenance(self):
        return pd.read_parquet(f'{self.artifact_path}/row_provenance_X_test.parquet')

    def load_X_y_train(self):
        X_train = np.load(f'{self.artifact_path}/X_train.npy')
        y_train = np.load(f'{self.artifact_path}/y_train.npy')
        return X_train, y_train

    def load_X_y_test(self):
        X_test = np.load(f'{self.artifact_path}/X_test.npy')
        y_test = np.load(f'{self.artifact_path}/y_test.npy')
        return X_test, y_test

    def load_y_pred(self):
        return np.load(f'{self.artifact_path}/y_pred.npy')

    def load_model(self):
        model = torch.load(f'{self.artifact_path}/model.pt')
        model.eval()
        return model
