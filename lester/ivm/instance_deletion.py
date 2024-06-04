import torch
import numpy as np
import copy

from lester.ivm.artifacts import Artifacts
from lester.ivm.provenance import ProvenanceQueries


def _first_order_unlearning(model, X_to_unlearn, y_to_unlearn):

    model_to_update = copy.deepcopy(model)
    loss_fn = torch.nn.BCELoss()

    X_to_unlearn_torch = torch.from_numpy(X_to_unlearn).float()
    y_to_unlearn_torch = torch.from_numpy(y_to_unlearn).float().reshape(-1, 1)

    unlearning_rate = 1.0

    loss = loss_fn(model_to_update(X_to_unlearn_torch), y_to_unlearn_torch)
    gradients = torch.autograd.grad(loss, list(model_to_update.parameters()))

    with torch.no_grad():
        for parameter, gradient in zip(model_to_update.parameters(), gradients):
            parameter.data = parameter.data - unlearning_rate * gradient

    return model_to_update


def delete_instances(pipeline_name, run_id, source_name, primary_keys):

    artifacts = Artifacts(pipeline_name, run_id)

    prov_column = artifacts.provenance_column_name(source_name)

    train_prov = artifacts.load_train_provenance()
    train_indexes_to_delete = train_prov.index[train_prov[prov_column].isin(primary_keys)].tolist()
    train_indexes_to_retain = train_prov.index[~train_prov[prov_column].isin(primary_keys)].tolist()

    test_prov = artifacts.load_test_provenance()
    test_indexes_to_retain = test_prov.index[~test_prov[prov_column].isin(primary_keys)].tolist()

    train_data = artifacts.load_relational_train_data()
    X_train, y_train = artifacts.load_X_y_train()

    updated_train_data = train_data.iloc[train_indexes_to_retain]
    updated_train_prov = train_prov.iloc[train_indexes_to_retain]

    test_data = artifacts.load_relational_test_data()
    X_test, y_test = artifacts.load_X_y_test()

    updated_test_data = test_data.iloc[test_indexes_to_retain]
    updated_test_prov = test_prov.iloc[test_indexes_to_retain]

    updated_X_test = X_test[test_indexes_to_retain, :]
    updated_y_test = y_test[test_indexes_to_retain]

    X_to_unlearn = np.copy(X_train[train_indexes_to_delete,:])
    y_to_unlearn = np.copy(y_train[train_indexes_to_delete])

    updated_X_train = X_train[train_indexes_to_retain, :]
    updated_y_train = y_train[train_indexes_to_retain]

    model = artifacts.load_model()

    updated_model = _first_order_unlearning(model, X_to_unlearn, y_to_unlearn)

    return (updated_train_data, updated_train_prov, updated_test_data, updated_test_prov,
            updated_X_train, updated_y_train, updated_X_test, updated_y_test, updated_model)
