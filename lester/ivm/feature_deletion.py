import torch
import numpy as np
import copy

from lester.ivm.artifacts import Artifacts
from lester.ivm.provenance import ProvenanceQueries


def _compute_updates(row_indexes, feature_ranges):
    updates = []
    for row_index in row_indexes:
        patches = []
        for feature_range in feature_ranges:
            start, end = feature_range
            length = end - start
            patch = (start, np.zeros(length))
            patches.append(patch)
        update = (row_index, patches)
        updates.append(update)
    return updates


def _compute_update_patches(X, y, updates):
    z_X = []
    updated_z_X = []
    z_y = []

    for row_index, patches in updates:
        sample = X[row_index, :]
        updated_sample = sample.detach().clone()
        for column_index, data in patches:
            column_start_index = column_index
            for i in range(0, len(data)):
                updated_sample[column_start_index + i] = data[i]

        z_X.append(sample)
        updated_z_X.append(updated_sample)
        z_y.append(y[row_index])

    z_X = torch.stack(z_X)
    updated_z_X = torch.stack(updated_z_X)
    z_y = torch.stack(z_y).reshape((-1,1))

    return z_X, updated_z_X, z_y


def _update_feature_matrix(X, updates):

    X_to_update = np.copy(X)

    for row_index, patches in updates:
        for column_index, data in patches:
            column_start_index = column_index
            for i in range(0, len(data)):
                X_to_update[row_index, column_start_index + i] = data[i]

    return X_to_update


def _update_train_data(artifacts, row_indexes, output_columns):

    train_df_to_update = artifacts.load_relational_train_data()

    train_column_indexes = [train_df_to_update.columns.get_loc(output_column)
                            for output_column in output_columns]

    for row_index in row_indexes:
        for column_index in train_column_indexes:
            train_df_to_update.iat[row_index, column_index] = np.nan

    return train_df_to_update


def _update_test_data(artifacts, row_indexes, output_columns):

    test_df_to_update = artifacts.load_relational_test_data()

    test_column_indexes = [test_df_to_update.columns.get_loc(output_column)
                           for output_column in output_columns]

    for row_index in row_indexes:
        for column_index in test_column_indexes:
            test_df_to_update.iat[row_index, column_index] = np.nan

    return test_df_to_update


def _compute_H_inv(X, theta):
    dot = torch.matmul(X, theta)
    probs = torch.sigmoid(dot)
    weighted_X = probs * (1 - probs) * X
    cov = torch.matmul(X.t(), weighted_X) + torch.eye(X.shape[1])
    cov_inv = torch.inverse(cov)
    return cov_inv


def _update_model(model_to_update, loss_fn, X, z_X, updated_z_X, z_y):

    loss_z_X = loss_fn(model_to_update(z_X), z_y)
    loss_updated_z_X = loss_fn(model_to_update(updated_z_X), z_y)

    gradients_z_X = torch.autograd.grad(loss_z_X, list(model_to_update.parameters()))
    gradients_updated_z_X = torch.autograd.grad(loss_updated_z_X, list(model_to_update.parameters()))

    gradient_differences = [gradient_updated_z_X - gradient_z_X
                            for (gradient_updated_z_X, gradient_z_X)
                            in zip(gradients_updated_z_X, gradients_z_X)]

    parameters = [parameter.data
                  for parameter
                  in model_to_update.parameters()]

    # Logreg specific code starting here
    theta = parameters[0].t()
    H_inv = _compute_H_inv(X, theta)

    gradient_difference = gradient_differences[0].t()

    delta_theta = -1.0 * torch.matmul(H_inv, gradient_difference)
    updated_theta = theta + delta_theta

    with torch.no_grad():
        model_to_update.linear.weight.copy_(updated_theta.t())

    return model_to_update


def delete_features(pipeline_name, run_id, source_name, source_column_name, primary_keys):

    artifacts = Artifacts(pipeline_name, run_id)
    prov_queries = ProvenanceQueries(artifacts)

    output_columns = prov_queries.output_columns(source_name, source_column_name)
    train_row_indexes = prov_queries.train_rows_originating_from(source_name, primary_keys)
    test_row_indexes = prov_queries.test_rows_originating_from(source_name, primary_keys)

    updated_train_data = _update_train_data(artifacts, train_row_indexes, output_columns)
    updated_test_data = _update_test_data(artifacts, test_row_indexes, output_columns)

    feature_ranges = prov_queries.feature_ranges(output_columns)

    train_updates = _compute_updates(train_row_indexes, feature_ranges)
    test_updates = _compute_updates(test_row_indexes, feature_ranges)

    X_train, y_train = artifacts.load_X_y_train()
    X_test, y_test = artifacts.load_X_y_test()

    updated_X_train = _update_feature_matrix(X_train, train_updates)
    updated_X_test = _update_feature_matrix(X_test, test_updates)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    model = artifacts.load_model()
    model_to_update = copy.deepcopy(model)
    loss_fn = torch.nn.BCELoss()

    z_X, updated_z_X, z_y = _compute_update_patches(X_train, y_train, train_updates)
    updated_model = _update_model(model_to_update, loss_fn, X_train, z_X, updated_z_X, z_y)
    return updated_train_data, updated_test_data, updated_X_train, updated_X_test, updated_model
