import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import lester as lt
import torch
from neuralnet import custom_mlp
from lester.classification import run_pipeline
from generated_pipeline_code import _lester_dataprep, encode_features, extract_label
from generated_pipeline_code import *

lt.make_accessible(locals(), globals())

source_paths = {
    'customers_path': 'data/synthetic_customers_100.csv',
    'mails_path': 'data/synthetic_mails_100.csv'
}

run_pipeline("lester-gen", source_paths, _lester_dataprep, encode_features, extract_label, custom_mlp)
