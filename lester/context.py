from dataclasses import dataclass, field
import enum


class LesterContext:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(LesterContext, cls).__new__(cls)
            cls._instance.prepare_function = None
            cls._instance.prepare_dialect = None
            cls._instance.prepare_sources = None
            cls._instance.split_function = None
            cls._instance.split_dialect = None
            cls._instance.encode_features_function = None
            cls._instance.encode_features_dialect = None
            cls._instance.encode_target_function = None
            cls._instance.encode_target_column = None
            cls._instance.model_training_function = None
            cls._instance.model_training_dialect = None
            cls._instance.source_counter = 0
        return cls._instance


class DataframeDialect(enum.Enum):
    PANDAS = 1
    PYSPARK = 2
    POLARS = 3


class EstimatorTransformerDialect(enum.Enum):
    SKLEARN = 1
    SPARKML = 2


@dataclass
class datasource:
    name: str
    track_provenance_by: list[str] = field(default_factory=list)


def prepare(*args, **kwargs):
    def inner(func):
        ctx = LesterContext()
        ctx.prepare_function = func
        ctx.prepare_dialect = kwargs['dialect']
        ctx.prepare_sources = kwargs['sources']
        return func
    return inner


def split(*args, **kwargs):
    def inner(func):
        ctx = LesterContext()
        ctx.split_function = func
        ctx.split_dialect = kwargs['dialect']
        return func
    return inner


def encode_features(*args, **kwargs):
    def inner(func):
        ctx = LesterContext()
        ctx.encode_features_function = func
        ctx.encode_features_dialect = kwargs['dialect']
        return func
    return inner


def encode_target(*args, **kwargs):
    def inner(func):
        ctx = LesterContext()
        ctx.encode_target_function = func
        ctx.encode_target_column = kwargs['target_column']
        return func
    return inner


def train_model(*args, **kwargs):
    def inner(func):
        ctx = LesterContext()
        ctx.model_training_function = func
        ctx.model_training_dialect = kwargs['dialect']
        return func
    return inner

