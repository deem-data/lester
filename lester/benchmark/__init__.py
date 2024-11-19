from abc import ABC, abstractmethod


class DataprepCodeTransformationTask(ABC):

    @property
    @abstractmethod
    def original_code(self):
        pass

    @property
    @abstractmethod
    def input_arg_names(self):
        pass

    @property
    @abstractmethod
    def input_schemas(self):
        pass

    @property
    @abstractmethod
    def output_columns(self):
        pass

    @abstractmethod
    def run_manually_rewritten_code(self, params):
        pass

    @abstractmethod
    def evaluate_transformed_code(self, transformed_code):
        pass


class FeaturisationCodeTransformationTask(ABC):

    @property
    @abstractmethod
    def original_code(self):
        pass

    @property
    @abstractmethod
    def input_schema(self):
        pass

    @abstractmethod
    def run_manually_rewritten_code(self, params):
        pass

    @abstractmethod
    def evaluate_transformed_code(self, transformed_code):
        pass


class ModelCodeTransformationTask(ABC):

    @property
    @abstractmethod
    def original_code(self):
        pass

    @abstractmethod
    def run_manually_rewritten_code(self, params):
        pass

    @abstractmethod
    def evaluate_transformed_code(self, transformed_code):
        pass