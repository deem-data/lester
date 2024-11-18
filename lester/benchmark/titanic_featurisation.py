# https://github.com/josephmisiti/kaggle-titanic/blob/master/Titanic%20Classification.ipynb
from lester.benchmark import FeaturisationCodeTransformationTask


class TitanicFeaturisationTask(FeaturisationCodeTransformationTask):

    @property
    def original_code(self):
        return """
import pandas as pd
import numpy as np

import sklearn
from sklearn import preprocessing

# drop the useless columns that we know are not going to be good for prediction
df.drop(["Name","Ticket","Cabin"], axis=1, inplace=True)

label_encoder = preprocessing.LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

df.drop(['Survived'], axis=1, inplace=True)

X = df.as_matrix().astype(np.float)

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)        
"""

    @property
    def input_schema(self):
        return ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
                'Ticket', 'Fare', 'Cabin', 'Embarked']

    def run_manually_rewritten_code(self, params):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import StandardScaler

        featuriser = ColumnTransformer(transformers=[
            ('categorical', OneHotEncoder(), ['Sex', 'Embarked']),
            ('numeric', StandardScaler(), ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ])

        return featuriser

    def evaluate_transformed_code(self, transformed_code):
        variables_for_exec = {}
        exec(transformed_code, variables_for_exec)

        import pandas as pd
        prepared_data = pd.read_csv("data/benchmark/titanic_featurisation.csv")

        generated_column_transformer = variables_for_exec['__featurise']()
        generated_X = generated_column_transformer.fit_transform(prepared_data)

        manual_column_transformer = self.run_manually_rewritten_code({})
        manual_X = manual_column_transformer.fit_transform(prepared_data)

        assert generated_X.shape[0] == manual_X.shape[0]

        # Generated code is allowed to add missing value imputation
        feature_diff = manual_X.shape[1] - generated_X.shape[1]
        assert feature_diff >= 0
        assert feature_diff <= 2
