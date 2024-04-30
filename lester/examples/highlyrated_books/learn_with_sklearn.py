from lester.context import train_model, EstimatorTransformerDialect

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


@train_model(dialect=EstimatorTransformerDialect.SKLEARN)
def logreg_with_hpo():
    param_grid = {
        'penalty': ['l2', 'l1'],
        'alpha': [0.0001, 0.001, 0.01],
    }

    learner = SGDClassifier(loss='log_loss', max_iter=1000)
    search = GridSearchCV(learner, param_grid, cv=5, verbose=1, n_jobs=-1)

    return search
