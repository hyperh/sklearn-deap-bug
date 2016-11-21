import itertools
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from pipeline.models import getModels
from pipeline.dimTransformers import getDimTransformers


def run(features, labels, hyperOpt):
    cv = list(TimeSeriesSplit(n_splits=10).split(features))

    models = getModels('model')
    dimTransformers = getDimTransformers(
        'dimTransform',
        numFeatures=features.shape[1]
    )

    pipe = Pipeline(steps=[
        # Need to initialize to s/t, just use the first one
        ('dimTransform', dimTransformers[0]['dimTransform'][0]),
        ('model', models[0]['model'][0]),
    ])

    param_grid = [
        {**m, **d} for m, d in itertools.product(models, dimTransformers)
    ]

    grid = hyperOpt(
        estimator=pipe,
        params=param_grid,
        cv=cv
    )
    grid.fit(features, labels)

    print('\nBest estimator: {}'.format(grid.best_estimator_))
    print('Best params: {}'.format(grid.best_params_))
    print('Best score: {}\n'.format(grid.best_score_))
