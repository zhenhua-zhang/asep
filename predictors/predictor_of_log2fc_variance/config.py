"""I'm docstrting

"""

from collections import defaultdict

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge


estimators_list = [
    ('imputator', SimpleImputer()),
    # ('standard_scaler', StandardScaler()),
    # ('normalizer', Normalizer()),
    ('feature_selection', SelectKBest()),

    # Elastic net regressor, a combination of l1 and l2 regularization
    # ('en', ElasticNet()),

    # Gaussian process regression
    # ('gpr', GaussianProcessRegressor())

    # Kernel ridge regression
    ('krr', KernelRidge())

    # Neural network, multi-layer perceptron regressor
    # ('mlp', MLPRegressor()),

    # Ada boost regressor
    # ('adb', AdaBoostRegressor()),

    # Random forest regressor
    # ('rfr', RandomForestRegressor(n_estimators=20))

    # Gradient boosting regressor
    # ('gbr', GradientBoostingRegressor())
]

grid_search_opt_params = defaultdict(None)
grid_search_opt_params.update(
    dict(
        cv=5,
        n_jobs=3,
        refit='r2',
        iid=False,
        scoring=dict(
            r2='r2', ev='explained_variance', nmae='neg_mean_absolute_error',
            nmse='neg_mean_squared_error', nmdae='neg_median_absolute_error'
        ),
        param_grid=[
            dict(
                imputator__strategy=['mean'],
                feature_selection__score_func=[mutual_info_regression],
                feature_selection__k=list(range(2, 20, 2)),

                krr__alpha=[1e0, 1e-3],
                # krr__kernel=[ExpSineSquared(0.2, 0.5)],
                # gpr__kernel=[WhiteKernel(1e-1) +
                # ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1]))],
                # gbr__min_samples_split=[2, 5, 10, 20],
                # rfr__min_samples_split=[2, 5, 10, 20],
                # en__max_iter=[100],
                # mlp__hidden_layer_sizes=[75, 100, 125],
                # mlp__alpha=[0.0001, 0.001, 0.01],
                # mlp__max_iter=[2000]
            ),
        ],
        return_train_score=True,
    )
)

random_search_opt_params = defaultdict(None)
random_search_opt_params.update(
    dict(
        cv=5,
        n_jobs=3,
        refit='ev',
        n_iters=10,
        iid=False,  # To supress warnings
        scoring=dict(
            r2='r2', ev='explained_variance', nmae='neg_mean_absolute_error',
            nmse='neg_mean_squared_error', nmdae='neg_median_absolute_error'
        ),
        param_distribution=[
            dict(
                transformer__imputator__strategy=['mean'],
                transformer__indicators__features=['missing-only'],
                transformer__indicators__error_on_new=[False],
                en__max_iter=[100],

                # mlp__hidden_layer_sizes=[75, 100, 125],
                # mlp__alpha=[0.0001, 0.001, 0.01], mlp__max_iter=[300]
            ),
        ],
        return_train_score=True,  # to supress a warning
    )
)
