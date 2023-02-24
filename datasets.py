"""This module gathers two datasets from the openml open source repository."""
import pandas as pd
import openml

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder, RobustScaler
    from sklearn.utils import shuffle
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
        RandomForestRegressor, HistGradientBoostingRegressor
except ImportError:
    pass


def get_open_ml_dataset(open_ml_id, version=1):
    dataset = openml.datasets.get_dataset(open_ml_id, version=version, download_data=True)
    class_label = dataset.default_target_attribute
    x_data = dataset.get_data()[0]
    return x_data, class_label


class Adult:

    def __init__(
            self,
            version=2,
            random_seed=None,
            shuffle_dataset=False
    ):
        assert version in [1, 2], "OpenML census dataset version must be '1' or '2'."
        dataset, class_label = get_open_ml_dataset("adult", version=version)
        self.num_feature_names = ['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt']
        self.cat_feature_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'education-num']
        dataset[self.num_feature_names] = dataset[self.num_feature_names].apply(pd.to_numeric)
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ('ordinal_encoder', OrdinalEncoder()),
        ])
        column_transformer = ColumnTransformer([
            ('numerical', num_pipeline, self.num_feature_names),
            ('categorical', cat_pipeline, self.cat_feature_names),
        ], remainder='passthrough')
        col_names = self.num_feature_names + self.cat_feature_names
        col_names += [feature for feature in dataset.columns if feature not in col_names]
        dataset = pd.DataFrame(column_transformer.fit_transform(dataset), columns=col_names)
        dataset.dropna(inplace=True)

        if shuffle_dataset:
            dataset = shuffle(dataset, random_state=random_seed)

        self.x_data = dataset
        self.y_data = dataset.pop(class_label)

        self.feature_names = list(self.x_data.columns)
        self.n_features = len(self.feature_names)
        self.n_samples = len(self.x_data)


class BikeSharing:

    def __init__(
            self,
            random_seed=None,
            shuffle_dataset=False
    ):
        dataset, class_label = get_open_ml_dataset(42713, version=1)
        self.num_feature_names = ['hour', 'temp', 'feel_temp', 'humidity', 'windspeed']
        self.cat_feature_names = ['season', 'year', 'month', 'holiday', 'weekday', 'workingday', 'weather']
        dataset[self.num_feature_names] = dataset[self.num_feature_names].apply(pd.to_numeric)
        num_pipeline = Pipeline([
            ('scaler', RobustScaler())
        ])
        cat_pipeline = Pipeline([
            ('ordinal_encoder', OrdinalEncoder()),
        ])
        column_transformer = ColumnTransformer([
            ('numerical', num_pipeline, self.num_feature_names),
            ('categorical', cat_pipeline, self.cat_feature_names),
        ], remainder='passthrough')
        col_names = self.num_feature_names + self.cat_feature_names
        col_names += [feature for feature in dataset.columns if feature not in col_names]
        dataset = pd.DataFrame(column_transformer.fit_transform(dataset), columns=col_names)
        dataset.dropna(inplace=True)

        if shuffle_dataset:
            dataset = shuffle(dataset, random_state=random_seed)

        self.x_data = dataset
        self.y_data = dataset.pop(class_label)

        self.feature_names = list(self.x_data.columns)
        self.n_features = len(self.feature_names)
        self.n_samples = len(self.x_data)


if __name__ == "__main__":
    test_dataset = BikeSharing(random_seed=42, shuffle_dataset=False)
    print(f"n_samples:     {test_dataset.n_samples}")
    print(f"n_features:    {test_dataset.n_features}")
    print(f"feature_names: {test_dataset.feature_names}")

    x_data = test_dataset.x_data
    y_data = test_dataset.y_data

    model = RandomForestRegressor()
    model.fit(x_data, y_data)
    print(model.score(x_data, y_data))

    model = HistGradientBoostingRegressor()
    model.fit(x_data, y_data)
    print(model.score(x_data, y_data))