import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightgbm import Dataset, LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from src.config.navigate import project_router
from src.train.metrics import rmse


@dataclass
class GBModuleData:
    library_name: str
    file_extension: str
    short_title: str


class GBModule(Enum):
    CATBOOST = GBModuleData("catboost", ".cbm", "cb")
    XGBOOST = GBModuleData("xgboost", ".model", "xgb")
    LIGHTGBM = GBModuleData("lightgbm", ".txt", "lgbm")

    def __init__(self, data):
        self.library_name = data.library_name
        self.file_extension = data.file_extension
        self.short_title = data.short_title

    @staticmethod
    def define_gb_module(algorithm):
        if "catboost" in algorithm.__module__:
            return GBModule.CATBOOST
        if "xgboost" in algorithm.__module__:
            return GBModule.XGBOOST
        if "lightgbm" in algorithm.__module__:
            return GBModule.LIGHTGBM
        raise ValueError("Algorithm not identified")


def save_model(
    model,
    score: Optional[float] = None,
    stor_path: Path = project_router.stor_path,
    create_dir: bool = False,
):
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    gb_module = GBModule.define_gb_module(model)

    info_string = f"{dt}-{gb_module.short_title}"
    if score is not None:
        info_string += f"-{score:.4f}"

    models_path = stor_path / "models"
    if create_dir:
        models_path = (
            models_path / info_string
        )  # /stor/models/cb-{datetime}-{score}.cbm
    models_path.mkdir(parents=True, exist_ok=True)

    filepath = models_path / (info_string + gb_module.file_extension)
    model.save_model(filepath)
    return models_path if create_dir else filepath


class Trainer(object):
    def __init__(
        self,
        algorithm,
        storage_dir: Path = project_router.stor_path,
        init_params: Optional[Union[Path, dict]] = None,
        fit_params: Optional[Union[Path, dict]] = None,
        cat_features: Optional[list[str]] = None,
        scoring_func: Callable = rmse,
        n_splits: int = 3,
        random_seed: int = 19,
    ):
        self.algorithm = algorithm
        self.gb_module = GBModule.define_gb_module(algorithm)
        self.storage_dir = storage_dir
        self.init_params = self._load_if_path(init_params)
        self.fit_params = self._load_if_path(fit_params)
        self.cat_features = cat_features
        self.scoring_func = scoring_func
        self.n_splits = n_splits
        self.random_seed = random_seed
        self._check_storage_dir()

        self.model_best = None
        self.score_best = None

    def _load_if_path(self, target: Path | dict | None):
        if target is None:
            return dict()
        if isinstance(target, Path):
            return self._load_json(target)
        return target

    def _load_json(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def _check_storage_dir(self):
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)

    def _save_json(self, data: dict, path: Path):
        return path.write_text(
            json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8"
        )

    def _save_params(self, folder: Path):
        self._save_json(self.init_params, folder / "init_params.json")
        self._save_json(self.fit_params, folder / "fit_params.json")

    def train(self, X, y):
        # Perform stratified k-fold cross-validation
        folding = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_seed
        )
        models = []
        scores = []

        for fold, (train_index, val_index) in enumerate(folding.split(X, y)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            fit_func = {
                GBModule.CATBOOST: self._fit_catboost,
                GBModule.XGBOOST: self._fit_xgboost,
                GBModule.LIGHTGBM: self._fit_lightgbm,
            }.get(self.gb_module)
            if fit_func is None:
                raise ValueError(f"No fit function for {self.gb_module}")
            model = fit_func(X_train, X_val, y_train, y_val)
            score = self.score_model(model, X_val, y_val)

            models.append(model)
            scores.append(score)

        idx_best = np.argmin(scores)
        score_best = scores[idx_best]
        model_best = models[idx_best]
        model_dir = save_model(model_best, score=score_best, create_dir=True)
        self._save_params(model_dir)
        # self.save_model(model_best, score_best)
        self.score_best = score_best
        self.model_best = model_best
        return model_best, score_best
        # if self.gb_module == GBModule.CATBOOST:
        #     self._fit_catboost(X_train, X_val, y_train, y_val, fold)
        # elif self.gb_module == GBModule.XGBOOST:
        #     self._fit_xgboost(X_train, X_val, y_train, y_val, fold)
        # elif self.gb_module == GBModule.LIGHTGBM:
        #     self._fit_lightgbm(X_train, X_val, y_train, y_val, fold)

    def _fit_catboost(self, X_train, X_val, y_train, y_val):
        model: CatBoostRegressor | CatBoostClassifier = self.algorithm(
            **self.init_params
        )
        dtrain = Pool(X_train, y_train, cat_features=self.cat_features)
        dval = Pool(X_val, y_val, cat_features=self.cat_features)
        model.fit(dtrain, eval_set=dval, **self.fit_params)
        return model

    def _fit_xgboost(self, X_train, X_val, y_train, y_val, fold):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = self.init_params
        params["eval_metric"] = "logloss"

        model = xgb.train(
            params,
            dtrain,
            evals=[(dval, "eval")],
        )

        model_path = self.storage_dir / f"xgboost_model_fold_{fold}.model"
        # ModelHandler.save_model(model)
        model.save_model(str(model_path))

    def _fit_lightgbm(self, X_train, X_val, y_train, y_val, fold):
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = self.init_params
        params["metric"] = "binary_logloss"

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
        )

        model_path = self.storage_dir / f"lightgbm_model_fold_{fold}.txt"
        model.save_model(str(model_path))

    def score_model(self, model, X, y_true):
        y_predict = model.predict(X)
        return self.scoring_func(y_true, y_predict)


def train_model(
    algorithm,
    X,
    y,
    early_stopping_rounds=500,
    init_params=None,
    cat_features=None,
    n_splits=3,
    random_seed=2023,
):
    scores = []
    models = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    print(f"========= TRAINING {algorithm.__name__} =========")

    for num_fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        if init_params is not None:
            model = algorithm(**init_params)
        else:
            model = algorithm()

        if algorithm.__name__ == "CatBoostRegressor":
            # Используйте соответствующий класс
            dtrain = Pool(X_train, y_train, cat_features)
            dval = Pool(X_val, y_val, cat_features)

            model.fit(
                dtrain,
                eval_set=dval,
                verbose=0,
                early_stopping_rounds=early_stopping_rounds,
            )

        elif algorithm.__name__ == "LGBMRegressor":
            # Используйте соответствующий класс
            dtrain = lgb.Dataset(X_train, y_train)
            dval = lgb.Dataset(X_val, y_val)

            model = lgb.train(
                params=init_params,
                train_set=dtrain,
                valid_sets=dval,
                categorical_feature=cat_features,
            )

        elif algorithm.__name__ == "XGBRegressor":
            # Используйте соответствующий класс
            dtrain = xgb.DMatrix(X_train, X_val)
            dval = xgb.DMatrix(X_val, y_val)

            model = xgb.train(
                params=init_params,
                dtrain=dtrain,
                evals=[(dtrain, "dtrain"), (dval, "dtest")],
                verbose_eval=False,
                early_stopping_rounds=early_stopping_rounds,
            )

        # Сделайте предсказание на X_val и посчитайте RMSE
        y_pred = model.predict(dval)
        score = np.sqrt(mean_squared_error(y_val, y_pred))

        models.append(model)
        scores.append(score)

        print(f"FOLD {num_fold}: SCORE {score}")

    mean_kfold_score = np.mean(scores, dtype="float16") - np.std(
        scores, dtype="float16"
    )
    print("\nMEAN RMSE SCORE", mean_kfold_score)

    # Выберите модель с наименьшим значением скора
    best_model = models[np.argmin(scores)]

    return mean_kfold_score, best_model


def tuning_hyperparams(
    algorithm,
    X,
    y,
    init_params,
    fit_params,
    grid_params,
    n_iter,
    cv=3,
    random_state=2023,
):
    estimator = algorithm(**init_params)

    # Можно использоавть GridSearchCV
    model = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid_params,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
        random_state=random_state,
    )

    model.fit(X, y, **fit_params)

    return model.best_params_ | init_params
