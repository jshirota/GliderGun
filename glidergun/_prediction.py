import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar, Union, cast

import numpy as np

from glidergun._literals import DataType, PredictorType

if TYPE_CHECKING:
    from glidergun._grid import Grid


class Predictor(Protocol):
    fit: Callable
    score: Callable
    predict: Callable


TPredictor = TypeVar("TPredictor", bound=Predictor)


@dataclass(frozen=True)
class Prediction:
    def fit(
        self,
        model: Union[TPredictor, PredictorType],
        *explanatory_grids: "Grid",
        **kwargs: Any,
    ):
        if model == "linear_regression":
            from sklearn.linear_model import LinearRegression

            actual_model = LinearRegression()
        elif model == "polynomial_regression":
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import PolynomialFeatures

            actual_model = Pipeline(
                [
                    (
                        "polynomial_features",
                        PolynomialFeatures(degree=2, include_bias=False),
                    ),
                    ("linear_regression", LinearRegression()),
                ]
            )
        elif model == "random_forest_classifier":
            from sklearn.ensemble import RandomForestClassifier

            actual_model = RandomForestClassifier()
        elif model == "random_forest_regression":
            from sklearn.ensemble import RandomForestRegressor

            actual_model = RandomForestRegressor()
        elif isinstance(model, str):
            raise ValueError(f"'{model}' is not a supported model.")
        else:
            actual_model = model

        return GridPredictor(actual_model).fit(
            cast("Grid", self), *explanatory_grids, **kwargs
        )


class GridPredictor(Generic[TPredictor]):
    def __init__(self, model: TPredictor) -> None:
        self.model: TPredictor = model
        self._dtype: DataType = "float32"

    def fit(self, dependent_grid: "Grid", *explanatory_grids: "Grid", **kwargs: Any):
        grids = self._flatten(*[dependent_grid, *explanatory_grids])
        head, *tail = grids
        self.model = self.model.fit(
            np.array([g.data.ravel() for g in tail]).transpose(1, 0),
            head.data.ravel(),
            **kwargs,
        )
        self._dtype = dependent_grid.dtype
        return self

    def score(self, dependent_grid: "Grid", *explanatory_grids: "Grid") -> float:
        head, *tail = self._flatten(dependent_grid, *explanatory_grids)
        return self.model.score(
            np.array([g.data.ravel() for g in tail]).transpose(1, 0), head.data.ravel()
        )

    def predict(self, *explanatory_grids: "Grid", **kwargs: Any) -> "Grid":
        grids = self._flatten(*explanatory_grids)
        array = self.model.predict(
            np.array([g.data.ravel() for g in grids]).transpose(1, 0), **kwargs
        )
        g = grids[0]
        return g.local(array.reshape((g.height, g.width))).type(self._dtype)

    def _flatten(self, *grids: "Grid"):
        return [
            g.is_nan().then(float(g.mean), g) for g in grids[0].standardize(*grids[1:])
        ]

    def save(self, file: str):
        with open(file, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, file: str):
        with open(file, "rb") as f:
            return GridPredictor(pickle.load(f))


def load_model(file: str) -> GridPredictor[Any]:
    return GridPredictor.load(file)
