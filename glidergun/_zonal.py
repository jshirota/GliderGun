from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Union, cast

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from glidergun._grid import Grid


@dataclass(frozen=True)
class Zonal:
    def zonal(self, func: Callable[[ndarray], Any], zone_grid: "Grid") -> "Grid":
        g = cast("Grid", self)
        zone_grid = zone_grid.type("int32")
        result = self
        for zone in set(zone_grid.data[np.isfinite(zone_grid.data)]):
            zone_value = int(zone + 0.5)
            data = g.set_nan(zone_grid != zone_value).data
            statistics = func(data[np.isfinite(data)])
            result = (zone_grid == zone_value).then(statistics, result)  # type: ignore
        return cast("Grid", result)

    def zonal_count(self, value: Union[float, int], zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.count_nonzero(a == value, **kwargs), zone_grid)

    def zonal_ptp(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.ptp(a, **kwargs), zone_grid)

    def zonal_median(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.median(a, **kwargs), zone_grid)

    def zonal_mean(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.mean(a, **kwargs), zone_grid)

    def zonal_std(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.std(a, **kwargs), zone_grid)

    def zonal_var(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.var(a, **kwargs), zone_grid)

    def zonal_min(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.min(a, **kwargs), zone_grid)

    def zonal_max(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.max(a, **kwargs), zone_grid)

    def zonal_sum(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.sum(a, **kwargs), zone_grid)
