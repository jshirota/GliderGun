from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, overload

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds

from glidergun._grid import Grid, grid
from glidergun._stack import Stack, stack
from glidergun._types import Extent


@dataclass
class Profile:
    count: int
    crs: CRS
    height: int
    width: int
    transform: Affine


class Mosaic:
    def __init__(self, *files: str) -> None:
        assert files, "No files provided"
        profiles = list(self._read_profiles(*files))
        count_set = {p.count for _, p in profiles}
        crs_set = {p.crs for _, p in profiles}
        assert len(count_set) == 1, "Inconsistent number of bands"
        assert len(crs_set) == 1, "Inconsistent CRS"
        self.crs = crs_set.pop()
        self.files: Dict[str, Extent] = {
            f: Extent(*array_bounds(p.height, p.width, p.transform))
            for f, p in profiles
        }
        self.extent = Extent(
            min(e.xmin for e in self.files.values()),
            min(e.ymin for e in self.files.values()),
            max(e.xmax for e in self.files.values()),
            max(e.ymax for e in self.files.values()),
        )

    def _read_profiles(self, *files: str):
        for f in files:
            with rasterio.open(f) as dataset:
                yield f, SimpleNamespace(**dataset.profile)

    def _read(self, extent: Tuple[float, float, float, float], index: int):
        for f, e in self.files.items():
            try:
                if e.intersects(*extent):
                    yield grid(f, extent, index=index)
            except ValueError:
                pass

    def tiles(
        self,
        width: float,
        height: float,
        clip_extent: Optional[Tuple[float, float, float, float]] = None,
    ):
        extent = Extent(*clip_extent) if clip_extent else self.extent
        for e in extent.tiles(width, height):
            g = self.clip(*e)
            assert g
            yield g

    @overload
    def clip(
        self, xmin: float, ymin: float, xmax: float, ymax: float, index: int = 1
    ) -> Optional[Grid]: ...

    @overload
    def clip(
        self, xmin: float, ymin: float, xmax: float, ymax: float, index: Tuple[int, ...]
    ) -> Optional[Stack]: ...

    def clip(self, xmin: float, ymin: float, xmax: float, ymax: float, index=None):
        if not index or isinstance(index, int):
            grids: List[Grid] = [
                g for g in self._read((xmin, ymin, xmax, ymax), index or 1) if g
            ]
            if grids:
                return mosaic(*grids)
            return None
        return stack(*(self.clip(xmin, ymin, xmax, ymax, index=i) for i in index))


@overload
def mosaic(*grids: str) -> Mosaic: ...


@overload
def mosaic(*grids: Grid) -> Grid: ...


def mosaic(*grids):
    g = grids[0]
    if isinstance(g, str):
        return Mosaic(*grids)
    return g.mosaic(*grids[1:])
