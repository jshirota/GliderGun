import dataclasses
from base64 import b64encode
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, overload

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio import DatasetReader
from rasterio.crs import CRS
from rasterio.drivers import driver_from_extension
from rasterio.io import MemoryFile
from rasterio.warp import Resampling

from glidergun._grid import (
    Extent,
    Grid,
    _metadata,
    _read,
    con,
    pca,
    standardize,
)
from glidergun._literals import BaseMap, DataType
from glidergun._types import Scaler
from glidergun._utils import create_directory, get_crs, get_nodata_value

Operand = Union["Stack", Grid, float, int]


@dataclass(frozen=True)
class Stack:
    grids: Tuple[Grid, ...]
    display: Tuple[int, int, int] = (1, 2, 3)

    def __repr__(self):
        g = self.grids[0]
        return f"crs: {g.crs} | count: {len(self.grids)} | rgb: {self.display}"

    def _thumbnail(self, figsize: Optional[Tuple[float, float]] = None):
        with BytesIO() as buffer:
            figure = plt.figure(figsize=figsize, frameon=False)
            axes = figure.add_axes((0, 0, 1, 1))
            axes.axis("off")
            obj = self.to_uint8_range()
            rgb = [
                obj.grids[i - 1].data
                for i in (self.display if self.display else (1, 2, 3))
            ]
            alpha = np.where(np.isfinite(rgb[0] + rgb[1] + rgb[2]), 255, 0)
            plt.imshow(np.dstack([*[np.asanyarray(g, "uint8") for g in rgb], alpha]))
            plt.savefig(buffer, bbox_inches="tight", pad_inches=0)
            plt.close(figure)
            return buffer.getvalue()

    @cached_property
    def img(self) -> str:
        image = b64encode(self._thumbnail()).decode()
        return f"data:image/png;base64, {image}"

    @property
    def crs(self) -> CRS:
        return self.grids[0].crs

    @property
    def xmin(self) -> float:
        return self.grids[0].xmin

    @property
    def ymin(self) -> float:
        return self.grids[0].ymin

    @property
    def xmax(self) -> float:
        return self.grids[0].xmax

    @property
    def ymax(self) -> float:
        return self.grids[0].ymax

    @property
    def extent(self) -> Extent:
        return self.grids[0].extent

    @property
    def md5s(self) -> Tuple[str, ...]:
        return tuple(g.md5 for g in self.grids)

    def __add__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__add__(n))

    __radd__ = __add__

    def __sub__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__sub__(n))

    def __rsub__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rsub__(n))

    def __mul__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__mul__(n))

    __rmul__ = __mul__

    def __pow__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__pow__(n))

    def __rpow__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rpow__(n))

    def __truediv__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__truediv__(n))

    def __rtruediv__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rtruediv__(n))

    def __floordiv__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__floordiv__(n))

    def __rfloordiv__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rfloordiv__(n))

    def __mod__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__mod__(n))

    def __rmod__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rmod__(n))

    def __lt__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__lt__(n))

    def __gt__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__gt__(n))

    __rlt__ = __gt__

    __rgt__ = __lt__

    def __le__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__le__(n))

    def __ge__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__ge__(n))

    __rle__ = __ge__

    __rge__ = __le__

    def __eq__(self, n: object):
        if not isinstance(n, (Grid, float, int)):
            return NotImplemented
        return self._apply(n, lambda g, n: g.__eq__(n))

    __req__ = __eq__

    def __ne__(self, n: object):
        if not isinstance(n, (Grid, float, int)):
            return NotImplemented
        return self._apply(n, lambda g, n: g.__ne__(n))

    __rne__ = __ne__

    def __and__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__and__(n))

    __rand__ = __and__

    def __or__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__or__(n))

    __ror__ = __or__

    def __xor__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__xor__(n))

    __rxor__ = __xor__

    def __rshift__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__rshift__(n))

    def __lshift__(self, n: Operand):
        return self._apply(n, lambda g, n: g.__lshift__(n))

    __rrshift__ = __lshift__

    __rlshift__ = __rshift__

    def __neg__(self):
        return self.each(lambda g: g.__neg__())

    def __pos__(self):
        return self.each(lambda g: g.__pos__())

    def __invert__(self):
        return self.each(lambda g: g.__invert__())

    def _apply(self, n: Operand, op: Callable):
        if isinstance(n, Stack):
            return self.zip_with(n, lambda g1, g2: op(g1, g2))
        return self.each(lambda g: op(g, n))

    def scale(self, scaler: Scaler, **fit_params):
        return self.each(lambda g: g.scale(scaler, **fit_params))

    def percent_clip(self, min_percent: float, max_percent: float):
        return self.each(lambda g: g.percent_clip(min_percent, max_percent))

    def to_uint8_range(self):
        return self.each(lambda g: g.to_uint8_range())

    def color(self, rgb: Tuple[int, int, int]):
        valid = set(range(1, len(self.grids) + 1))
        if set(rgb) - valid:
            raise ValueError("Invalid bands specified.")
        return dataclasses.replace(self, display=rgb)

    def map(
        self,
        opacity: float = 1.0,
        basemap: Union[BaseMap, Any, None] = None,
        width: int = 800,
        height: int = 600,
        attribution: Optional[str] = None,
        grayscale: bool = True,
        **kwargs,
    ):
        from glidergun._display import get_folium_map

        return get_folium_map(
            self, opacity, basemap, width, height, attribution, grayscale, **kwargs
        )

    def each(self, func: Callable[[Grid], Grid]):
        return stack(*map(func, self.grids))

    def georeference(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        crs: Union[int, CRS] = 4326,
    ):
        return self.each(lambda g: g.georeference(xmin, ymin, xmax, ymax, crs))

    def tiles(self, width: float, height: float):
        return (
            stack(*grids)
            for grids in zip(*(g.tiles(width, height) for g in self.grids))
        )

    def clip(self, xmin: float, ymin: float, xmax: float, ymax: float):
        return self.each(lambda g: g.clip(xmin, ymin, xmax, ymax))

    def clip_at(self, x: float, y: float, width: int = 8, height: int = 8):
        return self.each(lambda g: g.clip_at(x, y, width, height))

    def extract_bands(self, *bands: int):
        return stack(*(self.grids[i - 1] for i in bands))

    def pca(self, n_components: int = 3):
        return stack(*pca(n_components, *self.grids))

    def project(
        self, crs: Union[int, CRS], resampling: Resampling = Resampling.nearest
    ):
        if get_crs(crs).wkt == self.crs.wkt:
            return self
        return self.each(lambda g: g.project(crs, resampling))

    def resample(
        self,
        cell_size: Union[Tuple[float, float], float],
        resampling: Resampling = Resampling.nearest,
    ):
        return self.each(lambda g: g.resample(cell_size, resampling))

    def zip_with(self, other_stack: "Stack", func: Callable[[Grid, Grid], Grid]):
        grids = []
        for grid1, grid2 in zip(self.grids, other_stack.grids):
            grid1, grid2 = standardize(grid1, grid2)
            grids.append(func(grid1, grid2))
        return stack(*grids)

    def value_at(self, x: float, y: float):
        return tuple(grid.value_at(x, y) for grid in self.grids)

    def type(self, dtype: DataType):
        return self.each(lambda g: g.type(dtype))

    @overload
    def save(
        self, file: str, dtype: Optional[DataType] = None, driver: str = ""
    ) -> None: ...

    @overload
    def save(  # type: ignore
        self, file: MemoryFile, dtype: Optional[DataType] = None, driver: str = ""
    ) -> None: ...

    def save(self, file, dtype: Optional[DataType] = None, driver: str = ""):
        if isinstance(file, str) and (
            file.lower().endswith(".jpg")
            or file.lower().endswith(".kml")
            or file.lower().endswith(".kmz")
            or file.lower().endswith(".png")
        ):
            grids = self.extract_bands(*self.display).to_uint8_range().grids
            dtype = "uint8"
        else:
            grids = self.grids
            if dtype is None:
                dtype = grids[0].dtype

        nodata = get_nodata_value(dtype)

        if nodata is not None:
            grids = tuple(con(g.is_nan(), float(nodata), g) for g in grids)

        if isinstance(file, str):
            create_directory(file)
            with rasterio.open(
                file,
                "w",
                driver=driver if driver else driver_from_extension(file),
                count=len(grids),
                dtype=dtype,
                nodata=nodata,
                **_metadata(self.grids[0]),
            ) as dataset:
                for index, grid in enumerate(grids):
                    dataset.write(grid.data, index + 1)
        elif isinstance(file, MemoryFile):
            with file.open(
                driver=driver if driver else "GTiff",
                count=len(grids),
                dtype=dtype,
                nodata=nodata,
                **_metadata(self.grids[0]),
            ) as dataset:
                for index, grid in enumerate(grids):
                    dataset.write(grid.data, index + 1)


@overload
def stack(*grids: str) -> Stack:
    """Creates a new stack from file paths.

    Args:
        grids: File paths.

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(*grids: rasterio.DatasetReader) -> Stack:  # type: ignore
    """Creates a new stack from data readers.

    Args:
        grids: Data readers.

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(*grids: MemoryFile) -> Stack:  # type: ignore
    """Creates a new stack from memory files.

    Args:
        grids: Memory files.

    Returns:
        Stack: A new stack.
    """
    ...


@overload
def stack(*grids: Grid) -> Stack:
    """Creates a new stack from grids.

    Args:
        grids: Grids.

    Returns:
        Stack: A new stack.
    """
    ...


def stack(*grids) -> Stack:
    bands: List[Grid] = []

    for grid in grids:
        if isinstance(grid, DatasetReader):
            bands.extend(_read_grids(grid))
        elif isinstance(grid, str):
            with rasterio.open(grid) as dataset:
                bands.extend(_read_grids(dataset))
        elif isinstance(grid, MemoryFile):
            with grid.open() as dataset:
                bands.extend(_read_grids(dataset))
        elif isinstance(grid, Grid):
            bands.append(grid)

    crs = set(g.crs for g in bands)
    if len(crs) > 1:
        raise ValueError("All grids must have the same CRS.")

    extent = set(g.extent for g in bands)
    if len(extent) > 1:
        raise ValueError("All grids must have the same extent.")

    return Stack(tuple(bands))


def _read_grids(dataset) -> Iterator[Grid]:
    if dataset.subdatasets:
        for index, _ in enumerate(dataset.subdatasets):
            with rasterio.open(dataset.subdatasets[index]) as subdataset:
                yield from _read_grids(subdataset)
    elif dataset.indexes:
        for index in dataset.indexes:
            grid = _read(dataset, index, None)
            if grid:
                yield grid
