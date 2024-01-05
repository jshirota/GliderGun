import dataclasses
import hashlib
import numpy as np
import rasterio
import sys
import warnings
from dataclasses import dataclass
from numpy import arctan, arctan2, cos, gradient, ndarray, pi, sin, sqrt
from numpy.lib.stride_tricks import sliding_window_view
from rasterio import features
from rasterio.crs import CRS
from rasterio.drivers import driver_from_extension
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely import Point, Polygon
from sklearn.preprocessing import StandardScaler
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
from glidergun.literals import ColorMap, DataType


class Extent(Tuple[float, float, float, float]):
    def __new__(cls, xmin: float, ymin: float, xmax: float, ymax: float):
        return super(Extent, cls).__new__(cls, (xmin, ymin, xmax, ymax))

    def intersect(self, extent: "Extent"):
        return Extent(*[f(x) for f, x in zip((max, max, min, min), zip(self, extent))])

    def union(self, extent: "Extent"):
        return Extent(*[f(x) for f, x in zip((min, min, max, max), zip(self, extent))])

    __and__ = intersect
    __rand__ = __and__
    __or__ = union
    __ror__ = __or__


Operand = Union["Grid", float, int]
Value = Union[float, int]


@dataclass(frozen=True)
class Grid:
    data: ndarray
    crs: CRS
    transform: Affine
    _cmap: ColorMap = "gray"

    def __post_init__(self):
        self.data.flags.writeable = False

    def __repr__(self):
        d = 3 if self.dtype.startswith("float") else 0
        return (
            f"image: {self.width}x{self.height} {self.dtype} | "
            + f"range: {self.min:.{d}f}~{self.max:.{d}f} | "
            + f"mean: {self.mean:.{d}f} | "
            + f"std: {self.std:.{d}f} | "
            + f"crs: {self.crs} | "
            + f"cell: {self.cell_size}"
        )

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def dtype(self) -> DataType:
        return str(self.data.dtype)  # type: ignore

    @property
    def nodata(self):
        return _nodata(self.dtype)

    @property
    def has_nan(self) -> bool:
        return self.is_nan().data.any()  # type: ignore

    @property
    def xmin(self) -> float:
        return self.transform.c

    @property
    def ymin(self) -> float:
        return self.ymax + self.height * self.transform.e

    @property
    def xmax(self) -> float:
        return self.xmin + self.width * self.transform.a

    @property
    def ymax(self) -> float:
        return self.transform.f

    @property
    def extent(self) -> Extent:
        return Extent(self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def mean(self) -> float:
        return np.nanmean(self.data)  # type: ignore

    @property
    def std(self) -> float:
        return np.nanstd(self.data)  # type: ignore

    @property
    def min(self) -> float:
        return np.nanmin(self.data)

    @property
    def max(self) -> float:
        return np.nanmax(self.data)

    @property
    def cell_size(self) -> float:
        return self.transform.a

    @property
    def md5(self) -> str:
        return hashlib.md5(self.data).hexdigest()  # type: ignore

    def __add__(self, n: Operand):
        return self._apply(self, n, np.add)

    __radd__ = __add__

    def __sub__(self, n: Operand):
        return self._apply(self, n, np.subtract)

    def __rsub__(self, n: Operand):
        return self._apply(n, self, np.subtract)

    def __mul__(self, n: Operand):
        return self._apply(self, n, np.multiply)

    __rmul__ = __mul__

    def __pow__(self, n: Operand):
        return self._apply(self, n, np.power)

    def __rpow__(self, n: Operand):
        return self._apply(n, self, np.power)

    def __truediv__(self, n: Operand):
        return self._apply(self, n, np.true_divide)

    def __rtruediv__(self, n: Operand):
        return self._apply(n, self, np.true_divide)

    def __floordiv__(self, n: Operand):
        return self._apply(self, n, np.floor_divide)

    def __rfloordiv__(self, n: Operand):
        return self._apply(n, self, np.floor_divide)

    def __mod__(self, n: Operand):
        return self._apply(self, n, np.mod)

    def __rmod__(self, n: Operand):
        return self._apply(n, self, np.mod)

    def __lt__(self, n: Operand):
        return self._apply(self, n, np.less)

    def __gt__(self, n: Operand):
        return self._apply(self, n, np.greater)

    __rlt__ = __gt__

    __rgt__ = __lt__

    def __le__(self, n: Operand):
        return self._apply(self, n, np.less_equal)

    def __ge__(self, n: Operand):
        return self._apply(self, n, np.greater_equal)

    __rle__ = __ge__

    __rge__ = __le__

    def __eq__(self, n: Operand):
        return self._apply(self, n, np.equal)

    __req__ = __eq__

    def __ne__(self, n: Operand):
        return self._apply(self, n, np.not_equal)

    __rne__ = __ne__

    def __and__(self, n: Operand):
        return self._apply(self, n, np.bitwise_and)

    __rand__ = __and__

    def __or__(self, n: Operand):
        return self._apply(self, n, np.bitwise_or)

    __ror__ = __or__

    def __xor__(self, n: Operand):
        return self._apply(self, n, np.bitwise_xor)

    __rxor__ = __xor__

    def __rshift__(self, n: Operand):
        return self._apply(self, n, np.right_shift)

    def __lshift__(self, n: Operand):
        return self._apply(self, n, np.left_shift)

    __rrshift__ = __lshift__

    __rlshift__ = __rshift__

    def __neg__(self):
        return self._create(-1 * self.data)

    def __pos__(self):
        return self._create(1 * self.data)

    def __invert__(self):
        return con(self, False, True)

    def _create(self, data: ndarray):
        return _create(data, self.crs, self.transform)

    def _data(self, n: Operand):
        if isinstance(n, Grid):
            return n.data
        return n

    def _apply(self, left: Operand, right: Operand, op: Callable):
        if not isinstance(left, Grid) or not isinstance(right, Grid):
            return self._create(op(self._data(left), self._data(right)))

        if left.cell_size == right.cell_size and left.extent == right.extent:
            return self._create(op(left.data, right.data))

        l_adjusted, r_adjusted = _standardize(True, left, right)

        return self._create(op(l_adjusted.data, r_adjusted.data))

    def local(self, func: Callable[[ndarray], Any]):
        return self._create(func(self.data))

    def is_nan(self):
        return self.local(np.isnan)

    def abs(self):
        return self.local(np.abs)

    def sin(self):
        return self.local(np.sin)

    def cos(self):
        return self.local(np.cos)

    def tan(self):
        return self.local(np.tan)

    def arcsin(self):
        return self.local(np.arcsin)

    def arccos(self):
        return self.local(np.arccos)

    def arctan(self):
        return self.local(np.arctan)

    def round(self, decimals: int = 0):
        return self.local(lambda a: np.round(a, decimals))

    def focal(
        self, func: Callable[[ndarray], Any], buffer: int, circle: bool
    ) -> "Grid":
        return _batch(lambda g: _focal(func, buffer, circle, *g), buffer, self)[0]

    def focal_ptp(self, buffer=1, circle: bool = False, **kwargs):
        return self.focal(lambda a: np.ptp(a, axis=2, **kwargs), buffer, circle)

    def focal_percentile(
        self,
        percentile: float,
        buffer=1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        f = np.nanpercentile if ignore_nan else np.percentile
        return self.focal(lambda a: f(a, percentile, axis=2, **kwargs), buffer, circle)

    def focal_quantile(
        self,
        probability: float,
        buffer=1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        f = np.nanquantile if ignore_nan else np.quantile
        return self.focal(lambda a: f(a, probability, axis=2, **kwargs), buffer, circle)

    def focal_median(
        self,
        buffer=1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        f = np.nanmedian if ignore_nan else np.median
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_mean(
        self,
        buffer=1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        f = np.nanmean if ignore_nan else np.mean
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_std(
        self,
        buffer=1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        f = np.nanstd if ignore_nan else np.std
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_var(
        self,
        buffer=1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        f = np.nanvar if ignore_nan else np.var
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_min(
        self,
        buffer=1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        f = np.nanmin if ignore_nan else np.min
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_max(
        self,
        buffer=1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        f = np.nanmax if ignore_nan else np.max
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def focal_sum(
        self,
        buffer=1,
        circle: bool = False,
        ignore_nan: bool = True,
        **kwargs,
    ):
        f = np.nansum if ignore_nan else np.sum
        return self.focal(lambda a: f(a, axis=2, **kwargs), buffer, circle)

    def zonal(self, func: Callable[[ndarray], Any], zone_grid: "Grid"):
        zone_grid = zone_grid.type("int32")
        result = self
        for zone in set(zone_grid.data[np.isfinite(zone_grid.data)]):
            data = self.set_nan(zone_grid != zone).data
            statistics = func(data[np.isfinite(data)])
            result = con(zone_grid == zone, statistics, result)  # type: ignore
        return result

    def zonal_ptp(self, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.ptp(a, **kwargs), zone_grid)

    def zonal_percentile(self, percentile: float, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.percentile(a, percentile, **kwargs), zone_grid)

    def zonal_quantile(self, probability: float, zone_grid: "Grid", **kwargs):
        return self.zonal(lambda a: np.quantile(a, probability, **kwargs), zone_grid)

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

    def _reproject(
        self, transform, crs, width, height, resampling: Resampling
    ) -> "Grid":
        source = self * 1 if self.dtype == "bool" else self
        destination = np.ones((round(height), round(width))) * np.nan
        reproject(
            source=source.data,
            destination=destination,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=self.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=self.nodata,
            resampling=resampling,
        )
        result = _create(destination, crs, transform)
        if self.dtype == "bool":
            return result == 1
        return con(result == result.nodata, np.nan, result)

    def project(
        self, epsg: Union[int, CRS], resampling: Resampling = Resampling.nearest
    ) -> "Grid":
        crs = CRS.from_epsg(epsg) if isinstance(epsg, int) else epsg
        transform, width, height = calculate_default_transform(
            self.crs, crs, self.width, self.height, *self.extent
        )
        return self._reproject(transform, crs, width, height, resampling)

    def _resample(
        self,
        extent: Tuple[float, float, float, float],
        cell_size: float,
        resampling: Resampling,
    ) -> "Grid":
        (xmin, ymin, xmax, ymax) = extent
        xoff = (xmin - self.xmin) / self.transform.a
        yoff = (ymax - self.ymax) / self.transform.e
        scaling = cell_size / self.cell_size
        transform = (
            self.transform * Affine.translation(xoff, yoff) * Affine.scale(scaling)
        )
        width = (xmax - xmin) / abs(self.transform.a) / scaling
        height = (ymax - ymin) / abs(self.transform.e) / scaling
        return self._reproject(transform, self.crs, width, height, resampling)

    def clip(self, extent: Tuple[float, float, float, float]):
        return self._resample(extent, self.cell_size, Resampling.nearest)

    def resample(self, cell_size: float, resampling: Resampling = Resampling.nearest):
        return self._resample(self.extent, cell_size, resampling)

    def random(self):
        return self._create(np.random.rand(self.height, self.width))

    def aspect(self):
        x, y = gradient(self.data)
        return self._create(arctan2(-x, y))

    def slope(self):
        x, y = gradient(self.data)
        return self._create(pi / 2.0 - arctan(sqrt(x * x + y * y)))

    def hillshade(self, azimuth: float = 315, altitude: float = 45):
        azimuth = np.deg2rad(azimuth)
        altitude = np.deg2rad(altitude)
        aspect = self.aspect().data
        slope = self.slope().data
        shaded = sin(altitude) * sin(slope) + cos(altitude) * cos(slope) * cos(
            azimuth - aspect
        )
        return self._create((255 * (shaded + 1) / 2))

    def reclass(self, *mappings: Tuple[Value, Value, Value]):
        conditions = [
            (self.data >= min) & (self.data < max) for min, max, _ in mappings
        ]
        values = [value for _, _, value in mappings]
        return self._create(np.select(conditions, values, np.nan))

    def fill_nan(self, max_exponent: int = 4):
        if not self.has_nan:
            return self

        def f(grids):
            grid = grids[0]
            n = 0
            while grid.has_nan and n <= max_exponent:
                grid = con(grid.is_nan(), grid.focal_mean(2**n, True), grid)
                n += 1
            return (grid,)

        return _batch(f, 2**max_exponent, self)[0]

    def replace(
        self, value: Operand, replacement: Operand, fallback: Optional[Operand] = None
    ):
        return con(
            value if isinstance(value, Grid) else self == value,
            replacement,
            self if fallback is None else fallback,
        )

    def set_nan(self, value: Operand, fallback: Optional[Operand] = None):
        return self.replace(value, np.nan, fallback)

    def value(self, x: float, y: float) -> Value:
        xoff = (x - self.xmin) / self.transform.a
        yoff = (y - self.ymax) / self.transform.e
        if xoff < 0 or xoff >= self.width or yoff < 0 or yoff >= self.height:
            return np.nan
        return self.data[int(yoff), int(xoff)]

    def data_extent(self):
        xmin, ymin, xmax, ymax = None, None, None, None
        for (x, y), _ in self.to_points():
            if not xmin or x < xmin:
                xmin = x
            if not ymin or y < ymin:
                ymin = y
            if not xmax or x > xmax:
                xmax = x
            if not ymax or y > ymax:
                ymax = y
        if xmin is None or ymin is None or xmax is None or ymax is None:
            raise ValueError("None of the cells has a value.")
        n = self.cell_size / 2
        return Extent(xmin - n, ymin - n, xmax + n, ymax + n)

    def shrink(self):
        return self.clip(self.data_extent())

    def to_points(self) -> Iterable[Tuple[Point, Value]]:
        n = self.cell_size / 2
        for y, row in enumerate(self.data):
            for x, value in enumerate(row):
                if np.isfinite(value):
                    yield Point(
                        self.xmin + x * self.cell_size + n,
                        self.ymax - y * self.cell_size - n,
                    ), value

    def to_polygons(self) -> Iterable[Tuple[Polygon, Value]]:
        for shape, value in features.shapes(
            self.data, mask=np.isfinite(self.data), transform=self.transform
        ):
            coordinates = shape["coordinates"]
            yield Polygon(coordinates[0], coordinates[1:]), value

    def from_polygons(
        self, polygons: Iterable[Tuple[Polygon, Value]], all_touched: bool = False
    ):
        array = features.rasterize(
            shapes=polygons,
            out_shape=self.data.shape,
            fill=np.nan,  # type: ignore
            transform=self.transform,
            all_touched=all_touched,
            default_value=np.nan,  # type: ignore
        )
        return self._create(array)

    def scale(self, **fit_params):
        return self.local(lambda a: StandardScaler().fit_transform(a, **fit_params))

    def plot(self, cmap: ColorMap):
        return dataclasses.replace(self, _cmap=cmap)

    def map(
        self,
        cmap: ColorMap = "gray",
        opacity: float = 1.0,
        folium_map=None,
        width: int = 800,
        height: int = 600,
        basemap: Optional[str] = None,
        attribution: Optional[str] = None,
        grayscale: bool = True,
        **kwargs,
    ):
        from glidergun.ipython import _map

        return _map(
            self,
            cmap,
            opacity,
            folium_map,
            width,
            height,
            basemap,
            attribution,
            grayscale,
            **kwargs,
        )

    def type(self, dtype: DataType):
        if self.dtype == dtype:
            return self
        return self.local(lambda data: np.asanyarray(data, dtype=dtype))

    @overload
    def save(self, file: str, dtype: Optional[DataType] = None, driver: str = ""):
        ...

    @overload
    def save(
        self, file: MemoryFile, dtype: Optional[DataType] = None, driver: str = ""
    ):
        ...

    def save(self, file, dtype: Optional[DataType] = None, driver: str = ""):
        if dtype is None:
            dtype = self.dtype

        nodata = _nodata(dtype)

        grid = self if nodata is None else con(self.is_nan(), nodata, self)

        if isinstance(file, str):
            with rasterio.open(
                file,
                "w",
                driver=driver if driver else driver_from_extension(file),
                count=1,
                dtype=dtype,
                nodata=nodata,
                **_metadata(self),
            ) as dataset:
                dataset.write(grid.data, 1)
        elif isinstance(file, MemoryFile):
            with file.open(
                driver=driver if driver else "GTiff",
                count=1,
                dtype=dtype,
                nodata=nodata,
                **_metadata(self),
            ) as dataset:
                dataset.write(grid.data, 1)


@overload
def grid(file: str, index: int = 1) -> Grid:
    ...


@overload
def grid(file: MemoryFile, index: int = 1) -> Grid:
    ...


def grid(file, index: int = 1) -> Grid:
    if isinstance(file, str):
        with rasterio.open(file) as dataset:
            return _read(dataset, index)
    elif isinstance(file, MemoryFile):
        with file.open() as dataset:
            return _read(dataset, index)
    raise ValueError()


def _create(data: ndarray, crs: CRS, transform: Affine):
    if data.dtype == "float64":
        data = np.asanyarray(data, dtype="float32")
    elif data.dtype == "int64":
        data = np.asanyarray(data, dtype="int32")
    elif data.dtype == "uint64":
        data = np.asanyarray(data, dtype="uint32")
    return Grid(data, crs, transform)


def _read(dataset, index):
    grid = _create(dataset.read(index), dataset.crs, dataset.transform)
    return grid if dataset.nodata is None else grid.set_nan(dataset.nodata)


def _metadata(grid: Grid):
    return {
        "height": grid.height,
        "width": grid.width,
        "crs": grid.crs,
        "transform": grid.transform,
    }


def _mask(buffer: int) -> ndarray:
    size = 2 * buffer + 1
    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            d = ((x - buffer) ** 2 + (y - buffer) ** 2) ** (1 / 2)
            row.append(d <= buffer)
        rows.append(row)
    return np.array(rows)


def _pad(data: ndarray, buffer: int):
    row = np.zeros((buffer, data.shape[1])) * np.nan
    col = np.zeros((data.shape[0] + 2 * buffer, buffer)) * np.nan
    return np.hstack([col, np.vstack([row, data, row]), col], dtype="float32")


def _focal(func: Callable, buffer: int, circle: bool, *grids: Grid) -> Tuple[Grid, ...]:
    grids_adjusted = _standardize(True, *grids)
    size = 2 * buffer + 1
    mask = _mask(buffer) if circle else np.full((size, size), True)

    if len(grids) == 1:
        array = sliding_window_view(_pad(grids[0].data, buffer), (size, size))
        result = func(array[:, :, mask])
    else:
        array = np.stack(
            [
                sliding_window_view(_pad(g.data, buffer), (size, size))
                for g in grids_adjusted
            ]
        )
        transposed = np.transpose(array, axes=(1, 2, 0, 3, 4))[:, :, :, mask]
        result = func(tuple(transposed[:, :, i] for i, _ in enumerate(grids)))

    if isinstance(result, ndarray) and len(result.shape) == 2:
        return (grids_adjusted[0]._create(np.array(result)),)

    return tuple([grids_adjusted[0]._create(r) for r in result])


def _batch(
    func: Callable[[Tuple[Grid, ...]], Tuple[Grid, ...]], buffer: int, *grids: Grid
) -> Tuple[Grid, ...]:
    stride = 8000 // buffer // len(grids)
    grids1 = _standardize(True, *grids)
    grid = grids1[0]

    def tile():
        for x in range(0, grid.width // stride + 1):
            xmin, xmax = x * stride, min((x + 1) * stride, grid.width)
            if xmin < xmax:
                for y in range(0, grid.height // stride + 1):
                    ymin, ymax = y * stride, min((y + 1) * stride, grid.height)
                    if ymin < ymax:
                        yield xmin, ymin, xmax, ymax

    tiles = list(tile())
    count = len(tiles)

    if count <= 4:
        return func(tuple(grids1))

    results: List[Grid] = []
    cell_size = grid.cell_size
    n = 0

    for xmin, ymin, xmax, ymax in tiles:
        n += 1
        sys.stdout.write(f"\rProcessing {n} of {count} tiles...")
        sys.stdout.flush()
        grids2 = [
            g.clip(
                (
                    grid.xmin + (xmin - buffer) * cell_size,
                    grid.ymin + (ymin - buffer) * cell_size,
                    grid.xmin + (xmax + buffer) * cell_size,
                    grid.ymin + (ymax + buffer) * cell_size,
                )
            )
            for g in grids1
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            grids3 = func(tuple(grids2))

        grids4 = [
            g.clip(
                (
                    grid.xmin + xmin * cell_size,
                    grid.ymin + ymin * cell_size,
                    grid.xmin + xmax * cell_size,
                    grid.ymin + ymax * cell_size,
                )
            )
            for g in grids3
        ]

        if results:
            for i, g in enumerate(grids4):
                results[i] = mosaic(results[i], g)
        else:
            results = grids4

    print()
    return tuple(results)


def con(grid: Grid, trueValue: Operand, falseValue: Operand):
    return grid.local(
        lambda data: np.where(data, grid._data(trueValue), grid._data(falseValue))
    )


def _aggregate(func: Callable, *grids: Grid) -> Grid:
    grids_adjusted = _standardize(True, *grids)
    data = func(np.array([grid.data for grid in grids_adjusted]), axis=0)
    return grids_adjusted[0]._create(data)


def mean(*grids: Grid) -> Grid:
    return _aggregate(np.mean, *grids)


def std(*grids: Grid) -> Grid:
    return _aggregate(np.std, *grids)


def minimum(*grids: Grid) -> Grid:
    return _aggregate(np.min, *grids)


def maximum(*grids: Grid) -> Grid:
    return _aggregate(np.max, *grids)


def mosaic(*grids: Grid) -> Grid:
    grids_adjusted = _standardize(False, *grids)
    result = grids_adjusted[0]
    for grid in grids_adjusted[1:]:
        result = con(result.is_nan(), grid, result)
    return result


def _standardize(intersect: bool, *grids: Grid) -> List[Grid]:
    if len(grids) == 1:
        return list(grids)

    crs_set = set(grid.crs for grid in grids)

    if len(crs_set) > 1:
        raise ValueError("Input grids must have the same CRS.")

    cell_size = 0
    extent = None

    for grid in grids:
        cell_size = grid.cell_size if grid.cell_size > cell_size else cell_size
        extent = (
            grid.extent
            if extent is None
            else extent & grid.extent
            if intersect
            else extent | grid.extent
        )

    results = []

    for grid in grids:
        if grid.cell_size != cell_size:
            grid = grid.resample(cell_size)
        if grid.extent != extent:
            grid = grid.clip(extent)  # type: ignore
        results.append(grid)

    return results


def _nodata(dtype: str) -> Optional[Value]:
    if dtype == "bool":
        return None
    if dtype.startswith("float"):
        return np.finfo(dtype).min  # type: ignore
    if dtype.startswith("uint"):
        return np.iinfo(dtype).max
    return np.iinfo(dtype).min


@dataclass(frozen=True)
class Stack:
    grids: Tuple[Grid, ...]
    _rgb: Tuple[int, int, int] = (1, 2, 3)

    def __repr__(self):
        g = self.grids[0]
        return (
            f"image: {g.width}x{g.height} {g.dtype} | "
            + f"crs: {g.crs} | "
            + f"cell: {g.cell_size} | "
            + f"count: {len(self.grids)}"
        )

    @property
    def width(self) -> int:
        return self.grids[0].width

    @property
    def height(self) -> int:
        return self.grids[0].height

    @property
    def dtype(self) -> DataType:
        return self.grids[0].dtype

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
    def cell_size(self) -> float:
        return self.grids[0].cell_size

    @property
    def extent(self) -> Extent:
        return self.grids[0].extent

    def scale(self, **fit_params):
        return self.each(lambda g: g.scale(**fit_params))

    def plot(self, *rgb: int):
        return dataclasses.replace(self, _rgb=rgb)

    def map(
        self,
        rgb: Tuple[int, int, int] = (1, 2, 3),
        opacity: float = 1.0,
        folium_map=None,
        width: int = 800,
        height: int = 600,
        basemap: Optional[str] = None,
        attribution: Optional[str] = None,
        grayscale: bool = True,
        **kwargs,
    ):
        from glidergun.ipython import _map

        return _map(
            self,
            rgb,
            opacity,
            folium_map,
            width,
            height,
            basemap,
            attribution,
            grayscale,
            **kwargs,
        )

    def each(self, func: Callable[[Grid], Grid]):
        return stack(*map(func, self.grids))

    def clip(self, extent: Tuple[float, float, float, float]):
        return self.each(lambda g: g.clip(extent))

    def project(
        self, epsg: Union[int, CRS], resampling: Resampling = Resampling.nearest
    ):
        return self.each(lambda g: g.project(epsg, resampling))

    def resample(self, cell_size: float, resampling: Resampling = Resampling.nearest):
        return self.each(lambda g: g.resample(cell_size, resampling))

    def zip_with(self, other_stack: "Stack", func: Callable[[Grid, Grid], Grid]):
        grids = []
        for grid1, grid2 in zip(self.grids, other_stack.grids):
            grid1, grid2 = _standardize(True, grid1, grid2)
            grids.append(func(grid1, grid2))
        return stack(*grids)

    def values(self, x: float, y: float):
        return tuple(grid.value(x, y) for grid in self.grids)

    @overload
    def save(self, file: str, dtype: Optional[DataType] = None, driver: str = ""):
        ...

    @overload
    def save(
        self, file: MemoryFile, dtype: Optional[DataType] = None, driver: str = ""
    ):
        ...

    def save(self, file, dtype: Optional[DataType] = None, driver: str = ""):
        g = self.grids[0]

        if dtype is None:
            dtype = self.dtype

        nodata = _nodata(dtype)

        grids = (
            self.grids
            if nodata is None
            else self.each(lambda g: con(g.is_nan(), nodata, g)).grids
        )

        if isinstance(file, str):
            with rasterio.open(
                file,
                "w",
                driver=driver if driver else driver_from_extension(file),
                count=len(grids),
                dtype=dtype,
                nodata=nodata,
                **_metadata(g),
            ) as dataset:
                for index, grid in enumerate(grids):
                    dataset.write(grid.data, index + 1)
        elif isinstance(file, MemoryFile):
            with file.open(
                driver=driver if driver else "GTiff",
                count=len(grids),
                dtype=dtype,
                nodata=nodata,
                **_metadata(g),
            ) as dataset:
                for index, grid in enumerate(grids):
                    dataset.write(grid.data, index + 1)


@overload
def stack(*grids: str) -> Stack:
    ...


@overload
def stack(*grids: MemoryFile) -> Stack:
    ...


@overload
def stack(*grids: Grid) -> Stack:
    ...


def stack(*grids) -> Stack:
    bands: List[Grid] = []

    for grid in grids:
        if isinstance(grid, Grid):
            bands.append(grid)
        else:
            with rasterio.open(grid) if isinstance(
                grid, str
            ) else grid.open() as dataset:
                for index in dataset.indexes:
                    band = _read(dataset, index)
                    bands.append(band)

    return Stack(tuple(_standardize(True, *bands)))
