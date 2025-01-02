import dataclasses
import hashlib
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from io import BytesIO
from types import FunctionType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import patheffects
from numpy import arctan, arctan2, cos, gradient, ndarray, pi, sin, sqrt
from rasterio import DatasetReader, features
from rasterio.crs import CRS
from rasterio.drivers import driver_from_extension
from rasterio.io import MemoryFile
from rasterio.transform import Affine, from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from glidergun._focal import Focal
from glidergun._interpolation import Interpolation
from glidergun._literals import (
    BaseMap,
    ColorMap,
    DataType,
    ExtentResolution,
    ResamplingMethod,
)
from glidergun._prediction import Prediction
from glidergun._shapefile import Shapefile
from glidergun._types import CellSize, Defaults, Extent, GridCore, PointValue, Scaler
from glidergun._utils import (
    create_directory,
    format_type,
    get_crs,
    get_nodata_value,
)
from glidergun._zonal import Zonal

Operand = Union["Grid", float, int]


@dataclass(frozen=True)
class Grid(GridCore, Interpolation, Prediction, Focal, Zonal, Shapefile):
    display: Union[ColorMap, Any] = field(default_factory=lambda: Defaults.display)

    def __post_init__(self):
        self.data.flags.writeable = False

        if self.width * self.height == 0:
            raise ValueError("Empty raster.")

    def __repr__(self):
        d = 3 if self.dtype.startswith("float") else 0
        return (
            f"image: {self.width}x{self.height} {self.dtype} | "
            + f"range: {self.min:.{d}f}~{self.max:.{d}f} | "
            + f"mean: {self.mean:.{d}f} | "
            + f"std: {self.std:.{d}f} | "
            + f"crs: {self.crs} | "
            + f"cell: {self.cell_size.x}, {self.cell_size.y}"
        )

    def _thumbnail(
        self, figsize: Optional[Tuple[float, float]] = None, show_values: bool = False
    ):
        with BytesIO() as buffer:
            figure = plt.figure(figsize=figsize, frameon=False)
            axes = figure.add_axes((0, 0, 1, 1))
            axes.axis("off")
            obj = self.to_uint8_range()
            plt.imshow(obj.data, cmap=self.display)

            if (
                show_values
                and self.dtype != "bool"
                and self.width <= Defaults.annotation_threshold
                and self.height <= Defaults.annotation_threshold
            ):
                for row in range(self.height):
                    for column in range(self.width):
                        value = self.data[row][column]
                        plt.annotate(
                            str(round(value, 2)),
                            xy=(column + 0.02, row + 0.02),
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=9,
                            path_effects=[
                                patheffects.withStroke(linewidth=1, foreground="black")
                            ],
                        )

            plt.savefig(buffer, bbox_inches="tight", pad_inches=0)
            plt.close(figure)
            return buffer.getvalue()

    @cached_property
    def img(self) -> str:
        image = b64encode(self._thumbnail(show_values=True)).decode()
        return f"data:image/png;base64, {image}"

    @cached_property
    def width(self) -> int:
        return self.data.shape[1]

    @cached_property
    def height(self) -> int:
        return self.data.shape[0]

    @cached_property
    def dtype(self) -> DataType:
        return cast(DataType, str(self.data.dtype))

    @cached_property
    def nodata(self):
        return get_nodata_value(self.dtype)

    @cached_property
    def has_nan(self):
        return bool(self.is_nan().data.any())

    @cached_property
    def xmin(self) -> float:
        return self.extent.xmin

    @cached_property
    def ymin(self) -> float:
        return self.extent.ymin

    @cached_property
    def xmax(self) -> float:
        return self.extent.xmax

    @cached_property
    def ymax(self) -> float:
        return self.extent.ymax

    @cached_property
    def extent(self) -> Extent:
        return _extent(self.width, self.height, self.transform)

    @cached_property
    def mean(self):
        return float(np.nanmean(self.data))

    @cached_property
    def std(self):
        return float(np.nanstd(self.data))

    @cached_property
    def min(self):
        return float(np.nanmin(self.data))

    @cached_property
    def max(self):
        return float(np.nanmax(self.data))

    @cached_property
    def cell_size(self) -> CellSize:
        return CellSize(self.transform.a, -self.transform.e)

    @cached_property
    def bins(self) -> Dict[float, int]:
        unique, counts = zip(np.unique(self.data, return_counts=True))
        return dict(sorted(zip(map(float, unique[0]), map(int, counts[0]))))

    @cached_property
    def md5(self) -> str:
        return hashlib.md5(self.data.copy(order="C")).hexdigest()  # type: ignore

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

    def __eq__(self, n: object):
        if not isinstance(n, (Grid, float, int)):
            return NotImplemented
        return self._apply(self, n, np.equal)

    __req__ = __eq__

    def __ne__(self, n: object):
        if not isinstance(n, (Grid, float, int)):
            return NotImplemented
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
        return self.local(-1 * self.data)

    def __pos__(self):
        return self.local(1 * self.data)

    def __invert__(self):
        return con(self, False, True)

    def is_greater_than(self, n: Operand):
        return self > n

    def is_less_than(self, n: Operand):
        return self < n

    def is_greater_than_or_equal(self, n: Operand):
        return self >= n

    def is_less_than_or_equal(self, n: Operand):
        return self <= n

    def is_equal(self, n: Operand):
        return self == n

    def is_not_equal(self, n: Operand):
        return self != n

    def _data(self, n: Operand):
        if isinstance(n, Grid):
            return n.data
        return n

    def _apply(self, left: Operand, right: Operand, op: Callable):
        if not isinstance(left, Grid) or not isinstance(right, Grid):
            return self.local(op(self._data(left), self._data(right)))

        if left.cell_size == right.cell_size and left.extent == right.extent:
            return self.local(op(left.data, right.data))

        l_adjusted, r_adjusted = standardize(left, right)

        return self.local(op(l_adjusted.data, r_adjusted.data))

    def local(self, func: Union[Callable[[ndarray], ndarray], ndarray]):
        data = func if isinstance(func, ndarray) else func(self.data)
        return grid(data, self.transform, self.crs)

    def mosaic(self, *grids: "Grid"):
        grids_adjusted = standardize(self, *grids, extent="union")
        result = grids_adjusted[0]
        for g in grids_adjusted[1:]:
            result = con(result.is_nan(), g, result)
        return result

    def standardize(self, *grids: "Grid"):
        return standardize(self, *grids)

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

    def log(self, base: Optional[float] = None):
        if base is None:
            return self.local(np.log)
        return self.local(lambda a: np.log(a) / np.log(base))

    def round(self, decimals: int = 0):
        return self.local(lambda a: np.round(a, decimals))

    def georeference(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        crs: Union[int, CRS] = 4326,
    ):
        return grid(self.data, (xmin, ymin, xmax, ymax), crs)

    def _reproject(
        self,
        transform,
        crs,
        width,
        height,
        resampling: Union[Resampling, ResamplingMethod],
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
            resampling=(
                Resampling[resampling] if isinstance(resampling, str) else resampling
            ),
        )
        result = grid(destination, transform, crs)
        if self.dtype == "bool":
            return result == 1
        return con(result == result.nodata, np.nan, result)

    def project(
        self,
        crs: Union[int, CRS],
        resampling: Union[Resampling, ResamplingMethod] = "nearest",
    ) -> "Grid":
        if get_crs(crs).wkt == self.crs.wkt:
            return self
        transform, width, height = calculate_default_transform(
            self.crs, crs, self.width, self.height, *self.extent
        )
        return self._reproject(
            transform,
            crs,
            width,
            height,
            Resampling[resampling] if isinstance(resampling, str) else resampling,
        )

    def _resample(
        self,
        extent: Tuple[float, float, float, float],
        cell_size: Tuple[float, float],
        resampling: Union[Resampling, ResamplingMethod],
    ) -> "Grid":
        (xmin, ymin, xmax, ymax) = extent
        xoff = (xmin - self.xmin) / self.transform.a
        yoff = (ymax - self.ymax) / self.transform.e
        scaling_x = cell_size[0] / self.cell_size.x
        scaling_y = cell_size[1] / self.cell_size.y
        transform = (
            self.transform
            * Affine.translation(xoff, yoff)
            * Affine.scale(scaling_x, scaling_y)
        )
        width = (xmax - xmin) / abs(self.transform.a) / scaling_x
        height = (ymax - ymin) / abs(self.transform.e) / scaling_y
        return self._reproject(transform, self.crs, width, height, resampling)

    def tiles(self, width: float, height: float):
        for e in self.extent.tiles(width, height):
            yield self.clip(*e)

    def clip(self, xmin: float, ymin: float, xmax: float, ymax: float):
        return self._resample(
            (xmin, ymin, xmax, ymax), self.cell_size, Resampling.nearest
        )

    def clip_at(self, x: float, y: float, width: int = 8, height: int = 8):
        x_offset = self.cell_size.x * width / 2
        y_offset = self.cell_size.y * height / 2
        xmin = x - x_offset
        ymin = y - y_offset
        xmax = x + x_offset
        ymax = y + y_offset
        return self.clip(xmin, ymin, xmax, ymax)

    def resample(
        self,
        cell_size: Union[Tuple[float, float], float],
        resampling: Union[Resampling, ResamplingMethod] = "nearest",
    ):
        if isinstance(cell_size, (int, float)):
            cell_size = (cell_size, cell_size)
        if self.cell_size == cell_size:
            return self
        return self._resample(self.extent, cell_size, resampling)

    def resize(
        self,
        width: int,
        height: int,
        resampling: Union[Resampling, ResamplingMethod] = "nearest",
    ):
        cell_size_x = self.cell_size.x * self.width / width
        cell_size_y = self.cell_size.y * self.height / height
        return self.resample((cell_size_x, cell_size_y), resampling)

    def buffer(self, value: Union[float, int], count: int):
        if count < 0:
            g = (self != value).buffer(1, -count)
            return con(g == 0, value, self.set_nan(self == value))
        g = self
        for _ in range(count):
            g = con(g.focal_count(value, 1, True) > 0, value, g)
        return g

    def density(
        self,
        points: Optional[Sequence[Tuple[float, float]]] = None,
        max_workers: int = 1,
    ):
        if points is None:
            points = [(p.x, p.y) for p in self.to_points()]
        return density(points, self.extent, self.crs, self.cell_size, max_workers)

    def distance(
        self,
        points: Optional[Sequence[Tuple[float, float]]] = None,
        max_workers: int = 1,
    ):
        if points is None:
            points = [(p.x, p.y) for p in self.to_points()]
        return distance(points, self.extent, self.crs, self.cell_size, max_workers)

    def interp_idw(
        self,
        points: Optional[Sequence[Tuple[float, float, float]]] = None,
        cell_size: Union[Tuple[float, float], float, None] = None,
        radius: Optional[float] = None,
        max_workers: int = 1,
    ):
        if points is None:
            points = self.to_points()
        return idw(
            points,
            self.extent,
            self.crs,
            cell_size or self.cell_size,
            radius,
            max_workers,
        )

    def randomize(self, normal_distribution: bool = False):
        f = np.random.randn if normal_distribution else np.random.rand
        return self.local(f(self.height, self.width))

    def aspect(self, radians: bool = False):
        y, x = gradient(self.data)
        g = self.local(arctan2(-y, x))
        if radians:
            return g
        return (g.local(np.degrees) + 360) % 360

    def slope(self, radians: bool = False):
        y, x = gradient(self.data)
        g = self.local(arctan(sqrt(x * x + y * y)))
        if radians:
            return g
        return g.local(np.degrees)

    def hillshade(self, azimuth: float = 315, altitude: float = 45):
        azimuth = np.deg2rad(azimuth)
        altitude = np.deg2rad(altitude)
        aspect = self.aspect(True).data
        slope = (pi / 2.0 - self.slope(True)).data
        shaded = sin(altitude) * sin(slope) + cos(altitude) * cos(slope) * cos(
            azimuth - aspect
        )
        return self.local((255 * (shaded + 1) / 2))

    def con(
        self,
        predicate: Union[Operand, Callable[["Grid"], "Grid"]],
        replacement: Operand,
        fallback: Optional[Operand] = None,
    ):
        if isinstance(predicate, FunctionType):
            g = predicate(self)
        elif isinstance(predicate, Grid):
            g = predicate
        else:
            g = self == predicate
        return con(
            self.standardize(g)[1], replacement, self if fallback is None else fallback
        )

    def set_nan(
        self,
        predicate: Union[Operand, Callable[["Grid"], "Grid"]],
        fallback: Optional[Operand] = None,
    ):
        return self.con(predicate, np.nan, fallback)

    def then(self, trueValue: Operand, falseValue: Operand):
        return con(self, trueValue, falseValue)

    def process_tiles(
        self,
        func: Callable[["Grid"], "Grid"],
        tile_size: int = 256,
        buffer: int = 0,
        max_workers: int = 1,
    ) -> "Grid":
        cell_x, cell_y = self.cell_size
        tiles = list(self.extent.tiles(cell_x * tile_size, cell_y * tile_size))

        if len(tiles) <= 4:
            return func(self)

        x = buffer * cell_x
        y = buffer * cell_y

        def f(tile: Extent):
            xmin, ymin, xmax, ymax = tile
            g = func(self.clip(xmin - x, ymin - y, xmax + x, ymax + y))
            if buffer == 0:
                return g
            return g.clip(xmin, ymin, xmax, ymax)

        result: Optional["Grid"] = None

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers or 1) as executor:
                for r in executor.map(f, tiles):
                    result = result.mosaic(r) if result else r
        else:
            for i, tile in enumerate(tiles):
                print(f"Processing tile {i + 1} of {len(tiles)}...")
                result = result.mosaic(f(tile)) if result else f(tile)

        assert result
        return result

    def value_at(self, x: float, y: float) -> float:
        c = int((x - self.xmin) / self.cell_size.x)
        r = int((self.ymax - y) / self.cell_size.y)
        if c < 0 or c >= self.width or r < 0 or r >= self.height:
            return float(np.nan)
        return float(self.data[r, c])

    @cached_property
    def data_extent(self) -> Extent:
        if not self.has_nan:
            return self.extent
        points = self._coords_with_values()
        xmin, ymin = points[:, :2].min(axis=0)
        xmax, ymax = points[:, :2].max(axis=0)
        return Extent(
            xmin - self.cell_size.x / 2,
            ymin - self.cell_size.y / 2,
            xmax + self.cell_size.x / 2,
            ymax + self.cell_size.y / 2,
        )

    def _xs(self):
        offset = self.cell_size.x / 2
        return np.linspace(self.xmin + offset, self.xmax - offset, self.width)

    def _ys(self):
        offset = self.cell_size.y / 2
        return np.linspace(self.ymax - offset, self.ymin + offset, self.height)

    def _coords(self):
        x, y = np.meshgrid(self._xs(), self._ys())
        return np.asarray(np.column_stack([x.ravel(), y.ravel()]), "float32")

    def _coords_with_values(self, include_nan: bool = False):
        x, y = np.meshgrid(self._xs(), self._ys())
        array = np.column_stack([x.ravel(), y.ravel(), self.data.ravel()])
        if include_nan:
            return array
        return array[~np.isnan(array[:, 2])]

    def to_points(self, include_nan: bool = False) -> List[PointValue]:
        return [
            PointValue(float(x), float(y), float(v))
            for x, y, v in self._coords_with_values(include_nan)
        ]

    def to_polygons(self, include_nan: bool = False) -> List[Tuple[Polygon, float]]:
        g = self * 1
        mask = None if include_nan else np.isfinite(g.data)
        return [
            (Polygon(shape["coordinates"][0], shape["coordinates"][1:]), float(value))
            for shape, value in features.shapes(
                g.data, mask=mask, transform=g.transform
            )
        ]

    def rasterize(
        self,
        items: Iterable[Union[Tuple[BaseGeometry, float], Tuple[float, float, float]]],
        all_touched: bool = False,
    ):
        def get_geometries():
            for item in items:
                if isinstance(item[0], BaseGeometry):
                    yield (item[0], float(item[1]))
                else:
                    yield (Point(item[:2]), float(item[2]))  # type: ignore

        array = features.rasterize(
            shapes=get_geometries(),
            out_shape=self.data.shape,
            fill=np.nan,  # type: ignore
            transform=self.transform,
            all_touched=all_touched,
            default_value=np.nan,  # type: ignore
        )
        return self.local(array)

    def to_stack(self):
        from glidergun._stack import stack

        grid1 = self.percent_clip(0.1, 99.9)
        grid2 = grid1 - grid1.min
        grid3 = grid2 / grid2.max
        arrays = plt.get_cmap(self.display)(grid3.data).transpose(2, 0, 1)[:3]  # type: ignore
        mask = self.is_nan()
        r, g, b = [self.local(a * 253 + 1).set_nan(mask) for a in arrays]
        return stack(r, g, b)

    def percentile(self, percent: float) -> float:
        return np.nanpercentile(self.data, percent)  # type: ignore

    def percent_clip(self, min_percent: float, max_percent: float):
        min_value = self.percentile(min_percent)
        max_value = self.percentile(max_percent)

        if min_value == max_value:
            return self

        return self.cap_range(min_value, max_value)

    def to_uint8_range(self):
        if self.dtype == "bool" or self.min > 0 and self.max < 255:
            return self
        return self.percent_clip(0.1, 99.9).stretch(1, 254)

    def reclass(self, *mapping: Tuple[float, float, float]):
        conditions = [(self.data >= min) & (self.data < max) for min, max, _ in mapping]
        values = [value for _, _, value in mapping]
        return self.local(np.select(conditions, values, np.nan))

    def slice(self, count: int, percent_clip: float = 0.1):
        min = self.percentile(percent_clip)
        max = self.percentile(100 - percent_clip)
        interval = (max - min) / count
        mapping = [
            (
                min + (i - 1) * interval if i > 1 else float("-inf"),
                min + i * interval if i < count else float("inf"),
                float(i),
            )
            for i in range(1, count + 1)
        ]
        return self.reclass(*mapping)

    def stretch(self, min_value: float, max_value: float):
        expected_range = max_value - min_value
        actual_range = self.max - self.min

        if actual_range == 0:
            n = (min_value + max_value) / 2
            return self * 0 + n

        return (self - self.min) * expected_range / actual_range + min_value

    def cap_range(self, min: Operand, max: Operand, set_nan: bool = False):
        return self.cap_min(min, set_nan).cap_max(max, set_nan)

    def cap_min(self, value: Operand, set_nan: bool = False):
        return con(self < value, np.nan if set_nan else value, self)

    def cap_max(self, value: Operand, set_nan: bool = False):
        return con(self > value, np.nan if set_nan else value, self)

    def kmeans_cluster(self, n_clusters: int, nodata: float = 0.0, **kwargs):
        g = self.is_nan().then(nodata, self)
        kmeans = KMeans(n_clusters=n_clusters, **kwargs).fit(g.data.reshape(-1, 1))
        data = kmeans.cluster_centers_[kmeans.labels_].reshape(self.data.shape)
        result = self.local(data)
        return result.set_nan(self.is_nan())

    def scale(self, scaler: Scaler, **fit_params):
        g = cast("Grid", self)
        result = g.local(lambda a: scaler.fit_transform(a, **fit_params))
        return result

    def hist(self, **kwargs):
        return plt.bar(list(self.bins.keys()), list(self.bins.values()), **kwargs)

    def color(self, cmap: Union[ColorMap, Any]):
        return dataclasses.replace(self, display=cmap)

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

    def type(self, dtype: DataType):
        if self.dtype == dtype:
            return self
        return self.local(lambda a: np.asanyarray(a, dtype=dtype))

    @overload
    def save(
        self, file: str, dtype: Optional[DataType] = None, driver: str = ""
    ) -> None: ...

    @overload
    def save(  # type: ignore
        self, file: MemoryFile, dtype: Optional[DataType] = None, driver: str = ""
    ) -> None: ...

    def save(self, file, dtype: Optional[DataType] = None, driver: str = ""):
        g = self * 1 if self.dtype == "bool" else self

        if isinstance(file, str) and (
            file.lower().endswith(".jpg")
            or file.lower().endswith(".kml")
            or file.lower().endswith(".kmz")
            or file.lower().endswith(".png")
        ):
            g.to_stack().save(file)
            return

        if dtype is None:
            dtype = g.dtype

        nodata = get_nodata_value(dtype)

        if nodata is not None:
            g = con(g.is_nan(), nodata, g)

        if isinstance(file, str):
            create_directory(file)
            with rasterio.open(
                file,
                "w",
                driver=driver if driver else driver_from_extension(file),
                count=1,
                dtype=dtype,
                nodata=nodata,
                **_metadata(self),
            ) as dataset:
                dataset.write(g.data, 1)
        elif isinstance(file, MemoryFile):
            with file.open(
                driver=driver if driver else "GTiff",
                count=1,
                dtype=dtype,
                nodata=nodata,
                **_metadata(self),
            ) as dataset:
                dataset.write(g.data, 1)


@overload
def grid(
    data: str,
    extent: Optional[Tuple[float, float, float, float]] = None,
    crs: Union[int, CRS, None] = None,
    cell_size: Union[Tuple[float, float], float, None] = None,
    index: int = 1,
) -> Grid:
    """Creates a new grid from the file path.

    Args:
        data: File path.
        extent: Map extent used to clip the raster.
        crs: CRS or an EPSG code (e.g. 4326).
        index (int, optional): Band index.  Defaults to 1.

    Example:
        >>> grid("n55_e008_1arc_v3.bil")
        image: 1801x3601 float32 | range: -28.000~101.000 | mean: 11.566 | std: 14.645 | crs: EPSG:4326 | cell: 0.000555555555555556, 0.000277777777777778

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(  # type: ignore
    data: DatasetReader,
    extent: Optional[Tuple[float, float, float, float]] = None,
    crs: Union[int, CRS, None] = None,
    cell_size: Union[Tuple[float, float], float, None] = None,
    index: int = 1,
) -> Grid:
    """Creates a new grid from data reader.

    Args:
        data: Data reader.
        extent: Map extent used to clip the raster.
        crs: CRS or an EPSG code (e.g. 4326).
        index (int, optional): Band index.  Defaults to 1.

    Example:
        >>> with rasterio.open("n55_e008_1arc_v3.bil") as dataset:
        ...     grid(dataset)
        ...
        image: 1801x3601 float32 | range: -28.000~101.000 | mean: 11.566 | std: 14.645 | crs: EPSG:4326 | cell: 0.000555555555555556, 0.000277777777777778

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(  # type: ignore
    data: MemoryFile,
    extent: Optional[Tuple[float, float, float, float]] = None,
    crs: Union[int, CRS, None] = None,
    cell_size: Union[Tuple[float, float], float, None] = None,
    index: int = 1,
) -> Grid:
    """Creates a new grid from a memory file.

    Args:
        data: Memory file.
        extent: Map extent used to clip the raster.
        crs: CRS or an EPSG code (e.g. 4326).
        index (int, optional): Band index.  Defaults to 1.

    Example:
        >>> grid(memory_file)
        image: 1801x3601 float32 | range: -28.000~101.000 | mean: 11.566 | std: 14.645 | crs: EPSG:4326 | cell: 0.000555555555555556, 0.000277777777777778

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(
    data: ndarray,
    extent: Union[Tuple[float, float, float, float], Affine, None] = None,
    crs: Union[int, CRS] = 4326,
) -> Grid:
    """Creates a new grid from an array.

    Args:
        data: Array.
        extent: Map extent.  Defaults to (0.0, 0.0, 1.0, 1.0).
        crs: CRS or an EPSG code.  Defaults to 4326.

    Example:
        >>> grid(np.arange(12).reshape(3, 4))
        image: 4x3 int32 | range: 0~11 | mean: 6 | std: 3 | crs: EPSG:4326 | cell: 0.25, 0.3333333333333333

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(
    data: Tuple[int, int],
    extent: Optional[Tuple[float, float, float, float]] = None,
    crs: Union[int, CRS] = 4326,
) -> Grid:
    """Creates a new grid from a tuple (width, height).

    Args:
        data: Tuple (width, height).
        extent: Map extent.  Defaults to (0.0, 0.0, 1.0, 1.0).
        crs: CRS or an EPSG code.  Defaults to 4326.

    Example:
        >>> grid((40, 30))
        image: 40x30 int32 | range: 0~1199 | mean: 600 | std: 346 | crs: EPSG:4326 | cell: 0.025, 0.03333333333333333

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(
    data: float,
    extent: Tuple[float, float, float, float],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
) -> Grid:
    """Creates a new grid from a constant value.

    Args:
        data: Constant value.
        extent: Map extent.
        crs: CRS or an EPSG code.
        cell_size: Cell size.

    Example:
        >>> grid(123.456, (-120, 40, -100, 60), 4326, 1.0)
        image: 20x20 float32 | range: 123.456~123.456 | mean: 123.456 | std: 0.000 | crs: EPSG:4326 | cell: 1.0, 1.0

    Returns:
        Grid: A new grid.
    """
    ...


@overload
def grid(
    data: Iterable[Union[Tuple[BaseGeometry, float], Tuple[float, float, float]]],
    extent: Tuple[float, float, float, float],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
) -> Grid:
    """Creates a new grid from an iterable of tuples (geometry, value) or (x, y, value).

    Args:
        data: Tuples (geometry, value) or (x, y, value).
        extent: Map extent.
        crs: CRS or an EPSG code.
        cell_size: Cell size.

    Example:
        >>> grid([(-119, 55, 1.23), (-104, 52, 4.56), (-112, 47, 7.89)], (-120, 40, -100, 60), 4326, 1.0)
        image: 20x20 float32 | range: 1.230~7.890 | mean: 4.560 | std: 2.719 | crs: EPSG:4326 | cell: 1.0, 1.0

    Returns:
        Grid: A new grid.
    """
    ...


def grid(
    data: Union[
        DatasetReader,
        str,
        MemoryFile,
        ndarray,
        Tuple[int, int],
        int,
        float,
        Iterable[Union[Tuple[float, float, float], Tuple[BaseGeometry, float]]],
    ],
    extent: Union[Tuple[float, float, float, float], Affine, None] = None,
    crs: Union[int, CRS, None] = None,
    cell_size: Union[Tuple[float, float], float, None] = None,
    index: int = 1,
) -> Grid:
    def read(dataset) -> Grid:
        g = _read(dataset, index, extent)
        return g if g is None or cell_size is None else g.resample(cell_size)  # type: ignore

    match data:
        case DatasetReader():
            return read(data)
        case str():
            with rasterio.open(data) as dataset:
                return read(dataset)
        case MemoryFile():
            with data.open() as dataset:
                return read(dataset)
        case ndarray():
            array = data
        case w, h if isinstance(w, int) and isinstance(h, int):
            array = np.arange(w * h).reshape(h, w)
        case _:
            assert extent, "Extent is required."
            assert cell_size, "Cell size is required."

            if isinstance(data, (int, float)):
                if isinstance(cell_size, (int, float)):
                    cell_size = CellSize(cell_size, cell_size)
                else:
                    cell_size = CellSize(*cell_size)
                extent = Extent(*extent)
                extent.assert_valid()
                xmin, ymin, xmax, ymax = extent
                width = int((xmax - xmin) / cell_size.x + 0.5)
                height = int((ymax - ymin) / cell_size.y + 0.5)
                xmin = xmax - cell_size.x * width
                ymax = ymin + cell_size.y * height
                transform = Affine(cell_size.x, 0, xmin, 0, -cell_size.y, ymax, 0, 0, 1)
                return grid(np.ones((height, width)) * data, transform, get_crs(crs))

            g = grid(np.nan, extent, get_crs(crs or 4326), cell_size)  # type: ignore
            return g.rasterize(data)  # type: ignore

    if isinstance(extent, Affine):
        transform = extent
    else:
        transform = from_bounds(
            *(extent or (0, 0, 1, 1)), array.shape[1], array.shape[0]
        )
    return Grid(format_type(array), transform, get_crs(crs or 4326))  # type: ignore


def _read(dataset, index, extent) -> Optional[Grid]:
    if extent:
        w = int(dataset.profile.data["width"])
        h = int(dataset.profile.data["height"])
        e1 = Extent(*extent)
        e2 = _extent(w, h, dataset.transform)
        e = e1.intersect(e2)
        left = (e.xmin - e2.xmin) / (e2.xmax - e2.xmin) * w
        right = (e.xmax - e2.xmin) / (e2.xmax - e2.xmin) * w
        top = (e2.ymax - e.ymax) / (e2.ymax - e2.ymin) * h
        bottom = (e2.ymax - e.ymin) / (e2.ymax - e2.ymin) * h
        width = right - left
        height = bottom - top
        if width > 0 and height > 0:
            window = Window(left, top, width, height)  # type: ignore
            data = dataset.read(index, window=window)
            g = grid(data, from_bounds(*e, width, height), dataset.crs)
        else:
            return None
    else:
        data = dataset.read(index)
        g = grid(data, dataset.transform, dataset.crs)
    return g if dataset.nodata is None else g.set_nan(dataset.nodata)


def _extent(width, height, transform) -> Extent:
    xmin = transform.c
    ymax = transform.f
    ymin = ymax + height * transform.e
    xmax = xmin + width * transform.a
    return Extent(xmin, ymin, xmax, ymax)


def _metadata(grid: Grid):
    return {
        "height": grid.height,
        "width": grid.width,
        "crs": grid.crs,
        "transform": grid.transform,
    }


def con(grid: Grid, trueValue: Operand, falseValue: Operand):
    """Evaluates a boolean grid .

    Args:
        grid: Boolean grid.
        trueValue: Values applied for cells evaluating to true.
        falseValue: Values applied for cells evaluating to false.

    Returns:
        Grid: A new grid.
    """
    return grid.local(
        lambda data: np.where(data, grid._data(trueValue), grid._data(falseValue))
    )


def standardize(
    *grids: Grid,
    extent: Union[Extent, ExtentResolution] = "intersect",
    cell_size: Union[Tuple[float, float], float, None] = None,
) -> Tuple[Grid, ...]:
    """Standardizes input grids to have the same map extent and cell size.

    Args:
        extent: Rule for extent.  Defaults to "intersect".
        cell_size: Cell size.  Defaults to None.

    Raises:
        ValueError: If grids are in different coordinate systems.

    Returns:
        Tuple[Grid, ...]: Standardized grids.
    """
    if len(grids) == 1:
        return tuple(grids)

    crs_set = set(grid.crs for grid in grids)

    if len(crs_set) > 1:
        raise ValueError("Input grids must have the same CRS.")

    if isinstance(cell_size, (int, float)):
        cell_size_standardized = (cell_size, cell_size)
    elif cell_size is None:
        cell_size_standardized = grids[0].cell_size
    else:
        cell_size_standardized = cell_size

    if isinstance(extent, Extent):
        extent_standardized = extent
    else:
        extent_standardized = grids[0].extent
        for g in grids:
            if extent == "intersect":
                extent_standardized = extent_standardized & g.extent
            elif extent == "union":
                extent_standardized = extent_standardized | g.extent

    results = []

    for g in grids:
        if g.cell_size != cell_size_standardized:
            g = g.resample(cell_size_standardized)
        if g.extent != extent_standardized:
            g = g.clip(*extent_standardized)  # type: ignore
        results.append(g)

    return tuple(results)


def _aggregate(func: Callable, *grids: Grid) -> Grid:
    grids_adjusted = standardize(*grids)
    data = func(np.array([grid.data for grid in grids_adjusted]), axis=0)
    return grids_adjusted[0].local(data)


def mean(*grids: Grid) -> Grid:
    return _aggregate(np.mean, *grids)


def std(*grids: Grid) -> Grid:
    return _aggregate(np.std, *grids)


def minimum(*grids: Grid) -> Grid:
    return _aggregate(np.min, *grids)


def maximum(*grids: Grid) -> Grid:
    return _aggregate(np.max, *grids)


def density(
    points: Sequence[Tuple[float, float]],
    extent: Tuple[float, float, float, float],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
    max_workers: int = 1,
):
    g = grid(np.nan, extent, crs, cell_size)

    if len(points) == 0:
        return g

    def f(g: Grid) -> Grid:
        kernel_density = KernelDensity()
        kernel_density.fit(np.array(points))
        result = np.exp(kernel_density.score_samples(g._coords()))
        return g.local(result.reshape(g.height, g.width))

    return g.process_tiles(f, 256, 0, max_workers)


def distance(
    points: Sequence[Tuple[float, float]],
    extent: Tuple[float, float, float, float],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
    max_workers: int = 1,
):
    g = grid(np.nan, extent, crs, cell_size)

    if len(points) == 0:
        return g

    def f(g: Grid) -> Grid:
        distances = cdist(g._coords(), np.array(points))
        result = distances.min(axis=1)
        return g.local(result.reshape(g.height, g.width))

    return g.process_tiles(f, 256, 0, max_workers)


def idw(
    points: Sequence[Tuple[float, float, float]],
    extent: Tuple[float, float, float, float],
    crs: Union[int, CRS],
    cell_size: Union[Tuple[float, float], float],
    radius: Optional[float] = None,
    max_workers: int = 1,
):
    g = grid(np.nan, extent, crs, cell_size)

    if len(points) == 0:
        return g

    def f(g: Grid) -> Grid:
        if radius:
            e = Extent(
                g.extent.xmin - radius,
                g.extent.ymin - radius,
                g.extent.xmax + radius,
                g.extent.ymax + radius,
            )
            points_adjusted = [p for p in points if e.intersects(*p[:2], *p[:2])]
            if not points_adjusted:
                return g
        else:
            points_adjusted = points

        x, y, v = zip(*points_adjusted)
        values = np.array(v)
        coords = g._coords()
        xi, yi = coords[:, 0], coords[:, 1]
        distances = np.asarray(
            (xi[:, None] - x) ** 2 + (yi[:, None] - y) ** 2, "float32"
        )
        distances[distances == 0] = 1e-10
        weights = 1 / distances

        if radius:
            mask = distances > radius**2
            distances[mask] = np.inf
            weights[mask] = 0

        result = np.dot(weights / weights.sum(axis=1, keepdims=True), values)
        return g.local(result.reshape(g.height, g.width))

    return g.process_tiles(f, 256, 0, max_workers)


def pca(n_components: int = 1, *grids: Grid) -> Tuple[Grid, ...]:
    grids_adjusted = [con(g.is_nan(), float(g.mean), g) for g in standardize(*grids)]
    arrays = (
        PCA(n_components=n_components)
        .fit_transform(
            np.array(
                [g.scale(StandardScaler()).data.ravel() for g in grids_adjusted]
            ).transpose((1, 0))
        )
        .transpose((1, 0))
    )
    g = grids_adjusted[0]
    return tuple(g.local(a.reshape((g.height, g.width))) for a in arrays)
