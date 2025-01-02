from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import fiona
from shapely.geometry import Point, mapping

from glidergun._utils import create_directory

if TYPE_CHECKING:
    from glidergun._grid import Grid


@dataclass(frozen=True)
class Shapefile:
    def save_shapes(self, file: str, polygonize: bool = False) -> None:
        grid = cast("Grid", self)

        if polygonize:
            geometry = "Polygon"
            shapes = grid.to_polygons()
        else:
            geometry = "Point"
            shapes = [(Point(x, y), value) for x, y, value in grid.to_points()]

        schema = {"geometry": geometry, "properties": {"id": "int", "value": "float"}}

        create_directory(file)

        with fiona.open(
            file, "w", driver="ESRI Shapefile", crs=grid.crs, schema=schema
        ) as output:
            for i, (shape, value) in enumerate(shapes):
                output.write(
                    {
                        "geometry": mapping(shape),
                        "properties": {"id": i + 1, "value": value},
                    }
                )
