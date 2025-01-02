from base64 import b64encode
from typing import Any, Iterable, Optional, Union

import IPython
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from glidergun._grid import Grid
from glidergun._literals import ColorMap
from glidergun._stack import Stack
from glidergun._types import Extent


def get_folium_map(
    obj: Union[Grid, Stack],
    opacity: float,
    basemap,
    width: int,
    height: int,
    attribution: Optional[str],
    grayscale: bool = True,
    **kwargs,
):
    import folium
    import jinja2

    obj_4326 = obj.project(4326)

    extent = Extent(
        obj_4326.xmin, max(obj_4326.ymin, -85), obj_4326.xmax, min(obj_4326.ymax, 85)
    )

    if obj_4326.extent != extent:
        obj_4326 = obj_4326.clip(*extent)

    obj_3857 = obj_4326.project(3857)

    figure = folium.Figure(width=str(width), height=str(height))
    bounds = [[obj_4326.ymin, obj_4326.xmin], [obj_4326.ymax, obj_4326.xmax]]

    if isinstance(basemap, str) or basemap is None:
        if basemap:
            if not basemap.startswith("https://"):
                if basemap == "OpenStreetMap":
                    attribution = "&copy OpenStreetMap"
                else:
                    basemap = (
                        "https://{s}.basemaps.cartocdn.com/"
                        + basemap
                        + "/{z}/{x}/{y}{r}.png"
                    )
                    attribution = "&copy OpenStreetMap &copy CARTO"
            tile_layer = folium.TileLayer(basemap, attr=attribution)
        else:
            tile_layer = folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="&copy; Esri",
            )

        options = {"zoom_control": False, **kwargs}
        folium_map = folium.Map(tiles=tile_layer, **options).add_to(figure)
        folium_map.fit_bounds(bounds)  # type: ignore

        if grayscale:
            macro = folium.MacroElement().add_to(folium_map)
            macro._template = jinja2.Template(
                f"""
                {{% macro script(this, kwargs) %}}
                tile_layer_{tile_layer._id}.getContainer()
                    .setAttribute("style", "filter: grayscale(100%); -webkit-filter: grayscale(100%);")
                {{% endmacro %}}
            """
            )
    else:
        folium_map = basemap

    color: Any = obj.display

    folium.raster_layers.ImageOverlay(  # type: ignore
        image=f"data:image/png;base64, {b64encode(obj_3857.color(color)._thumbnail((20, 20))).decode()}",
        bounds=bounds,
        opacity=opacity,
    ).add_to(folium_map)

    return folium_map


def get_html(obj: Union[Grid, Stack, ArtistAnimation]):
    if isinstance(obj, ArtistAnimation):
        return f"<div>{obj.to_jshtml()}</div>"
    description = str(obj).replace("|", "<br />")
    return f'<div><div>{description}</div><img src="{obj.img}" /><div>{obj.extent}</div></div>'


def animate(
    grids: Iterable[Grid],
    cmap: Union[ColorMap, Any] = "gray",
    interval: int = 100,
):
    first = next(iter(grids))
    n = 5 / first.width
    figsize = (first.width * n, first.height * n)

    def iterate():
        yield first
        yield from grids

    figure = plt.figure(figsize=figsize, frameon=False)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.axis("off")
    frames = [[plt.imshow(g.data, cmap=cmap, animated=True)] for g in iterate()]
    plt.close()
    return ArtistAnimation(figure, frames, interval=interval, blit=True)


if ipython := IPython.get_ipython():  # type: ignore
    formatters = ipython.display_formatter.formatters  # type: ignore
    formatter = formatters["text/html"]
    formatter.for_type(Grid, get_html)
    formatter.for_type(Stack, get_html)
    formatter.for_type(ArtistAnimation, get_html)
    formatter.for_type(
        tuple,
        lambda items: (
            f"""
            <table>
                <tr style="text-align: left">
                    {"".join(f"<td>{get_html(item)}</td>" for item in items)}
                </tr>
            </table>
        """
            if all(isinstance(item, (Grid, Stack)) for item in items)
            else f"{items}"
        ),
    )
