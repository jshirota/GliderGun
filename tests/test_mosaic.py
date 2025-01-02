from glidergun._grid import Grid, grid
from glidergun._mosaic import mosaic


def test_mosaic_function_with_grids():
    g1 = grid((40, 30), (0, 0, 4, 3))
    g2 = grid((50, 40), (0, 0, 4, 3))
    result_grid = mosaic(g1, g2)
    assert isinstance(result_grid, Grid)
