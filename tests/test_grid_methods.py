import numpy as np

from glidergun._grid import grid


def test_grid_reclass():
    g = grid(np.array([[1, 2], [3, 4]]))
    result = g.reclass((1, 2, 10), (2, 3, 20), (3, 4, 30))
    expected = grid(np.array([[10, 20], [30, np.nan]]))
    assert np.array_equal(result.data, expected.data, equal_nan=True)


def test_grid_percentile():
    g = grid(np.array([[1, 2], [3, 4]]))
    result = g.percentile(50)
    expected = 2.5
    assert result == expected


def test_grid_slice():
    g = grid(np.array([[1, 2], [3, 4]]))
    result = g.slice(2)
    expected = grid(np.array([[1, 1], [2, 2]]))
    assert np.array_equal(result.data, expected.data)


def test_grid_replace():
    g = grid(np.array([[1, 2], [3, 4]]))
    result = g.con(2, 20)
    expected = grid(np.array([[1, 20], [3, 4]]))
    assert np.array_equal(result.data, expected.data)


def test_grid_set_nan():
    g = grid(np.array([[1, 2], [3, 4]]))
    result = g.set_nan(2)
    expected = grid(np.array([[1, np.nan], [3, 4]]))
    assert np.array_equal(result.data, expected.data, equal_nan=True)


def test_grid_value():
    g = grid(np.array([[1, 2], [3, 4]]), extent=(0, 0, 2, 2))
    result = g.value_at(1, 1)
    expected = 4
    assert result == expected


def test_grid_interp_clough_tocher():
    points = [(1, 1, 10), (4, 7, 40), (8, 2, 7)]
    extent = (0, 0, 10, 10)
    g = grid(points, extent, 4326, 1).interp_clough_tocher()
    assert g.extent == extent
    assert g.crs == 4326
    assert g.cell_size == (1.0, 1.0)
    assert g.value_at(2, 2) == 12.54273509979248
    assert g.has_nan is True
    assert g.md5 == "292a3f1ccc57fb50ac1dd8fa183cef3d"


def test_grid_interp_linear():
    points = [(1, 1, 10), (4, 7, 40), (8, 2, 7)]
    extent = (0, 0, 10, 10)
    g = grid(points, extent, 4326, 1).interp_linear()
    assert g.extent == extent
    assert g.crs == 4326
    assert g.cell_size == (1.0, 1.0)
    assert g.value_at(2, 2) == 12.54273509979248
    assert g.has_nan is True
    assert g.md5 == "993f7adb314bf2f48f447fd685230711"


def test_grid_interp_nearest():
    points = [(40, 30, 123), (30, 34, 777)]
    extent = (28, 28, 42, 36)
    g = grid(points, extent, 4326, 1).interp_nearest()
    assert g.extent == extent
    assert g.crs == 4326
    assert g.cell_size == (1.0, 1.0)
    assert g.value_at(30, 30) == 777
    assert g.value_at(40, 30) == 123
    assert g.has_nan is False
    assert g.md5 == "4de29fbe3e04e25ee55c3eff4cf553d0"


def test_grid_interp_rbf():
    points = [(1, 1, 10), (4, 7, 40), (8, 2, 7)]
    extent = (0, 0, 10, 10)
    g = grid(points, extent, 4326, 1).interp_rbf()
    assert g.extent == extent
    assert g.crs == 4326
    assert g.cell_size == (1.0, 1.0)
    assert g.value_at(2, 2) == 12.54273509979248
    assert g.has_nan is False
    assert g.md5 == "47087fd42c4e7e7e79c49cb53604b99a"


def test_grid_interp_compare():
    points = [(1, 1, 10), (4, 7, 40), (8, 2, 7)]
    extent = (0, 0, 10, 10)
    g1 = grid(points, extent, 4326, 1).interp_linear()
    g2 = grid(points, extent, 4326, 1).interp_rbf()
    g3 = g1 - g2
    assert g3.min == g3.max == 0

    points = [(1, 1, 10), (4, 7, 40), (8, 2, 7), (9, 2, 7)]
    extent = (0, 0, 10, 10)
    g1 = grid(points, extent, 4326, 1).interp_linear()
    g2 = grid(points, extent, 4326, 1).interp_rbf()
    g3 = g1 - g2
    assert g3.min != g3.max


def test_slope():
    g = grid(
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
    )

    assert g.slope().min == 0
    assert g.slope(True).min == 0
