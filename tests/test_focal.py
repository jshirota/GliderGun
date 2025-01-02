from typing import Tuple

import numpy as np
import pytest

from glidergun._grid import Grid, grid


class TestFocal:
    @pytest.fixture
    def sample_grid(self):
        data = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        return grid(data)

    def test_focal_count(self, sample_grid: Grid):
        result = sample_grid.focal_count(value=5, buffer=1, circle=False)
        expected = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_ptp(self, sample_grid: Grid):
        result = sample_grid.focal_ptp(buffer=1, circle=False)
        expected = np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, 8, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_median(self, sample_grid: Grid):
        result = sample_grid.focal_median(buffer=1, circle=False)
        expected = np.array(
            [
                [3.0, 3.5, 4.0],
                [4.5, 5.0, 5.5],
                [6.0, 6.5, 7.0],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_mean(self, sample_grid: Grid):
        result = sample_grid.focal_mean(buffer=1, circle=False)
        expected = np.array(
            [
                [3.0, 3.5, 4.0],
                [4.5, 5.0, 5.5],
                [6.0, 6.5, 7.0],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_std(self, sample_grid: Grid):
        result = sample_grid.focal_std(buffer=1, circle=False)
        expected = np.array(
            [
                [1.581139, 1.707825, 1.581139],
                [2.5, 2.581989, 2.5],
                [1.581139, 1.707825, 1.581139],
            ]
        )
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_focal_var(self, sample_grid: Grid):
        result = sample_grid.focal_var(buffer=1, circle=False)
        expected = np.array(
            [
                [2.5, 2.916667, 2.5],
                [6.25, 6.666667, 6.25],
                [2.5, 2.916667, 2.5],
            ]
        )
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_focal_min(self, sample_grid: Grid):
        result = sample_grid.focal_min(buffer=1, circle=False)
        expected = np.array(
            [
                [1, 1, 2],
                [1, 1, 2],
                [4, 4, 5],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_max(self, sample_grid: Grid):
        result = sample_grid.focal_max(buffer=1, circle=False)
        expected = np.array(
            [
                [5, 6, 6],
                [8, 9, 9],
                [8, 9, 9],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def test_focal_sum(self, sample_grid: Grid):
        result = sample_grid.focal_sum(buffer=1, circle=False)
        expected = np.array(
            [
                [12, 21, 16],
                [27, 45, 33],
                [24, 39, 28],
            ]
        )
        np.testing.assert_array_equal(result.data, expected)

    def get_focal_grids(self, g: Grid, func) -> Tuple[Grid, Grid]:
        extent = 0.2, 0.2, 0.3, 0.3
        g1 = g.resample(0.001, "bilinear")
        g2 = func(g1, buffer=20, circle=True, max_workers=1).clip(*extent)
        g3 = func(g1.clip(0.1, 0.1, 0.4, 0.4), buffer=20, circle=True).clip(*extent)
        return g2, g3

    def test_compare_focal_mean(self, sample_grid: Grid):
        g1, g2 = self.get_focal_grids(sample_grid, Grid.focal_mean)
        assert g1.md5 == g2.md5

    def test_compare_focal_std(self, sample_grid: Grid):
        g1, g2 = self.get_focal_grids(sample_grid, Grid.focal_std)
        assert g1.md5 == g2.md5

    def test_compare_focal_sum(self, sample_grid: Grid):
        g1, g2 = self.get_focal_grids(sample_grid, Grid.focal_sum)
        assert g1.md5 == g2.md5

    def test_compare_focal_ptp(self, sample_grid: Grid):
        g1, g2 = self.get_focal_grids(sample_grid, Grid.focal_ptp)
        assert g1.md5 == g2.md5
