import os
import pytest
import rasterio
import shutil
from glidergun import Stack, stack

landsat = stack(
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B1.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B2.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B3.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B4.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B5.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B6.TIF",
    ".data/LC08_L2SP_197021_20220324_20220330_02_T1_SR_B7.TIF",
)


def test_extract_bands():
    s = landsat.extract_bands(4, 3, 2)
    assert s.grids[0].md5 == landsat.grids[3].md5
    assert s.grids[1].md5 == landsat.grids[2].md5
    assert s.grids[2].md5 == landsat.grids[1].md5


def test_operators():
    s = landsat * 1000
    for g1, g2 in zip(landsat.grids, s.grids):
        assert pytest.approx(g2.min, 0.001) == g1.min * 1000
        assert pytest.approx(g2.max, 0.001) == g1.max * 1000


def test_operators_2():
    s = landsat / 1000
    for g1, g2 in zip(landsat.grids, s.grids):
        assert pytest.approx(g2.min, 0.001) == g1.min / 1000
        assert pytest.approx(g2.max, 0.001) == g1.max / 1000


def test_operators_3():
    s = landsat + 1000
    for g1, g2 in zip(landsat.grids, s.grids):
        assert pytest.approx(g2.min, 0.001) == g1.min + 1000
        assert pytest.approx(g2.max, 0.001) == g1.max + 1000


def test_operators_4():
    s = landsat - 1000
    for g1, g2 in zip(landsat.grids, s.grids):
        assert pytest.approx(g2.min, 0.001) == g1.min - 1000
        assert pytest.approx(g2.max, 0.001) == g1.max - 1000


def test_percent_clip():
    s = landsat.percent_clip(1, 99)
    for g1, g2 in zip(landsat.grids, s.grids):
        assert pytest.approx(g2.min, 0.001) == g1.percentile(1)
        assert pytest.approx(g2.max, 0.001) == g1.percentile(99)


def test_percent_clip_to_uint8_range():
    s = landsat.percent_clip_to_uint8_range()
    for g in s.grids:
        assert pytest.approx(g.min, 0.001) == 1
        assert pytest.approx(g.max, 0.001) == 254


def test_pca():
    s = landsat.pca(4)
    assert len(s.grids) == 4


def test_project():
    s = landsat.project(4326)
    assert s.crs.wkt.startswith('GEOGCS["WGS 84",DATUM["WGS_1984",')


def test_properties():
    assert landsat.width == 2010
    assert landsat.height == 2033
    assert landsat.dtype == "float32"


def test_properties_2():
    for g in landsat.grids:
        assert g.cell_size == landsat.cell_size
        assert g.crs == landsat.crs
        assert g.extent == landsat.extent
        assert g.width == landsat.width
        assert g.height == landsat.height
        assert g.dtype == landsat.dtype
        assert g.xmin == landsat.xmin
        assert g.ymin == landsat.ymin
        assert g.xmax == landsat.xmax
        assert g.ymax == landsat.ymax


def test_resample():
    s = landsat.resample(1000)
    assert pytest.approx(s.cell_size.x, 0.001) == 1000
    assert pytest.approx(s.cell_size.y, 0.001) == 1000
    for g in s.grids:
        assert pytest.approx(g.cell_size.x, 0.001) == 1000
        assert pytest.approx(g.cell_size.y, 0.001) == 1000


def test_resample_2():
    s = landsat.resample((1000, 600))
    assert pytest.approx(s.cell_size.x, 0.001) == 1000
    assert pytest.approx(s.cell_size.y, 0.001) == 600
    for g in s.grids:
        assert pytest.approx(g.cell_size.x, 0.001) == 1000
        assert pytest.approx(g.cell_size.y, 0.001) == 600


def save(s1: Stack, file: str, strict: bool = True):
    folder = ".output/test"
    file_path = f"{folder}/{file}"
    os.makedirs(folder, exist_ok=True)
    s1.save(file_path)
    s2 = stack(file_path)
    if strict:
        assert s2.md5s == s1.md5s
    assert s2.extent == s1.extent
    shutil.rmtree(folder)


def test_save():
    memory_file = rasterio.MemoryFile()
    landsat.save(memory_file)
    s = stack(memory_file)
    assert s.md5s == landsat.md5s


def test_save_2():
    save(landsat, "test_stack.img")


def test_save_3():
    save(landsat, "test_stack.tif")


def test_save_4():
    save(landsat.extract_bands(4, 3, 2), "test_stack.jpg", strict=False)


def test_save_5():
    save(landsat.extract_bands(4, 3, 2), "test_stack.png", strict=False)
