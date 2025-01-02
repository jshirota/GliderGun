import numpy as np

from glidergun._grid import grid
from glidergun._utils import format_type, get_nodata_value


def test_format_type_float64():
    data = np.array([1.0, 2.0, 3.0], dtype="float64")
    formatted_data = format_type(data)
    assert formatted_data.dtype == "float32"
    assert np.array_equal(formatted_data, np.array([1.0, 2.0, 3.0], dtype="float32"))


def test_format_type_int64():
    data = np.array([1, 2, 3], dtype="int64")
    formatted_data = format_type(data)
    assert formatted_data.dtype == "int32"
    assert np.array_equal(formatted_data, np.array([1, 2, 3], dtype="int32"))


def test_format_type_uint64():
    data = np.array([1, 2, 3], dtype="uint64")
    formatted_data = format_type(data)
    assert formatted_data.dtype == "uint32"
    assert np.array_equal(formatted_data, np.array([1, 2, 3], dtype="uint32"))


def test_format_type_float32():
    data = np.array([1.0, 2.0, 3.0], dtype="float32")
    formatted_data = format_type(data)
    assert formatted_data.dtype == "float32"
    assert np.array_equal(formatted_data, data)


def test_format_type_int32():
    data = np.array([1, 2, 3], dtype="int32")
    formatted_data = format_type(data)
    assert formatted_data.dtype == "int32"
    assert np.array_equal(formatted_data, data)


def test_format_type_uint32():
    data = np.array([1, 2, 3], dtype="uint32")
    formatted_data = format_type(data)
    assert formatted_data.dtype == "uint32"
    assert np.array_equal(formatted_data, data)


def test_get_nodata_value_bool():
    assert get_nodata_value("bool") is None


def test_get_nodata_value_float32():
    assert get_nodata_value("float32") == float(np.finfo("float32").min)


def test_get_nodata_value_float64():
    assert get_nodata_value("float64") == float(np.finfo("float64").min)


def test_get_nodata_value_uint8():
    assert get_nodata_value("uint8") == np.iinfo("uint8").max


def test_get_nodata_value_uint16():
    assert get_nodata_value("uint16") == np.iinfo("uint16").max


def test_get_nodata_value_uint32():
    assert get_nodata_value("uint32") == np.iinfo("uint32").max


def test_get_nodata_value_int8():
    assert get_nodata_value("int8") == np.iinfo("int8").min


def test_get_nodata_value_int16():
    assert get_nodata_value("int16") == np.iinfo("int16").min


def test_get_nodata_value_int32():
    assert get_nodata_value("int32") == np.iinfo("int32").min


def test_get_nodata_value_int64():
    assert get_nodata_value("int64") == np.iinfo("int64").min


def test_process_tiles():
    g = grid(".data/n55_e008_1arc_v3.bil")

    def assert_eq(g2):
        assert g2.md5 == g.md5
        assert g2.cell_size == g.cell_size
        assert g2.crs == g.crs
        assert g2.extent == g.extent

    assert_eq(g.process_tiles(lambda x: x, 456, 0, 2))
    assert_eq(g.process_tiles(lambda x: x, 678, 7, 2))
    assert_eq(g.process_tiles(lambda x: x, 3456, 0, 1))
    assert_eq(g.process_tiles(lambda x: x, 7891, 19, 1))
