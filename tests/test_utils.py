import numpy as np
import pytest

from src.utils import apply_filter, get_class_mapping, get_filter_coeffs


def test_get_filter_coeffs_shape_and_normalisation():
    b, a = get_filter_coeffs(order=5)
    assert len(b) == len(a) == 6
    assert a[0] == pytest.approx(1.0)


def test_apply_filter_attenuates_high_frequencies():
    fs = 360
    t = np.arange(187) / fs
    low = np.sin(2 * np.pi * 5 * t)
    high = np.sin(2 * np.pi * 120 * t)

    b, a = get_filter_coeffs(cutoff_freq=40, fs=fs)
    filtered = apply_filter(low + high, b, a)

    assert filtered.shape == (187,)
    high_residual = np.std(filtered - low)
    assert high_residual < np.std(high)


def test_class_mapping_covers_five_classes():
    mapping = get_class_mapping()
    assert set(mapping.values()) == {0, 1, 2, 3, 4}
    assert mapping["N"] == 0
    assert mapping["V"] == 2
