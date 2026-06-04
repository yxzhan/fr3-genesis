import math

import numpy as np

import scene_robot_tables as s


def test_wrap_to_pi_scalar_matches_math():
    for a in [0.0, 3.0, -3.0, 7.0, -7.0, math.pi, -math.pi]:
        expected = math.atan2(math.sin(a), math.cos(a))
        assert np.isclose(s._wrap_to_pi(a), expected)


def test_wrap_to_pi_array_broadcasts():
    a = np.array([0.0, 7.0, -7.0, math.pi + 0.1])
    out = s._wrap_to_pi(a)
    assert out.shape == a.shape
    assert np.all(out >= -math.pi - 1e-9) and np.all(out <= math.pi + 1e-9)


def test_steering_alignment_scale_clamps_and_fades():
    # error >= ZERO threshold -> 0; error <= FULL threshold -> 1; linear between.
    assert np.isclose(s._steering_alignment_scale(s.STEERING_ZERO_SPEED_ERROR_RAD), 0.0)
    assert np.isclose(s._steering_alignment_scale(s.STEERING_FULL_SPEED_ERROR_RAD), 1.0)
    assert np.isclose(s._steering_alignment_scale(0.0), 1.0)  # clamped to 1
    assert np.isclose(s._steering_alignment_scale(math.pi), 0.0)  # clamped to 0


def test_steering_alignment_scale_array():
    err = np.array([0.0, s.STEERING_FULL_SPEED_ERROR_RAD, s.STEERING_ZERO_SPEED_ERROR_RAD, math.pi])
    out = s._steering_alignment_scale(err)
    assert out.shape == err.shape
    assert np.allclose(out, [1.0, 1.0, 0.0, 0.0])
