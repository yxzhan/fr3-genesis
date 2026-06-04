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


import math as _math


def _ref_scalar_drive_targets(cur_steer_angles, vx, vy, wz):
    """Frozen copy of the ORIGINAL scalar compute_drive_targets, used as the
    equivalence oracle for the vectorized version."""
    steer_targets = np.zeros(len(s.DRIVE_MODULES))
    drive_targets = np.zeros(len(s.DRIVE_MODULES))
    vectors = []
    max_speed = 0.0
    for (_s, _d, x, y) in s.DRIVE_MODULES:
        wvx = vx - wz * y
        wvy = vy + wz * x
        sp = _math.hypot(wvx, wvy)
        vectors.append((wvx, wvy, sp))
        max_speed = max(max_speed, sp)
    allowed = s.MAX_WHEEL_SPEED_RADPS * s.WHEEL_RADIUS_M
    scale = allowed / max_speed if max_speed > allowed else 1.0
    for i, (wvx, wvy, sp) in enumerate(vectors):
        wvx *= scale
        wvy *= scale
        sp *= scale
        cur = cur_steer_angles[i]
        if sp < s.STOP_EPS:
            steer_targets[i] = cur
            continue
        raw = _math.atan2(wvy, wvx)
        direct = _math.atan2(_math.sin(raw - cur), _math.cos(raw - cur))
        flipped = _math.atan2(_math.sin(raw + _math.pi - cur), _math.cos(raw + _math.pi - cur))
        use_flipped = abs(flipped) < abs(direct)
        delta = flipped if use_flipped else direct
        steer_targets[i] = cur + delta
        align = max(0.0, min(1.0, (s.STEERING_ZERO_SPEED_ERROR_RAD - abs(delta)) /
                              (s.STEERING_ZERO_SPEED_ERROR_RAD - s.STEERING_FULL_SPEED_ERROR_RAD)))
        wheel_speed = (sp / s.WHEEL_RADIUS_M) * align
        drive_targets[i] = -wheel_speed if use_flipped else wheel_speed
    return steer_targets, drive_targets


def test_compute_drive_targets_batched_matches_scalar_reference():
    rng = np.random.default_rng(0)
    M = len(s.DRIVE_MODULES)
    twists = [
        (0.0, 0.0, 0.0),       # stopped
        (0.5, 0.0, 0.0),       # forward
        (0.0, 0.5, 0.0),       # strafe
        (0.0, 0.0, 1.2),       # rotate
        (2.0, -1.0, 0.7),      # fast mixed (triggers speed limit)
    ]
    cur = rng.uniform(-math.pi, math.pi, size=(len(twists), M))
    vx = np.array([t[0] for t in twists])
    vy = np.array([t[1] for t in twists])
    wz = np.array([t[2] for t in twists])

    steer, drive = s.compute_drive_targets(cur, vx, vy, wz)
    assert steer.shape == (len(twists), M)
    assert drive.shape == (len(twists), M)

    for i, (tvx, tvy, twz) in enumerate(twists):
        ref_steer, ref_drive = _ref_scalar_drive_targets(cur[i], tvx, tvy, twz)
        assert np.allclose(steer[i], ref_steer, atol=1e-9), f"steer env {i}"
        assert np.allclose(drive[i], ref_drive, atol=1e-9), f"drive env {i}"


def test_quat_to_yaw_scalar_and_batched():
    # Identity quat -> yaw 0. 90 deg about z -> yaw pi/2. Quats are (w, x, y, z).
    import math
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * (math.pi / 2)
    q_90 = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])

    assert np.isclose(s.quat_to_yaw(q_id), 0.0)
    assert np.isclose(s.quat_to_yaw(q_90), math.pi / 2)

    batched = np.stack([q_id, q_90])
    out = s.quat_to_yaw(batched)
    assert out.shape == (2,)
    assert np.allclose(out, [0.0, math.pi / 2])
