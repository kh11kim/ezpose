import pytest
import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation

from ezpose import SE3, SO3


def setup_function():
    global pose_single, pose_multiple
    pose_single = SE3.random()
    pose_multiple = SE3.random(10)


def is_SO3(rot: SO3):
    assert np.testing.assert_array_equal(
        rot.as_matrix() @ rot.as_matrix().T, np.eye(3)
    )


def is_SE3(pose: SE3):
    identity = pose @ pose.inv()
    np.testing.assert_almost_equal(identity.as_matrix(), np.eye(4))


def is_SE3_multiple(poses: SE3):
    identities = poses @ poses.inv()
    answer = np.eye(4)[None, ...].repeat(len(poses), axis=0)
    np.testing.assert_almost_equal(identities.as_matrix(), answer)


def test_SE3_multiply():
    is_SE3(pose_single @ pose_single)
    is_SE3_multiple(pose_single @ pose_multiple)
    is_SE3_multiple(pose_multiple @ pose_multiple)


def test_SO3_generation():
    xyzw = SO3.random().as_quat()
    rot = SO3.from_xyzw(xyzw)
    assert isinstance(rot, SO3)
    wxyz = rot.as_wxyz()
    xyzw2 = np.roll(wxyz, shift=-1)
    np.testing.assert_almost_equal(xyzw, xyzw2)


def test_equal():
    T = SE3.random()
    assert T == T
    T = SE3.random(10)
    assert all(T == T)


def _assert_strict_so3_instance(rot, *, single: Optional[bool] = None, length: Optional[int] = None):
    assert isinstance(rot, SO3)
    assert isinstance(rot, Rotation)
    assert rot.__class__ is SO3
    if single is not None:
        assert rot.single is single
    if length is not None:
        assert len(rot) == length


@pytest.mark.parametrize(
    ("factory", "expected_single", "expected_length"),
    [
        (lambda: SO3.from_quat([0.0, 0.0, 0.0, 1.0]), True, None),
        (lambda: SO3.from_quat(np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])), False, 2),
        (lambda: SO3.from_xyzw([0.0, 0.0, 0.0, 1.0]), True, None),
        (lambda: SO3.from_wxyz([1.0, 0.0, 0.0, 0.0]), True, None),
        (lambda: SO3.from_euler("ZYX", [0.0, 0.0, 0.0], degrees=True), True, None),
        (lambda: SO3.from_euler("ZYX", [[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]], degrees=True), False, 2),
        (lambda: SO3.from_rotvec([0.0, 0.0, 0.0]), True, None),
        (lambda: SO3.from_rotvec([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]), False, 2),
        (lambda: SO3.from_matrix(np.eye(3)), True, None),
        (lambda: SO3.from_matrix(np.stack([np.eye(3), np.eye(3)])), False, 2),
        (lambda: SO3.from_rot6d(np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])), True, None),
        (lambda: SO3.from_rot6d(np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])), False, 2),
        (lambda: SO3.identity(), True, None),
        (lambda: SO3.identity(0), False, 0),
        (lambda: SO3.identity(3), False, 3),
        (lambda: SO3.random(0), False, 0),
        (lambda: SO3.random(), True, None),
        (lambda: SO3.random(3), False, 3),
        (lambda: SO3.concatenate([SO3.identity(), SO3.identity()]), False, 2),
        (lambda: SO3.concatenate([SO3.identity(2), SO3.identity(3)]), False, 5),
        (lambda: SO3.concatenate([]), False, 0),
    ],
)
def test_so3_initializers_always_return_strict_so3_instances(factory, expected_single, expected_length):
    rot = factory()
    _assert_strict_so3_instance(rot, single=expected_single, length=expected_length)


@pytest.mark.parametrize(
    ("factory", "expected_single", "expected_length"),
    [
        (lambda: SO3.random(4)[0], True, None),
        (lambda: SO3.random(4)[:2], False, 2),
        (lambda: SO3.random().inv(), True, None),
        (lambda: SO3.random(4).inv(), False, 4),
        (lambda: SO3.random() * SO3.random(), True, None),
        (lambda: SO3.random(4) * SO3.identity(4), False, 4),
        (lambda: SO3.random() @ SO3.random(), True, None),
        (lambda: SO3.random(4) @ SO3.identity(4), False, 4),
        (lambda: SO3.identity().interpolate(SO3.random(), 0.5), True, None),
        (lambda: SO3.from_rot6d(SO3.random(3).as_rot6d()), False, 3),
    ],
)
def test_so3_operations_always_return_strict_so3_instances(factory, expected_single, expected_length):
    rot = factory()
    _assert_strict_so3_instance(rot, single=expected_single, length=expected_length)


def test_so3_single_item_is_not_subscriptable() -> None:
    with pytest.raises(TypeError):
        _ = SO3.random()[0]


def test_se3_from_xyz_qtn_uses_strict_so3() -> None:
    pose = SE3.from_xyz_qtn(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    _assert_strict_so3_instance(pose.rot, single=True, length=None)


def test_dfc_hand_info_regression_path_uses_strict_so3() -> None:
    quat = Rotation.from_euler("ZYX", [0.0, 0.0, 0.0], degrees=True).as_quat()
    pose = SE3.from_xyz_qtn(np.asarray([0.0, 0.0, 0.0, *quat], dtype=float))
    _assert_strict_so3_instance(pose.rot, single=True, length=None)


def test_se3_constructor_accepts_plain_scipy_rotation() -> None:
    pose = SE3(
        rot=Rotation.from_euler("ZYX", [10.0, 20.0, 30.0], degrees=True),
        trans=np.asarray([1.0, 2.0, 3.0]),
    )
    _assert_strict_so3_instance(pose.rot, single=True, length=None)
