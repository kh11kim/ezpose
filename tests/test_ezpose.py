import pytest
import numpy as np
from ezpose import SE3, SO3
from scipy.spatial.transform import Rotation


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


def _assert_strict_so3_instance(rot, *, single: bool | None = None, length: int | None = None):
    assert isinstance(rot, SO3)
    assert isinstance(rot, Rotation)
    assert rot.__class__ is SO3
    if single is not None:
        assert rot.single is single
    if length is not None:
        assert len(rot) == length


@pytest.mark.parametrize(
    ("name", "factory", "expected_single", "expected_length"),
    [
        ("from_quat_single", lambda: SO3.from_quat([0.0, 0.0, 0.0, 1.0]), True, None),
        (
            "from_quat_multi",
            lambda: SO3.from_quat(np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])),
            False,
            2,
        ),
        ("from_xyzw_single", lambda: SO3.from_xyzw([0.0, 0.0, 0.0, 1.0]), True, None),
        ("from_wxyz_single", lambda: SO3.from_wxyz([1.0, 0.0, 0.0, 0.0]), True, None),
        ("from_euler_single", lambda: SO3.from_euler("ZYX", [0.0, 0.0, 0.0], degrees=True), True, None),
        (
            "from_euler_multi",
            lambda: SO3.from_euler("ZYX", [[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]], degrees=True),
            False,
            2,
        ),
        ("from_matrix_single", lambda: SO3.from_matrix(np.eye(3)), True, None),
        ("from_matrix_multi", lambda: SO3.from_matrix(np.stack([np.eye(3), np.eye(3)])), False, 2),
        ("from_rot6d_single", lambda: SO3.from_rot6d(np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])), True, None),
        (
            "from_rot6d_multi",
            lambda: SO3.from_rot6d(
                np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    ]
                )
            ),
            False,
            2,
        ),
        ("identity_single", lambda: SO3.identity(), True, None),
        ("identity_zero", lambda: SO3.identity(0), False, 0),
        ("identity_multi", lambda: SO3.identity(3), False, 3),
        ("random_zero", lambda: SO3.random(0), False, 0),
        ("random_single", lambda: SO3.random(), True, None),
        ("random_multi", lambda: SO3.random(3), False, 3),
        (
            "concatenate_singletons",
            lambda: SO3.concatenate([SO3.identity(), SO3.identity()]),
            False,
            2,
        ),
        (
            "concatenate_batches",
            lambda: SO3.concatenate([SO3.identity(2), SO3.identity(3)]),
            False,
            5,
        ),
    ],
)
def test_so3_initializers_always_return_strict_so3_instances(
    name: str,
    factory,
    expected_single: bool,
    expected_length: int | None,
) -> None:
    _ = name
    rot = factory()
    _assert_strict_so3_instance(rot, single=expected_single, length=expected_length)


@pytest.mark.parametrize(
    ("name", "factory", "expected_single", "expected_length"),
    [
        ("getitem_single_from_batch", lambda: SO3.random(4)[0], True, None),
        ("getitem_slice_from_batch", lambda: SO3.random(4)[:2], False, 2),
        ("inv_single", lambda: SO3.random().inv(), True, None),
        ("inv_multi", lambda: SO3.random(4).inv(), False, 4),
        ("mul_single", lambda: SO3.random() * SO3.random(), True, None),
        ("mul_multi", lambda: SO3.random(4) * SO3.identity(4), False, 4),
        ("matmul_single", lambda: SO3.random() @ SO3.random(), True, None),
        ("matmul_multi", lambda: SO3.random(4) @ SO3.identity(4), False, 4),
        ("interpolate_single", lambda: SO3.identity().interpolate(SO3.random(), 0.5), True, None),
        (
            "from_rot6d_roundtrip",
            lambda: SO3.from_rot6d(SO3.random(3).as_rot6d()),
            False,
            3,
        ),
    ],
)
def test_so3_operations_always_return_strict_so3_instances(
    name: str,
    factory,
    expected_single: bool,
    expected_length: int | None,
) -> None:
    _ = name
    rot = factory()
    _assert_strict_so3_instance(rot, single=expected_single, length=expected_length)


def test_so3_concatenate_empty_returns_empty_strict_so3_instance() -> None:
    rot = SO3.concatenate([])
    _assert_strict_so3_instance(rot, single=False, length=0)


def test_so3_single_item_is_not_subscriptable() -> None:
    with pytest.raises(TypeError):
        _ = SO3.random()[0]
