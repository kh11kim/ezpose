import pytest
import numpy as np
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
