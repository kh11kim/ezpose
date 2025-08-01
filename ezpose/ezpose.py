"""
pose.py

Author: Kanghyun Kim
Contact: kh11kim@kaist.ac.kr,  https://github.com/kh11kim

Description:
A class to represent and manipulate 6D poses.
This module is designed to provide utility functions for rotation and transformation,
similar to the `scipy.spatial.transform.Rotation` class.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation


class SO3(Rotation):
    """
    A class to represent a 3D rotation matrix in SO(3).
    It is a subclass of the `scipy.spatial.transform.Rotation` class.
    Please refer to the documentation of the `scipy.spatial.transform.Rotation` class for more details.
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
    """

    def __repr__(self):
        return (
            f"SO3(qtn-xyzw): {np.array2string(self.as_xyzw(), separator=', ')}"
        )

    @classmethod
    def from_wxyz(cls, wxyz: ArrayLike) -> SO3:
        xyzw = np.roll(wxyz, shift=-1)
        return cls.from_quat(xyzw)

    @classmethod
    def from_xyzw(cls, xyzw: ArrayLike) -> SO3:
        return cls.from_quat(xyzw)

    @classmethod
    def from_matrix(cls, mat: ArrayLike) -> SO3:
        assert isinstance(mat, np.ndarray) and mat.shape[-2:] == (3, 3)
        return super().from_matrix(np.array(mat))  # copy

    def as_rot6d(self) -> np.ndarray:
        mat = self.as_matrix()
        rx, ry = mat[..., 0], mat[..., 1]
        return np.concatenate([rx, ry], axis=-1)

    @classmethod
    def from_rot6d(cls, rot6d: np.ndarray):
        assert rot6d.shape[-1] == 6
        rx = rot6d[..., :3]
        ry = rot6d[..., 3:6]
        rz = np.cross(rx, ry, axisa=-1, axisb=-1)
        rotmat = np.stack([rx, ry, rz], axis=-1)
        return cls.from_matrix(rotmat)

    def as_wxyz(self) -> np.ndarray:
        """
        Return the quaternion as an array in w, x, y, z order.

        Returns
        -------
        wxyz : (N, 4) or (4,) ndarray
            The quaternion as an array in w, x, y, z order.
        """
        return np.roll(super().as_quat(), shift=1)

    def as_xyzw(self) -> np.ndarray:
        """
        Return the quaternion as an array in x, y, z, w order.

        Returns
        -------
        xyzw : (N, 4) or (4,) ndarray
            The quaternion as an array in x, y, z, w order.
        """
        return super().as_quat()

    def __matmul__(self, target: Rotation) -> SO3 | np.ndarray:
        """
        Overload the matrix multiply operator to perform the rotation operation.
        Note that this library prefer to use the @ operator to perform a multiplication of two rotations or poses.

        Parameters
        ----------
        target : Rotation
            The target vector or array of vectors to be rotated.

        Returns
        -------
        rotated_target : ndarray
            The rotated target vector or array of vectors.

        """
        return self.__mul__(target)

    def __eq__(self, other: SO3):
        return self.approx_equal(other)

    def interpolate(self, other: SO3, ratio: float):
        # delta = other.as_rotvec() - self.as_rotvec()
        # rotvec = self.as_rotvec() + ratio * delta
        rot_diff = self.inv() @ other
        return self @ SO3.from_rotvec(rot_diff.as_rotvec() * ratio)


class SE3:
    """
    A class to represent a 3D rigid transformation in SE(3).
    It is designed to be similar to the `scipy.spatial.transform.Rotation` class.
    """

    def __init__(
        self,
        rot: SO3 = None,
        trans: ArrayLike = None,
    ) -> SE3:
        """
        Initialize an SE3 object.

        Parameters
        ----------
        p : ArrayLike, optional
            The translation vector. The default is np.zeros(3).
        rot : SO3, optional
            The rotation matrix. The default is SO3.identity().

        Notes
        -----
        The translation vector and the rotation matrix can both be either single or multiple.
        If the translation vector is multiple, it is expected to have the same length as the rotation matrix.
        If the rotation matrix is multiple, it is expected to have the same length as the translation vector.
        """

        if trans is None and rot is None:
            trans, rot = np.zeros(3), SO3.identity()
        if trans is None:
            if rot.single:
                trans = np.zeros(3)
            else:
                trans = np.zeros((len(rot), 3))
        if rot is None:
            trans = np.array(trans)
            if len(trans.shape) == 1:
                rot = SO3.identity()
            else:
                rot = SO3.identity(trans.shape[0])

        self.trans = np.array(trans)
        self.rot = rot
        if len(self.trans.shape) == 2 or not rot.single:
            assert self.trans.shape[0] == len(rot)

    @property
    def single(self) -> bool:
        return self.rot.single

    def __len__(self):
        return len(self.rot)

    def __repr__(self):
        return f"SE3(xyz_qtn): \n{np.array2string(self.as_xyz_qtn(), separator=', ')}"

    def __getitem__(self, i):
        if self.single:
            raise TypeError("Single transformation is not subscriptable.")
        return SE3(trans=self.trans[i], rot=self.rot[i])

    def __eq__(self, other: SE3):
        rot_eq = self.rot == other.rot
        trans_eq = np.all(self.trans == other.trans, axis=-1)
        return rot_eq & trans_eq

    @classmethod
    def identity(cls) -> SE3:
        """
        Return the identity transformation.

        Returns
        -------
        SE3
            The identity transformation.
        """
        return cls(trans=[0, 0, 0], rot=SO3.identity())

    @classmethod
    def random(cls, num=None) -> SE3:
        """
        Generate a random SE3 object. The translation is uniformly distributed between 0 and 1.

        Parameters
        ----------
        num : int, optional
            The number of random SE3 objects to generate. The default is None (single).

        Returns
        -------
        SE3
            A random SE3 object.
        """

        rot: SO3 = SO3.random(num)
        if num is None:
            trans: np.ndarray = np.random.uniform(0, 1, 3)
        else:
            trans: np.ndarray = np.random.uniform(0, 1, (num, 3))
        return cls(trans=trans, rot=rot)

    @classmethod
    def concatenate(cls, poses):
        p = [pose.p for pose in poses]
        rot = [pose.rot for pose in poses]
        p = np.vstack(p)
        rot = SO3.concatenate(rot)
        return SE3(trans=p, rot=rot)

    @classmethod
    def from_matrix(cls, mat: ArrayLike) -> SE3:
        assert isinstance(mat, np.ndarray) and mat.shape[-2:] == (4, 4)
        rot: SO3 = SO3.from_matrix(mat[..., :3, :3])
        trans: np.ndarray = mat[..., :3, -1]
        return cls(rot=rot, trans=trans)

    def as_xyz_qtn(self, qtn_order="xyzw") -> np.ndarray:
        """
        Return the transformation as an array in x, y, z, w, x, y, z order.

        Returns
        -------
        xyz_qtn : (N, 7) or (7,) ndarray
            The transformation as an array in x, y, z, w, x, y, z order.
        """
        if qtn_order == "xyzw":
            qtn = self.rot.as_xyzw()
        elif qtn_order == "wxyz":
            qtn = self.rot.as_wxyz()
        return np.hstack([self.trans, qtn])

    def as_matrix(self) -> np.ndarray:
        """Return the transformation as a 4x4 matrix.

        Returns
        -------
        mat : (N, 4, 4) or (4, 4) ndarray
            The transformation as a 4x4 matrix.
        """
        if self.single:
            mat = np.eye(4)
            mat[:3, :3] = self.rot.as_matrix()
            mat[:3, 3] = self.trans
            return mat
        else:
            mat = np.repeat(np.eye(4)[None, ...], self.__len__(), axis=0)
            mat[:, :3, :3] = self.rot.as_matrix()
            mat[:, :3, 3] = self.trans
            return mat

    def as_pose9d(self) -> np.ndarray:
        return np.hstack([self.trans, self.rot.as_rot6d()])

    @classmethod
    def from_pose9d(cls, pose9d: np.ndarray) -> SE3:
        trans = pose9d[..., :3]
        rot = SO3.from_rot6d(pose9d[..., 3:])
        return cls(rot=rot, trans=trans)

    def apply(self, target: ArrayLike) -> ArrayLike:
        """
        Apply the transformation to the given target vector(s).

        Parameters
        ----------
        target : ArrayLike
            The target vector(s) to be transformed.

        Returns
        -------
        transformed_target : ArrayLike
            The transformed target vector(s).
        """
        target = np.asarray(target)
        if self.single:
            assert target.shape == (3,) or target.shape[1] == 3
        else:
            assert self.__len__() == target.shape[0] and target.shape[1] == 3

        return self.rot.apply(target) + self.trans

    def multiply(self, other: SE3) -> SE3:
        """
        Multiply this transformation with another SE3 object.

        Parameters
        ----------
        other : SE3
            The other transformation to be multiplied with.

        Returns
        -------
        SE3
            The resulting transformation.
        """
        rot = self.rot @ other.rot
        trans = self.rot.apply(other.trans) + self.trans
        return self.__class__(rot=rot, trans=trans)

    def inv(self) -> SE3:
        """
        Return the inverse of the transformation.

        Returns
        -------
        SE3
            The inverse of the transformation.
        """
        rot: SO3 = self.rot.inv()
        trans: np.ndarray = -rot.apply(self.trans)
        return self.__class__(rot=rot, trans=trans)

    def __matmul__(self, target: SE3) -> SE3:
        """
        Overload the matrix multiply operator to perform the transformation multiplication operation.
        This is equivalent to calling the `multiply` method.

        Parameters
        ----------
        target : SE3
            The other transformation to be multiplied with.

        Returns
        -------
        SE3
            The resulting transformation.
        """
        return self.multiply(target)

    @classmethod
    def look_at(
        cls,
        camera_pos: ArrayLike,
        target_pos=np.zeros(3),
        up_vector=np.array([0.0, 0, 1]),
    ):
        """
        Return the transformation that represents the view matrix from the given eye position to the target position.
        Coordinate convention is z-forward, y-down, x-right.

        Parameters
        ----------
        eye_pos : ArrayLike
            The eye position.
        target_pos : ArrayLike, optional
            The target position. The default is np.zeros(3).
        up_vector : ArrayLike, optional
            The up vector. The default is np.array([0.,0, 1]).

        Returns
        -------
        SE3
            The transformation that represents the view matrix from the given eye position to the target position.
        """
        forward = np.asarray(target_pos) - np.asarray(camera_pos)
        forward /= np.linalg.norm(forward)
        left = np.cross(forward, up_vector)
        left /= np.linalg.norm(left)
        up = np.cross(left, forward)
        rot_mat = np.vstack([left, -up, forward]).T
        trans = np.asarray(camera_pos)
        return cls(rot=SO3.from_matrix(rot_mat), trans=trans)

    def interpolate(self, other: SE3, ratio: float):
        """interpolate SO3, R3 seperately"""
        rot = self.rot.interpolate(other.rot, ratio)
        trans = self.trans + ratio * (other.trans - self.trans)
        return SE3(rot=rot, trans=trans)
