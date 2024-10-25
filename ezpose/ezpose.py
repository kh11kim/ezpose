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
    def __repr__(self): return f"SO3(qtn-wxyz):\n{np.array2string(self.as_wxyz(), separator=', ')}"
    @classmethod
    def from_wxyz(cls, wxyz: ArrayLike) -> SO3:
        xyzw = np.roll(wxyz, shift=-1)
        return cls.from_quat(xyzw)
    @classmethod
    def from_xyzw(cls, xyzw: ArrayLike) -> SO3:
        return cls.from_quat(xyzw)
    
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
    
    def __matmul__(self, target:Rotation) -> SO3 | np.ndarray:
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


class SE3:
    """
    A class to represent a 3D rigid transformation in SE(3).
    It is designed to be similar to the `scipy.spatial.transform.Rotation` class.
    """
    def __init__(
        self, 
        p:ArrayLike = np.zeros(3),
        rot:SO3 = SO3.identity(),
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
        self.p = np.array(p)
        self.rot = rot
        if len(self.p.shape) == 2 or not rot.single:
            assert self.p.shape[0] == len(rot)
    
    @property
    def single(self) -> bool: return self.rot.single
    def __len__(self): return len(self.rot)
    def __repr__(self): return f"SE3(xyz_qtn): \n{np.array2string(self.as_xyz_qtn(), separator=', ')}"
    def __getitem__(self, i): 
        if self.single: raise TypeError("Single transformation is not subscriptable.")
        return SE3(p=self.p[i], rot=self.rot[i])

    @classmethod
    def identity(cls) -> SE3: 
        """
        Return the identity transformation.

        Returns
        -------
        SE3
            The identity transformation.
        """
        return cls([0,0,0], SO3.identity())

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
        if num is None: trans: np.ndarray = np.random.uniform(0, 1, 3)
        else: trans: np.ndarray = np.random.uniform(0, 1, (num, 3))
        return cls(trans, rot)
    
    def as_xyz_qtn(self) -> np.ndarray:
        """
        Return the transformation as an array in x, y, z, w, x, y, z order.

        Returns
        -------
        xyz_qtn : (N, 7) or (7,) ndarray
            The transformation as an array in x, y, z, w, x, y, z order.
        """
        
        return np.hstack([self.p, self.rot.as_wxyz()])

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
            mat[:3, 3] = self.p
            return mat
        else:
            mat = np.repeat(np.eye(4)[None, ...], self.__len__(), axis=0)
            mat[:, :3, :3] = self.rot.as_matrix()
            mat[:, :3, 3] = self.p
            return mat
    
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
        if self.single: assert target.shape == (3,) or target.shape[1] == 3
        else: assert self.__len__() == target.shape[0] and target.shape[1] == 3

        return self.rot.apply(target) + self.p
            
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
        trans = self.rot.apply(other.p) + self.p
        return self.__class__(trans, rot)

    def inv(self) -> SE3:
        """
        Return the inverse of the transformation.

        Returns
        -------
        SE3
            The inverse of the transformation.
        """
        rot: SO3 = self.rot.inv()
        trans: np.ndarray = -rot.apply(self.p)
        return self.__class__(trans, rot)
    
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
    def look_at(cls, camera_pos:ArrayLike, target_pos=np.zeros(3), up_vector=np.array([0.,0, 1])):
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
        trans =np.asarray(camera_pos)
        return cls(trans, SO3.from_matrix(rot_mat))
