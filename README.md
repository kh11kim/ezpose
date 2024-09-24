# ezpose

A Python package for working with 3D transformations and poses. \
This package aims to extend scipy.spatial.transform.Rotation class to work with rigid transformations(i.e. SE(3) poses).

## Installation

To install this package, you can use pip:

```bash
pip install ezpose
```

## Usage

Here's an example of how to use this package:

```python
from ezpose import SE3, SO3

# Create a random 3D transformation
transform = SE3.random()

# Apply the transformation to a vector
vector = np.array([1, 2, 3])
transformed_vector = transform.apply(vector)

# Multiply two transformations
transform2 = SE3.random()
result = transform.multiply(transform2)
```

## API Reference

Here's a list of the classes and functions in this package:

- `SO3`: A class representing a 3D rotation matrix in SO(3).
  - this is a subclass of `scipy.spatial.transform.Rotation`
  
- `SE3`: A class representing a 3D transformation matrix in SE(3).
  - `random()`: A method to generate a random SE3 object.
  - `apply()`: A method to apply the transformation to a vector.
  - `multiply()`: A method to multiply two SE3 objects. (= pose1 @ pose2)
  - `inv()`: A method to return the inverse of the transformation.
  - `lookat()`: A method that returns the view matrix from the given camera position to the target position.
  - `as_matrix()`: A method to convert the transformation to a 4x4 matrix.
  - `as_xyz_qtn()`: A method to convert the transformation to a numpy array of (position, quaternion).

## License

This package is licensed under the MIT License. 
