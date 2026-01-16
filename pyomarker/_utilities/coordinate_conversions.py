""" Utilities for coordinate conversions of dirichlet variates """

import numpy as np
from numpy.typing import NDArray


def cartesian2barycentric(x: NDArray, triangle: NDArray = None) -> NDArray:
    """ Convert 3 dimensional cartesian coordinates into barycentric coordinates

    Args:
        x (NDArray): Array of shape (n_samples, 3) simplexes.
        triangle (NDArray): Array of shape (2, 3), representing the cartesian coordinates in which
            to draw the triangle. Defaults is [[0, 1, 0.5],[0, 0, sqrt(0.75)]], an equilateral
            triangle.

    Returns:
        NDArray: Array of shape (n_samples, 2) comprising the converted coordinates
    """
    x = np.array(x)
    if triangle is None:
        triangle = np.array([[0, 0.5, 1.0], [0, np.sqrt(0.75), 0]])
    triangle = np.array(triangle)
    return np.dot(triangle, x.T).T


def barycentric2cartesian(b: NDArray, triangle: NDArray = None) -> NDArray:
    """ Convert 2 dimensional barycentric coordinates into cartesian coordinates

    Args:
        b (NDArray): Array of shape (n_samples, 2) representing the barycentric coordinates.
        triangle (NDArray): Array of shape (2, 3), representing the cartesian coordinates in which
            to draw the triangle. Defaults is [[0, 1, 0.5],[0, 0, sqrt(0.75)]], an equilateral
            triangle.

    Returns:
        NDArray: Array of shape (n_samples, 3) comprising the converted coordinates (simplexes)
    """

    b = np.array(b)
    if triangle is None:
        triangle = np.array([[0, 0.5, 1.0], [0, np.sqrt(0.75), 0]])
    triangle = np.array(triangle)

    t = np.matrix(np.c_[triangle[:, 0] - triangle[:, 2], triangle[:, 1] - triangle[:, 2]])
    ti = np.array(t.I)
    x = np.dot(ti, (b - triangle[:, 2]).T).T
    x = np.c_[x, 1.0 - np.sum(x, axis=1)]

    return x


def polar2cartesian(p: NDArray) -> NDArray:
    """ Convert polar coordinates into cartesian coordinates

    Args:
        p (NDArray) : Array of shape (n_samples, D) representing the polar coordinates.

    Returns:
        NDArray: Array of shape (n_samples, D+1) comprising the converted coordinates (simplexes).
    """
    n_samples, d_dims = p.shape
    x = np.zeros((n_samples, d_dims + 1))
    x[:, 0] = np.cos(p[:, 0])**2
    for i in range(1, d_dims):
        x[:, i] = np.prod(np.sin(p[:, 0:i])**2, axis=1) * np.cos(p[:, i])**2
    x[:, d_dims] = np.prod(np.sin(p)**2, axis=1)
    return x


def cartesian2polar(x: NDArray) -> NDArray:
    """ Convert cartesian coordinates into polar coordinates

    Args:
        x (NDArray) : Array of shape (n_samples, D) representing the cartesian coordinates
            (simplexes).

    Returns:
        NDArray: Array of shape (n_samples, D-1) comprising the converted coordinates.
    """
    n_samples, d_dims = x.shape
    p = np.zeros((n_samples, d_dims - 1))
    p[:, 0] = np.arccos(np.sqrt(x[:, 0]))
    for i in range(1, d_dims - 1):
        p[:, i] = np.arccos(np.sqrt(x[:, i])/np.prod(np.sin(p[:, 0:i]), axis=1))
    return p
