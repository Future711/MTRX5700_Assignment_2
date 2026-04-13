"""Two-dimensional ICP helpers for camera-LiDAR calibration."""

import math
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Original source: https://github.com/richardos/icp

def euclidean_distance(point1, point2):
    """Compute Euclidean distance between two points.

    Args:
        point1: First point sequence.
        point2: Second point sequence.

    Returns:
        Euclidean distance between the points.
    """
    a = np.array(point1)
    b = np.array(point2)

    return np.linalg.norm(a - b, ord=2)


def point_based_matching(point_pairs):
    """Estimate 2D rigid motion from matched point pairs.

    This implementation follows the approach from "Robot Pose Estimation in
    Unknown Environments by Matching 2D Range Scans" by F. Lu and E. Milios.

    Args:
        point_pairs: Matched point pairs as [((x, y), (x', y')), ...].

    Returns:
        Tuple of (rotation_angle, translation_x, translation_y), or
        (None, None, None) when no pairs are available.
    """

    x_mean = 0
    y_mean = 0
    xp_mean = 0
    yp_mean = 0
    n = len(point_pairs)

    if n == 0:
        return None, None, None

    for pair in point_pairs:

        (x, y), (xp, yp) = pair

        x_mean += x
        y_mean += y
        xp_mean += xp
        yp_mean += yp

    x_mean /= n
    y_mean /= n
    xp_mean /= n
    yp_mean /= n

    s_x_xp = 0
    s_y_yp = 0
    s_x_yp = 0
    s_y_xp = 0
    for pair in point_pairs:

        (x, y), (xp, yp) = pair

        s_x_xp += (x - x_mean)*(xp - xp_mean)
        s_y_yp += (y - y_mean)*(yp - yp_mean)
        s_x_yp += (x - x_mean)*(yp - yp_mean)
        s_y_xp += (y - y_mean)*(xp - xp_mean)

    rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
    translation_x = xp_mean - (x_mean*math.cos(rot_angle) - y_mean*math.sin(rot_angle))
    translation_y = yp_mean - (x_mean*math.sin(rot_angle) + y_mean*math.cos(rot_angle))

    return rot_angle, translation_x, translation_y


def icp(reference_points, points, max_iterations=100, distance_threshold=0.3, convergence_translation_threshold=1e-3,
        convergence_rotation_threshold=1e-4, point_pairs_threshold=10, verbose=False):
    """Align one 2D point set to another using Iterative Closest Point.

    Args:
        reference_points: Reference point set with shape (N, 2).
        points: Points to align with shape (M, 2).
        max_iterations: Maximum number of ICP iterations.
        distance_threshold: Maximum pairwise distance for correspondences.
        convergence_translation_threshold: Translation convergence threshold.
        convergence_rotation_threshold: Rotation convergence threshold.
        point_pairs_threshold: Minimum point pairs required to continue.
        verbose: Whether to print progress information.

    Returns:
        Tuple of (transformation_history, aligned_points).
    """

    transformation_history = []

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)

    for iter_num in range(max_iterations):
        if verbose:
            print('------ iteration', iter_num, '------')

        closest_point_pairs = []  # list of point correspondences for closest point rule

        distances, indices = nbrs.kneighbors(points)
        for nn_index in range(len(distances)):
            if distances[nn_index][0] < distance_threshold:
                closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))

        # if only few point pairs, stop process
        if verbose:
            print('number of pairs found:', len(closest_point_pairs))
        if len(closest_point_pairs) < point_pairs_threshold:
            if verbose:
                print('No better solution can be found (very few point pairs)!')
            break

        # compute translation and rotation using point correspondences
        closest_rot_angle, closest_translation_x, closest_translation_y = point_based_matching(closest_point_pairs)
        if closest_rot_angle is not None:
            if verbose:
                print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                print('Translation:', closest_translation_x, closest_translation_y)
        if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
            if verbose:
                print('No better solution can be found!')
            break

        # transform 'points' (using the calculated rotation and translation)
        c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
        rot = np.array([[c, -s],
                        [s, c]])
        aligned_points = np.dot(points, rot.T)
        aligned_points[:, 0] += closest_translation_x
        aligned_points[:, 1] += closest_translation_y

        # update 'points' for the next iteration
        points = aligned_points

        # update transformation history
        transformation_history.append(np.hstack((rot, np.array([[closest_translation_x], [closest_translation_y]]))))

        # check convergence
        if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                and (abs(closest_translation_x) < convergence_translation_threshold) \
                and (abs(closest_translation_y) < convergence_translation_threshold):
            if verbose:
                print('Converged!')
            break

    return transformation_history, points


def icp_per_line(reference_points, points, max_iterations=100, distance_threshold=0.3, convergence_translation_threshold=1e-3,
                convergence_rotation_threshold=1e-4, point_pairs_threshold=10, verbose=False):
    """Align per-line 2D point sets using Iterative Closest Point.

    Args:
        reference_points: List of reference point arrays.
        points: List of point arrays to align.
        max_iterations: Maximum number of ICP iterations.
        distance_threshold: Maximum pairwise distance for correspondences.
        convergence_translation_threshold: Translation convergence threshold.
        convergence_rotation_threshold: Rotation convergence threshold.
        point_pairs_threshold: Minimum point pairs required to continue.
        verbose: Whether to print progress information.

    Returns:
        Tuple of (transformation_history, aligned_points_array).
    """

    transformation_history = []

    assert len(reference_points) == len(points), "The list length of reference points mismatch with that of points"

    list_length = len(reference_points)

    nbrs = [NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(ref) for ref in reference_points]
    # for i in range(list_length):
    #     nbrs.append(NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points[i]))

    for iter_num in range(max_iterations):
        if verbose:
            print('------ iteration', iter_num, '------')

        closest_point_pairs = []  # list of point correspondences for closest point rule

        for i in range(list_length):
            distances, indices = nbrs[i].kneighbors(points[i])
            for nn_index in range(len(distances)):
                if distances[nn_index][0] < distance_threshold:
                    closest_point_pairs.append((points[i][nn_index], reference_points[i][indices[nn_index][0]]))

        # if only few point pairs, stop process
        if verbose:
            print('number of pairs found:', len(closest_point_pairs))
        if len(closest_point_pairs) < point_pairs_threshold:
            if verbose:
                print('No better solution can be found (very few point pairs)!')
            break

        # compute translation and rotation using point correspondences
        closest_rot_angle, closest_translation_x, closest_translation_y = point_based_matching(closest_point_pairs)
        if closest_rot_angle is not None:
            if verbose:
                print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                print('Translation:', closest_translation_x, closest_translation_y)
        if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
            if verbose:
                print('No better solution can be found!')
            break

        # transform 'points' (using the calculated rotation and translation)
        c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
        rot = np.array([[c, -s],
                        [s, c]])
        for i in range(list_length):
            aligned_points = np.dot(points[i], rot.T)
            aligned_points[:, 0] += closest_translation_x
            aligned_points[:, 1] += closest_translation_y

            # update 'points' for the next iteration
            points[i] = aligned_points

        # update transformation history
        transformation_history.append(np.hstack((rot, np.array([[closest_translation_x], [closest_translation_y]]))))

        # check convergence
        if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                and (abs(closest_translation_x) < convergence_translation_threshold) \
                and (abs(closest_translation_y) < convergence_translation_threshold):
            if verbose:
                print('Converged!')
            break

    points_array = np.vstack(points)

    return transformation_history, points_array

