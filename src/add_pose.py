
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    odometry = gtsam.Pose2(
        2.0 * np.cos(np.deg2rad(45)),
        2.0 * np.sin(np.deg2rad(45)),
        np.deg2rad(90)
    )

    graph.add(
        gtsam.BetweenFactorPose2(
            X(3),
            X(4),
            odometry,
            ODOMETRY_NOISE
        )
    )

    pose4_initial = gtsam.Pose2(
        4.0 + 2.0 * np.cos(np.deg2rad(45)),
        2.0 * np.sin(np.deg2rad(45)),
        np.deg2rad(90)
    )

    initial_estimate.insert(X(4), pose4_initial)

    return graph, initial_estimate