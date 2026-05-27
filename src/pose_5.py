import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()
    print(result)

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = None
    best_landmark = None
    best_marginal = float("inf")
    sum_of_marginals = 0

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            test_graph = gtsam.NonlinearFactorGraph(graph)
            test_estimate = gtsam.Values(initial_estimate)

            test_graph, test_estimate = add_pose(test_graph, test_estimate, pose_5)
            result = optimize(test_graph, test_estimate)

            test_graph = add_landmark_measurement(test_graph, result, pose_5, landmark)
            result = optimize(test_graph, test_estimate)

            marginals = gtsam.Marginals(test_graph, result)

            # Use this to decide the best pose/landmark
            current_marginal = marginals.marginalCovariance(L(landmark)).sum()

            # This is the returned sum for the selected final graph
            current_total_marginals = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )

            if current_marginal < best_marginal:
                best_marginal = current_marginal
                best_pose = pose_name
                best_landmark = landmark
                sum_of_marginals = current_total_marginals

    return best_pose, best_landmark, sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = None
    best_landmark = None
    best_error = float("inf")

    # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
    list_of_errors = []

    for pose_name, pose_5 in pose_options.items():
        best_error_for_this_pose = float("inf")

        for landmark in [1, 2]:
            test_graph = gtsam.NonlinearFactorGraph(graph)
            test_estimate = gtsam.Values(initial_estimate)

            test_graph, test_estimate = add_pose(test_graph, test_estimate, pose_5)
            result = optimize(test_graph, test_estimate)

            test_graph = add_landmark_measurement(test_graph, result, pose_5, landmark)
            result = optimize(test_graph, test_estimate)

            error = test_graph.error(result)

            if error < best_error_for_this_pose:
                best_error_for_this_pose = error

            if error < best_error:
                best_error = error
                best_pose = pose_name
                best_landmark = landmark

        list_of_errors.append(best_error_for_this_pose)

    # TODO: compute the sum of the errors and return it along with the best pose and landmark
    sum_of_errors = sum(list_of_errors)

    return best_pose, best_landmark, sum_of_errors