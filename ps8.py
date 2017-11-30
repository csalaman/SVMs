import numpy as np
import matplotlib.pyplot as plt
import sys
import math


def generate_training_data_binary(num):
    if num == 1:
        data = np.zeros((10, 3))
        for i in range(5):
            data[i] = [i - 5, 0, 1]
            data[i + 5] = [i + 1, 0, -1]

    elif num == 2:
        data = np.zeros((10, 3))
        for i in range(5):
            data[i] = [0, i - 5, 1]
            data[i + 5] = [0, i + 1, -1]

    elif num == 3:
        data = np.zeros((10, 3))
        data[0] = [3, 2, 1]
        data[1] = [6, 2, 1]
        data[2] = [3, 6, 1]
        data[3] = [4, 4, 1]
        data[4] = [5, 4, 1]
        data[5] = [-1, -2, -1]
        data[6] = [-2, -4, -1]
        data[7] = [-3, -3, -1]
        data[8] = [-4, -2, -1]
        data[9] = [-4, -4, -1]
    elif num == 4:
        data = np.zeros((10, 3))
        data[0] = [-1, 1, 1]
        data[1] = [-2, 2, 1]
        data[2] = [-3, 5, 1]
        data[3] = [-3, -1, 1]
        data[4] = [-2, 1, 1]
        data[5] = [3, -6, -1]
        data[6] = [0, -2, -1]
        data[7] = [-1, -7, -1]
        data[8] = [1, -10, -1]
        data[9] = [0, -8, -1]

    else:
        print("Incorrect num", num, "provided to generate_training_data_binary.")
        sys.exit()

    return data


def generate_training_data_multi(num):
    if num == 1:
        data = np.zeros((20, 3))
        for i in range(5):
            data[i] = [i - 5, 0, 1]
            data[i + 5] = [i + 1, 0, 2]
            data[i + 10] = [0, i - 5, 3]
            data[i + 15] = [0, i + 1, 4]
        C = 4

    elif num == 2:
        data = np.zeros((15, 3))
        data[0] = [-5, -5, 1]
        data[1] = [-3, -2, 1]
        data[2] = [-5, -3, 1]
        data[3] = [-5, -4, 1]
        data[4] = [-2, -9, 1]
        data[5] = [0, 6, 2]
        data[6] = [-1, 3, 2]
        data[7] = [-2, 1, 2]
        data[8] = [1, 7, 2]
        data[9] = [1, 5, 2]
        data[10] = [6, 3, 3]
        data[11] = [9, 2, 3]
        data[12] = [10, 4, 3]
        data[13] = [8, 1, 3]
        data[14] = [9, 0, 3]
        C = 3

    else:
        print("Incorrect num", num, "provided to generate_training_data_binary.")
        sys.exit()

    return [data, C]


def plot_training_data_binary(data, w, b):
    for item in data:
        if item[-1] == 1:
            plt.plot(item[0], item[1], 'b+')
        else:
            plt.plot(item[0], item[1], 'ro')
    m = max(data.max(), abs(data.min())) + 1
    plt.axis([-m, m, -m, m])

    line = np.linspace(-100, 100)
    if w[1] != 0:
        plt.plot(line, (-w[0] * line - b) / w[1])
    else:
        plt.axvline(x=b)

    plt.show()


def plot_training_data_multi(data):
    colors = ['b', 'r', 'g', 'm']
    shapes = ['+', 'o', '*', '.']

    for item in data:
        plt.plot(item[0], item[1], colors[int(item[2]) - 1] + shapes[int(item[2]) - 1])
    m = max(data.max(), abs(data.min())) + 1
    plt.axis([-m, m, -m, m])
    plt.show()


# Distance from point to hyperplane
def distance_point_to_hyperplane(pt, w, b):
    return abs(np.dot(pt, w) + b) / np.sqrt(np.dot(w, w))


# Compute minimum margin
def compute_margin(data, w, b):
    mini = float('inf')
    for pt in data:
        # Look for the smallest margin
        if distance_point_to_hyperplane(pt[:-1], w, b) < mini:
            mini = distance_point_to_hyperplane(pt[:-1], w, b)
        # Check if the point is classified correctly
        if svm_test_brute(w, b, pt) != pt[2]:
            return 0
    return mini


# Classifies data correctly
def classification_verification(data, w, b):
    for pt in data:
        if not (pt[-1] * (np.dot(w, pt[:-1]) + b) >= 1):
            return False
    return True


# Train SVM brute force
def svm_train_brute(data):
    pos = data[data[:, 2] == 1]
    neg = data[data[:, 2] == -1]
    max_margin = -float('inf')
    w_f = None
    b_f = None
    s_f = None

# 2 vectors; 1 positive and 1 negative
    for p1 in pos:
        for p2 in neg:
            mid_point = np.array([(float(p1[0] + p2[0])) / 2, (float(p1[1] + p2[1])) / 2])
            w = np.array(p1[:-1] - p2[:-1])
            w = w / np.sqrt(np.dot(w, w))
            b = -1 * (np.dot(w, (mid_point)))

            if max_margin == -float('inf'):
                max_margin = compute_margin(data, w, b)
                s_f = np.array([p1, p2])
                w_f = w
                b_f = b

            elif compute_margin(data, w, b) >= max_margin:
                max_margin = compute_margin(data, w, b)
                if distance_point_to_hyperplane(p1[:-1], w, b) < distance_point_to_hyperplane(s_f[0][:-1], w_f,
                                                                                              b_f):
                    s_f = np.array([p1, p2])
                    w_f = w
                    b_f = b


# 3 vectors; 2 positives and 1 negative
    for p1 in pos:
        for p2 in pos:
            if ((p1[0] != p2[0]) and (p1[1] != p2[1])):
                for q in neg:
                        # Get line between 2 positive points
                        v = (p2[:-1] - p1[:-1]) / np.sqrt(np.dot(p2[:-1] - p1[:-1], p2[:-1] - p1[:-1]))
                        # Get the projected point from the line
                        p = np.append(p1[:-1] + (np.dot(v, (q[:-1] - p1[:-1]))) * v, [1])

                        # Calculate the weights and bias
                        mid_point = np.array([(float(p[0] + q[0])) / 2, (float(p[1] + q[1])) / 2])
                        w = np.array(p[:-1] - q[:-1])
                        if np.sqrt(np.dot(w, w)) == 0:
                            break
                        w = w / np.sqrt(np.dot(w, w))
                        b = -1 * (np.dot(w, (mid_point)))

                        # Determine best params
                        if compute_margin(data, w, b) >= max_margin:
                            if distance_point_to_hyperplane(p[:-1], w, b) < distance_point_to_hyperplane(s_f[0][:-1],
                                                                                                              w_f,
                                                                                                              b_f):
                                  max_margin = compute_margin(data, w, b)
                                  s_f = np.array([p1, p2,q])
                                  w_f = w
                                  b_f = b


# 3 vectors; 1 positive and 2 negatives
    for q1 in neg:
        for q2 in neg:
            if ((q1[0] != q2[0]) and (q1[1] != q2[1])):
                for p in pos:
                    # Get line between 2 positive points
                    v = (q2[:-1] - q1[:-1]) / np.sqrt(np.dot(q2[:-1] - q1[:-1], q2[:-1] - q1[:-1]))
                    # Get the projected point from the line
                    q = np.append(q1[:-1] + (np.dot(v, (p[:-1] - q1[:-1]))) * v, [-1])

                    # Calculate the weights and bias
                    mid_point = np.array([(float(p[0] + q[0])) / 2, (float(p[1] + q[1])) / 2])
                    w = np.array(p[:-1] - q[:-1])
                    if np.sqrt(np.dot(w, w)) == 0:
                        break
                    w = w / np.sqrt(np.dot(w, w))
                    b = -1 * (np.dot(w, (mid_point)))

                    # Determine best params
                    if compute_margin(data, w, b) >= max_margin:
                        if distance_point_to_hyperplane(p[:-1], w, b) < distance_point_to_hyperplane(s_f[0][:-1],
                                                                                                          w_f,
                                                                                                          b_f):
                                max_margin = compute_margin(data, w, b)
                                s_f = np.array([p,q1, q1])
                                w_f = w
                                b_f = b


    return [w_f, b_f, s_f]  # SVM params


def svm_test_brute(w, b, x):
    if np.dot(w, x[:-1]) + b > 0:
        return 1
    else:
        return -1


# # da = np.array([[4,2,-1],[6,1,1]])
# da = generate_training_data_binary(4)
# [w, b, s] = svm_train_brute(da)
# plot_training_data_binary(da, w, b)
# print "W: ", w
# print "b: ", b
# print "s: ", s, "\n"
# print da
