import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import copy


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
    elif num == 5:
        data = np.zeros((4,3))
        data[0] = [-1,-1,-1]
        data[1] = [-1,1,1]
        data[2] = [1,1,-1]
        data[3] = [1,-1,1]
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


def plot_training_data_binary(data):
    for item in data:
        if item[-1] == 1:
            plt.plot(item[0], item[1], 'b+')
        else:
            plt.plot(item[0], item[1], 'ro')
    m = max(data.max(), abs(data.min())) + 1
    plt.axis([-m, m, -m, m])
    plt.show()


def plot_training_data_multi(data):
    colors = ['b', 'r', 'g', 'm']
    shapes = ['+', 'o', '*', '.']

    for item in data:
        plt.plot(item[0], item[1], colors[int(item[2]) - 1] + shapes[int(item[2]) - 1])
    m = max(data.max(), abs(data.min())) + 1
    plt.axis([-m, m, -m, m])

    plt.show()

def plot_hyper_binary(w,b,data):
    line = np.linspace(-100, 100)
    if w[1] != 0:
        plt.plot(line, (-w[0] * line - b) / w[1])
    else:
        plt.axvline(x=b)
    plot_training_data_binary(data)



def plot_hyper_multi(W,B,(data,C)):
    for w,b in zip(W,B):
        line = np.linspace(-100, 100)
        if w[0] == 0:
            plt.axvline(x =-b/w[1])
            plt.plot(line, (-w[0] * line - b) / w[1])
        elif w[1] == 0:
            plt.axvline(x= -b/w[0])
        else:
            plt.axvline(x=-b / w[1])
    plot_training_data_multi(data)




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
                s_f = np.array([p1, p2])
                w_f = w
                b_f = b
                # if distance_point_to_hyperplane(p1[:-1], w, b) < distance_point_to_hyperplane(s_f[0][:-1], w_f,
                #                                                                               b_f):
                #     s_f = np.array([p1, p2])
                #     w_f = w
                #     b_f = b

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
                            max_margin = compute_margin(data, w, b)
                            s_f = np.array([p1, p2, q])
                            w_f = w
                            b_f = b
                        # if distance_point_to_hyperplane(p1[:-1], w, b) < distance_point_to_hyperplane(s_f[0][:-1], w_f,
                        #                                                                               b_f):
                        #     s_f = np.array([p1, p2])
                        #     w_f = w
                        #     b_f = b

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
                        max_margin = compute_margin(data, w, b)
                        s_f = np.array([p,q1, q1])
                        w_f = w
                        b_f = b
                        # if distance_point_to_hyperplane(p1[:-1], w, b) < distance_point_to_hyperplane(s_f[0][:-1], w_f,
                        #                                                                               b_f):
                        #     s_f = np.array([p1, p2])
                        #     w_f = w
                        #     b_f = b

    return [w_f, b_f, s_f]  # SVM params


def svm_test_brute(w, b, x):
    if np.dot(w, x[:-1]) + b > 0:
        return 1
    else:
        return -1

############ Part 2 ##################
def svm_train_multiclass((data,C)):
    W = []
    B = []

    for i in range(1,C+1):
        arr_copy = copy.deepcopy(data)
        arr_copy[np.where(arr_copy[:, 2] != i), 2] = -1
        arr_copy[np.where(arr_copy[:,2] == i),2] = 1
        [w,b,s] = svm_train_brute(arr_copy)

        W.append(w)
        B.append(b)
    return [W,B]


def svm_test_multiclass(W,B,x):
    c = -1
    max = 0

    counter = 0
    for w,b in zip(W,B):
        if np.dot(w,x)+b > 0:
            if c == -1:
                c == counter
                max = np.dot(w,x)+b
            elif np.dot(w,x)+b > max:
                max = np.dot(w,x)+b
                c = counter
        counter = counter + 1
    return c

def kernel_svm_train(data):
    for x in data:
        x[1] = x[0]*x[1]

    [w,b,s] =svm_train_brute(data)
    # Uncomment to see the transformed points
    #plot_hyper_binary(w, b, data)
    return [w,b,s]
