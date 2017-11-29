import numpy as np
import matplotlib.pyplot as plt
import sys
import math

def generate_training_data_binary(num):
  if num == 1:
    data = np.zeros((10,3))
    for i in range(5):
      data[i] = [i-5, 0, 1]
      data[i+5] = [i+1, 0, -1]
    
  elif num == 2:
    data = np.zeros((10,3))
    for i in range(5):
      data[i] = [0, i-5, 1]
      data[i+5] = [0, i+1, -1]

  elif num == 3:
    data = np.zeros((10,3))
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
    data = np.zeros((10,3))
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
    data = np.zeros((20,3))
    for i in range(5):
      data[i] = [i-5, 0, 1]
      data[i+5] = [i+1, 0, 2]
      data[i+10] = [0, i-5, 3]
      data[i+15] = [0, i+1, 4]
    C = 4

  elif num == 2:
    data = np.zeros((15,3))
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

def plot_training_data_binary(data,w,b):
  for item in data:
    if item[-1] == 1:
      plt.plot(item[0], item[1], 'b+')
    else:
      plt.plot(item[0], item[1], 'ro')
  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])

  x = np.linspace(-10,10,1000)
  # if w[0] == 0:
  #   plt.axhline(-b/w[1])
  # elif w[1] == 0:
  #   plt.axhline(-b/w[0])
  # else:
  #   slope = -w[0]/w[1]
  #   plt.plot(x, slope*x - b/w[1])
  #   plt.plot(x,-x/slope)
  plt.plot(x,((-1*float(w[0])))/float(w[1])*x + b)

  plt.show()

def plot_training_data_multi(data):
  colors = ['b', 'r', 'g', 'm']
  shapes = ['+', 'o', '*', '.']

  for item in data:
    plt.plot(item[0], item[1], colors[int(item[2])-1] + shapes[int(item[2])-1])
  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])
  plt.show()


# Distance from point to hyperplane
def distance_point_to_hyperplane(pt,w,b):
  pt = np.array(pt)
  w = np.array(w)
  len_w = math.sqrt(sum(w*w))
  u = w / len_w     #normalized W
  p = np.dot(np.dot(u,pt),u) # Projection of pt onto norm(w)
  # TO-D0: Make the bias negative later for testing

  return math.sqrt(sum(p*p)) + (-1*b/len_w) # Return the length of p and add the bias

#Compute minimum margin
def compute_margin(data, w, b):
  mini = float('inf')
  for pt in data:
      if 2*distance_point_to_hyperplane(pt[:-1],w,b) < mini:
        mini = 2*distance_point_to_hyperplane(pt[:-1],w,b)
  return mini

# Train SVM brute force
def svm_train_brute(data):
  pos = data[data[:,2] == 1]
  neg = data[data[:,2] == -1]
  max_margin = -1*float('inf')

  w_f = None
  b_f = None
  support_vectors = None
  for p1 in pos:
    for p2 in neg:
      mid_point = np.array([(p1[0]+p2[0])/2,(p1[1]+p2[1])/2])
      w = np.array(p1[:-1]-p2[:-1])
      # ortho_line = np.array([w[1],-1*w[0]])
      # slope = ortho_line[1]/ortho_line[0]
      slope = -1*(float(w[0])/float(w[1]))
      b = (-slope*float(mid_point[0]) + float(mid_point[1]))


      if max_margin < compute_margin(data,w,b):
        support_vectors = np.array([p1,p2])
        w_f = w
        b_f = b
        max_margin = compute_margin(data, w, b)

  w_f = w_f/math.sqrt(sum(w_f*w_f))

  return [w_f,b_f,support_vectors]


# SVM prediction of class
def svm_test_brute(w,b,x):
  if w*x + b > 0:
    return 1
  else:
    return -1

da = np.array([[4,2,-1],[6,1,1]])

# [w,b,s] = svm_train_brute(generate_training_data_binary(2))
# plot_training_data_binary(generate_training_data_binary(2),w,b)
da = generate_training_data_binary(3)
[w,b,s] = svm_train_brute(da)
plot_training_data_binary(da,w,b)
print "W: ", w
print "b: ", b
print "s: ", s
