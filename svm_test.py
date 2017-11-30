from ps8 import *

print "########### Problem 1: Report 1 #################"
for i in range(1,5):
    print("     ######### Dataset "+str(i)+" ###############")
    [w,b,S] = svm_train_brute(generate_training_data_binary(i))
    plot_training_data_binary(generate_training_data_binary(i),w,b)
    print("     Weights: " +str(w))
    print("     Bias: " + str(b))
    print("     Support Vectors:")
    for v in S:
        print("\t\t" + str(v))
    print("     Weights: " + str(w))
    print("     Margin Separator: " + str(compute_margin(generate_training_data_binary(i),w,b)))

    print("     ############################################\n")
print "#################################################"