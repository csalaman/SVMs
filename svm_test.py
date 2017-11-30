from ps8 import *

print "########### Problem 1: Report 1 #################"
for i in range(1,5):
    print("     ######### Dataset "+str(i)+" ###############")
    [w,b,S] = svm_train_brute(generate_training_data_binary(i))
    plot_hyper_binary(w,b,generate_training_data_binary(i))
    print("     Weights: " +str(w))
    print("     Bias: " + str(b))
    print("     Support Vectors:")
    for v in S:
        print("\t\t" + str(v))
    print("     Weights: " + str(w))
    print("     Margin Separator: " + str(compute_margin(generate_training_data_binary(i),w,b)))

    print("     ############################################\n")
print "#################################################"


(data,C) = generate_training_data_multi(1)
[W,B] = svm_train_multiclass(generate_training_data_multi(1))
print W,B
plot_hyper_multi(W,B,(data,C))

[w1,b1,s1] = svm_train_brute(generate_training_data_binary(5))
[w2,b2,s2] = kernel_svm_train(generate_training_data_binary(5))

plot_hyper_binary(w1,b1,generate_training_data_binary(5))
# plot_hyper_binary(w2,b2,generate_training_data_binary(5))