#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 13:57:04 2018

@author: dead
"""


import resource
import numpy as np
import matplotlib.pyplot as plt
import os
import sys 
from sklearn import linear_model
from tqdm import tqdm
from time import time as t

# Please PRovide the path for your dataset
path = "/home/dead/Documents/Advances in Data Mining/1st project/Git_Project/recommender_system-master"

def Cross_Validation(data,nfolds,fold):
    
    # Function that generates train/test set using 5-fold Cross Validation
    np.random.seed(17)
    seqs = [x % nfolds for x in range(len(data))]
    np.random.shuffle(seqs)
    # Generate the index of train and test
    train_index = np.array([x != fold for x in seqs])
    test_index = np.array([x == fold for x in seqs])
    # Build the final train/test set using the above indexing
    train=data[train_index]
    test=data[test_index]
    
    return train, test  
    
def RMSE(x,y_hat):
    return np.sqrt(np.mean((x - y_hat) ** 2)) 
  
def MAE(x,y_hat):
    return np.mean(np.abs(x - y_hat))

def global_average():
    print("Naive Approach_1_:_Global_Average")
    print("_________________________________")
    np.random.seed(17)
    # allocate memory for results:
    RMSE_train = np.zeros(nfolds)
    RMSE_test = np.zeros(nfolds)
    MAE_train = np.zeros(nfolds)
    MAE_test = np.zeros(nfolds)   
    start = t()
    for fold in range(nfolds):
        train, test = Cross_Validation(data=ratings, nfolds=nfolds, fold=fold)
    
    # Global Average of all ratings as estimation
        gmr = np.mean(train[:, 2])

    # Measure the RMSE for train/test
        RMSE_train[fold] = RMSE(train[:,2], gmr)
        RMSE_test[fold] = RMSE(test[:,2], gmr)
        
    # Measure the MAE for train/test
        MAE_train[fold] = MAE(train[:,2],gmr)
        MAE_test[fold] = MAE(test[:,2],gmr)
        
    # Print Errors
        print("Fold " + str(fold) + ": RMSE_train=" + str(round(RMSE_train[fold] , 6)) +\
              " RMSE_test=" + str(round(RMSE_test[fold],6)) +" || "+"MAE_train=" +\
              str(round(MAE_train[fold],6)) + " MAE_test=" + str(round(MAE_test[fold],6)))
        
    # Time and Memory usage for the computation
    elapsed = t() - start # time stops
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # print the final conclusion:
    print("\n")
    print("Average RMSE on Test_set: " + str(round(np.mean(RMSE_test),5)))
    print("Average MAE on Test_set: " + str(round(np.mean(MAE_test),5)))
    print("Time: " + str(elapsed % 60) + " seconds")
    print("Memory: " + str(mem_usage) + " kilobytes")
    print("Global Average value :" + str(gmr))
    print("=============================================================")
    print("=============================================================")
    print("\n")
    
    return RMSE_test, MAE_test

def user_average():
    np.random.seed(17)
    # allocate memory for results:
    RMSE_train = np.zeros(nfolds)
    RMSE_test = np.zeros(nfolds)
    MAE_train = np.zeros(nfolds)
    MAE_test = np.zeros(nfolds)
    u_pred_final_train = []
    u_pred_final_test = []
    
    print("Naive Approach_2_:_User_Average")
    print("_________________________________")

    start = t()
    # for each fold:
    for fold in range(nfolds):
        train, test = Cross_Validation(data=ratings, nfolds=nfolds, fold=fold)            
        # CumSum for the index of the occurance of each user
        index_per_user_train = np.cumsum(np.bincount(train[:, 0]))
        index_per_user_test = np.cumsum(np.bincount(test[:, 0]))     
        
        # Initialize empty vectors for predictions
        pred_per_user_train = np.empty(len(train))
        pred_per_user_test = np.empty(len(test))
        
        # Store unique user_id
        num_users = max(np.vstack([train, test])[:, 0])
        uniq = np.unique(train[:,0])
                
        # Iterate for each user
        # 'i' iterates through [0:num_users]
        for i in range(num_users):
            user_indices_train = slice(index_per_user_train[i], index_per_user_train[i+1])
            user_indices_test = slice(index_per_user_test[i], index_per_user_test[i+1])
            
            # Check if the specific user exists in the dataset and if 
            if (i+1) in uniq:
                pred = np.mean(train[user_indices_train, 2]) 
            else:
                pred = np.mean(train[:, 2])

            # Fill in the vectors with the predictions for each user    
            pred_per_user_train[user_indices_train] = pred
            pred_per_user_test[user_indices_test] = pred
        
        # Measure the RMSE for train/test
        RMSE_train[fold] = RMSE(train[:,2], pred_per_user_train)
        RMSE_test[fold] = RMSE(test[:,2],  pred_per_user_test)
        
        # Measure the MAE for train/test
        MAE_train[fold] = MAE(train[:,2],pred_per_user_train)
        MAE_test[fold] = MAE(test[:,2],pred_per_user_test)
        
        # Store predictions
        u_pred_final_train.append(pred_per_user_train)
        u_pred_final_test.append(pred_per_user_test)
        
        # Print Errors
        print("Fold " + str(fold) + ": RMSE_train=" + str(round(RMSE_train[fold] , 6)) +\
              " RMSE_test=" + str(round(RMSE_test[fold],6)) +" || "+"MAE_train=" +\
              str(round(MAE_train[fold],6)) + " MAE_test=" + str(round(MAE_test[fold],6)))
                
    elapsed = t()- start
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
    print("\n")
    print("Average RMSE on Test_set: " + str(round(np.mean(RMSE_test),5)))
    print("Average MAE on Test_set: " + str(round(np.mean(MAE_test),5)))
    print("Time: " + str(elapsed % 60) + " seconds")
    print("Memory: " + str(mem_usage) + " kilobytes")
    print("=============================================================")
    print("=============================================================")
    print("\n")
    
    # Return predictions for each user
    return u_pred_final_train, u_pred_final_test, RMSE_test, MAE_test


def item_average():
    np.random.seed(17)
    # allocate memory for results:
    RMSE_train = np.zeros(nfolds)
    RMSE_test = np.zeros(nfolds)
    MAE_train = np.zeros(nfolds)
    MAE_test = np.zeros(nfolds)
    i_pred_final_train = []
    i_pred_final_test = []
    pred_final = []        
    print("Naive Approach_3_:_Item_Average")
    print("_________________________________")

    start = t()
    # for each fold:
    for fold in range(nfolds):
        train, test = Cross_Validation(data=ratings, nfolds=nfolds, fold=fold)
    
        # Sort training/test set by item
        train = train[train[:, 1].argsort()]
        test = test[test[:, 1].argsort()]
        
        # Store max number of items and uniq item ids
        num_items = max(np.vstack([train, test])[:, 1])
        uniq = np.unique(train[:,1])
        
        # CumSum for the index of the occurance of each user
        index_per_item_train = np.cumsum(np.bincount(train[:, 1]))
        index_per_item_test = np.cumsum(np.bincount(test[:, 1]))     
        
        # Initialize empty vectors for predictions
        pred_per_item_train = np.empty(len(train))
        pred_per_item_test = np.empty(len(test))
        
        # Create a list to store predictions for each unique movie
        prediction = np.empty(num_items)
                
        # Iterate for each movie
        for i in range(num_items):
            item_indices_train = slice(index_per_item_train[i], index_per_item_train[i+1])
            item_indices_test = slice(index_per_item_test[i], index_per_item_test[i+1])
                        
            # Check if the specific item exists in the dataset and if 
            if (i+1) in uniq:
                prediction[i] = np.mean(train[item_indices_train, 2]) 
            else:
                prediction[i] = np.mean(train[:, 2])

            # Fill in the vectors with the predictions for each user    
            pred_per_item_train[item_indices_train] = prediction[i]
            pred_per_item_test[item_indices_test] = prediction[i]
                    
        # Measure the RMSE for train/test
        RMSE_train[fold] = RMSE(train[:,2], pred_per_item_train)
        RMSE_test[fold] = RMSE(test[:,2],  pred_per_item_test)
        
        # Measure the MAE for train/test
        MAE_train[fold] = MAE(train[:,2],pred_per_item_train)
        MAE_test[fold] = MAE(test[:,2],pred_per_item_test)

        # Store predictions
        i_pred_final_train.append(pred_per_item_train)
        i_pred_final_test.append(pred_per_item_test)
        pred_final.append(prediction)
        
        # Print Errors
        print("Fold " + str(fold) + ": RMSE_train=" + str(round(RMSE_train[fold] , 6)) +\
              " RMSE_test=" + str(round(RMSE_test[fold],6)) +" || "+"MAE_train=" +\
              str(round(MAE_train[fold],6)) + " MAE_test=" + str(round(MAE_test[fold],6)))
                
    elapsed = t()- start
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
    print("\n")
    print("Average RMSE on Test_set: " + str(round(np.mean(RMSE_test),5)))
    print("Average MAE on Test_set: " + str(round(np.mean(MAE_test),5)))
    print("Time: " + str(elapsed % 60) + " seconds")
    print("Memory: " + str(mem_usage) + " kilobytes")
    print("=============================================================")
    print("=============================================================")
    print("\n")
    
    # Return predictions for each item
    return i_pred_final_train, i_pred_final_test, pred_final, RMSE_test, MAE_test


def user_item_average(u_train, u_test,I_train, I_test,prediction):

    np.random.seed(17)
    # allocate memory for results:
    RMSE_train = np.zeros(nfolds)
    RMSE_test = np.zeros(nfolds)
    MAE_train = np.zeros(nfolds)
    MAE_test = np.zeros(nfolds)
    y_final = []
    test_final = []
    coef = []

    print("Naive Approach_4_:_User+Movie_Average")
    print("Linear Model: Y = a*User + b*Item + g")
    print("_____________________________________")
    
    # Start timer
    start = t()    
    
    # For each fold
    for fold in range(nfolds):

        train, test = Cross_Validation(data=ratings, nfolds=nfolds, fold=fold)
        
        pred_item_train = np.empty(len(train))
        pred_item_test = np.empty(len(test))
        for i in range(len(train)):
            pred_item_train[i] = prediction[fold][train[i,1]-1]
        for i in range(len(test)):
            pred_item_test[i] = prediction[fold][test[i,1]-1]        

        #return pred_item_train        
        # calculate parameters a,b,c using the least squares method
        A = np.vstack((u_train[fold],pred_item_train,np.ones(len(train)))).T
        alpha, beta, gamma = np.linalg.lstsq(A, train[:, 2], rcond=1)[0]        
            
        # Calculate y_hat for train/test set
        y_hat_train = alpha*u_train[fold] + beta*pred_item_train + gamma
        y_hat_test = alpha*u_test[fold] + beta*pred_item_test + gamma   
        
        # Measure the RMSE for train/test
        RMSE_train[fold] = RMSE(train[:,2], y_hat_train)
        RMSE_test[fold] = RMSE(test[:,2],  y_hat_test)
        
        # Measure the MAE for train/test
        MAE_train[fold] = MAE(train[:,2],y_hat_train)
        MAE_test[fold] = MAE(test[:,2],y_hat_test)
        
        # Store coefficients
        coef.append(np.array([alpha,beta,gamma]))
        del(A)
        
        y_final.append(y_hat_test)
        test_final.append(test)
        
        # Print Errors
        print("Fold " + str(fold) + ": RMSE_train=" + str(round(RMSE_train[fold] , 6)) +\
              " RMSE_test=" + str(round(RMSE_test[fold],6)) +" || "+"MAE_train=" +\
              str(round(MAE_train[fold],6)) + " MAE_test=" + str(round(MAE_test[fold],6)))            

                
    elapsed = t() - start
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # print errors:
    print("\n")
    print("Average RMSE on Test_set: " + str(round(np.mean(RMSE_test),6)))
    print("Average MAE on Test_set: " + str(round(np.mean(MAE_test),6)))
    print("Time: " + str(elapsed % 60) + " seconds")
    print("Memory: " + str(mem_usage) + " kilobytes")
    print("=============================================================")
    print("=============================================================")
    print("\n")

    flag1 = input("Print Coefficients? [y/n]        >_ ")
    if flag1 == 'Y' or flag1 == 'y' or flag1 == 'yes':    
        print("\t Coefficients\n")
        print("Fold || alpha || beta || gamma\n")
        for i in range(5):
            print(str(i) + "    ||" + str(coef[i]))
    
    return RMSE_test, MAE_test
#############################################################################
#############################################################################    
def MatrixFactorization():    
    # start timer
    start = t()  

    # set the learning rate and the lambda coefficient
    learning_rate = 0.005
    lambda_reg = 0.05
    k = 10 # Num of features for Matrix Factorization
    num_iter = input("Provide number of iterations:  >_")
    num_iter = int(num_iter)  
        
    # initialize the arrays that will contain the errors for each fold
    RMSE_train = np.zeros(nfolds)
    RMSE_test = np.zeros(nfolds)
    MAE_train = np.zeros(nfolds)
    MAE_test = np.zeros(nfolds)
    RMSE_all = np.empty((nfolds,1,num_iter))
    MAE_all = np.empty((nfolds,1,num_iter))
           
    # for each fold
    for fold in range(nfolds):
        # start fold timer
        start_fold = t()
        # Cross Validation, generate train/test set
        train, test = Cross_Validation(data=ratings, nfolds=nfolds, fold=fold)                   
        # Initialize U and M matrices with random numbers
        U = np.random.rand(max(train[:,0]),k)
        M = np.random.rand(k,max(train[:,1]))
        # initialize two lists that will contain the RMSE and MAE of each iteration, respectively
        RMSE_list = []
        MAE_list = []
    
        # for each iteration:
        for iteration in range(num_iter):
            # print current fold and current iteration
            print("Fold: " + str(fold + 1) + "," + " Iteration: " + str(iteration + 1))
            x_hat = np.empty(len(train))

            # for each record in the train set 
            for idx,rating in enumerate(train):
                # create a copy of the user vector # 
                u = U[rating[0] - 1,:].copy()
                # calculate the rating (prediction) the user would give to the movie 
                x_hat[idx] = np.dot(u,M[:,rating[1] - 1])
                # supress the rating between 1 and 5
                if x_hat[idx] < 1:
                    x_hat[idx] = 1
                elif x_hat[idx] > 5:
                    x_hat[idx] = 5
                # calculate the error
                e_ij = rating[2] - x_hat[idx]
             
                # update matrices U and M
                U[rating[0] - 1,:] += learning_rate*(2*e_ij*M[:,rating[1] - 1] - lambda_reg*u) 
                M[:,rating[1] - 1] += learning_rate*(2*e_ij*u - lambda_reg*M[:,rating[1] - 1])
            # calculate the RMSE and MAE of this iteration, respectively    
            rmse_iter = RMSE(train[:,2],x_hat )
            mae_iter = MAE(train[:,2],x_hat)
            
            print("RMSE: " + str(rmse_iter))
            print("MAE : " + str(mae_iter))
            print("--------------")
            # Add/append the errors to the lists containing the errors of previous iterations
            RMSE_list.append(rmse_iter)
            MAE_list.append(mae_iter)
        # RMSE_all and MAE_all contain the list of errors of all iterations for the current fold.
        RMSE_all[fold] = RMSE_list
        MAE_all[fold] = MAE_list
        # the RMSE and MAE of the current fold are the last calculated RMSE and MAE, respectively    
        RMSE_train[fold] = RMSE_list[-1]
        MAE_train[fold] = MAE_list[-1]           
        # calculate the number of users
        num_users = max(train[:,0])
        # calculate the number of times a user appears in the test set
        num_ratings_perUser_test = np.bincount(test[:,0])
        # the cumulative sum indicates the index in which every new user (user_id) 
        # appears in the test set.
        index_perUser_test = np.cumsum(num_ratings_perUser_test)     
        
        # evaluate the model on the test set (make predictions on the test set)
        pred = np.empty(len(test[:,2]),object)
        for user_id in range(num_users):
            test_subset = test[index_perUser_test[user_id]:index_perUser_test[user_id + 1],:]
            pred[index_perUser_test[user_id]:index_perUser_test[user_id + 1]] = np.dot(U[user_id,:],M[:,test_subset[:,1] - 1])

        # calculate the RMSE and MAE of the test set  
        RMSE_test[fold] = RMSE(test[:,2],pred)   
        MAE_test[fold] = MAE(test[:,2], pred)        
        # stop fold timer
        end_fold = t()
        # print how much time taken (in minutes) for the fold to be executed. 
        # Also print the RMSE and MAE of the train and test fold 
        print("Time taken for fold " + str(fold + 1) + "(in minutes):" + str((end_fold - start_fold)/60))
        print("Fold " + str(fold + 1) + ": Root Mean Squared Error (RMSE) on train set: "+\
              str(RMSE_train[fold]) + "; Root Mean Squared Error (RMSE) on test set: " +\
              str(RMSE_test[fold]) )
        
        print("Fold " + str(fold + 1) + ": Mean Absolute Error (MAE) on train set: " +\
              str(MAE_train[fold]) + "; Mean Absolute Error (MAE) on test set: " +\
              str(MAE_test[fold]))
        print("")
    
    # print the average RMSE of the 5 folds, for the train and test sets. 
    print("Mean of Root Mean Squared Error (RMSE) on train sets: " + str(np.mean(RMSE_train)))
    print("Mean of Root Mean Squared Error (RMSE) on test sets: " + str(np.mean(RMSE_test)))
    
    # print the average MAE of the 5 folds, for the train and test sets. 
    print("Mean of Mean Absolute Error (MAE) on train sets: " + str(np.mean(MAE_train)))
    print("Mean of Mean Absolute Error (MAE) on test sets: " + str(np.mean(MAE_test)))
    
    # end timer
    end = t()
    # print how much time (in hours) took for the function to be evaluated
    print("Time taken to evaluate function (in hours): " + str(((end - start)/60)/60))
    
    # return matrix U and M, RMSE and MAE for each iteration of the 5 folds, and finally
    # the RMSE and MAE of the 5 test folds
    return(U,M,RMSE_all,MAE_all,RMSE_test,MAE_test)
    
def Load(path=path):
    #load data
    #ratings=read_data("ratings.dat")
    print("--------------------")
    print("Loading the dataset")
    print("--------------------\n")    
    os.chdir(path)
    # Load the dataset
    ratings=[]
    f = open("datasets/ratings.dat", 'r')
    for line in f:
        data = line.split('::')
        ratings.append([int(z) for z in data[:3]])
    f.close()
    ratings=np.array(ratings)
    print("+...Done...+\n")    
    return ratings
    

if __name__ == "__main__":
        
    # Set the parameters for the 5-fold Cross Validation
    nfolds = 5
    np.random.seed(17)
    # Load the dataset
    ratings = Load()
    # Run Recomendation algorithms                 
    rmse_g, mae_g=global_average()
    train_u,test_u,rmse_u, mae_u = user_average()
    train_I, test_I, pred, rmse_i, mae_i = item_average()
    rmse, mae = user_item_average(train_u,test_u, train_I, test_I, prediction=pred)  
    U,M,rmse_all,mae_all,rmse_test,mae_test=MatrixFactorization()
    
    # Plotting RMSE & MAE for MatrixFactorization
    plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.plot(rmse_all[0][0], label="RMSE")
    plt.plot(mae_all[0][0], label="MAE")
    plt.legend(loc=4)
    plt.grid(True)
    plt.title("Error Vs iterations")
    plt.show()

    # Plotting RMSE for Naive
    plt.figure()
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.plot(rmse_g, label="GlobalAverage")
    plt.plot(rmse_u, label="UserAverage")
    plt.plot(rmse_i, label="MovieAverage")
    plt.plot(rmse, label="LeastSquares")
    plt.legend(loc=4)
    plt.grid(True)
    plt.title("RMSE vs Folds")
    plt.show()

    # Plotting MAE for Naive
    plt.figure()
    plt.xlabel("Fold")
    plt.ylabel("MAE")
    plt.plot(mae_g, label="GlobalAverage")
    plt.plot(mae_u, label="UserAverage")
    plt.plot(mae_i, label="MovieAverage")
    plt.plot(mae, label="LeastSquares")
    plt.legend(loc=4)
    plt.grid(True)
    plt.title("MAE vs Folds")
    plt.show()
