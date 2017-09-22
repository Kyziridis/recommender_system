import resource
import scipy.sparse as ss
from helpers.ComputeError import *
import numpy as np
import helpers.Time as t
import helpers.load_data as data
import helpers.SplitMatrix as splitmatrix
from sklearn import linear_model
import pandas as pd

print("\n")
print("___This is a Recommender System__")
print("##################################")
print("########################")
print("##########")
print("###")
print("It is developed by Sevak and George")
print("###")
print("##########")
print("########################")
print("###################################")
def naive_global():

    # Load the data 
    ratings = data.load_ratings()

    #split data into 5 train and test folds
    nfolds=5

#allocate memory for results:
    err_train=np.zeros(nfolds)
    err_test=np.zeros(nfolds)

#to make sure you are able to repeat results, set the random seed to something:
    np.random.seed(17)

    seqs=[x%nfolds for x in range(len(ratings))]
    np.random.shuffle(seqs)

    print("Naiv Approach_1_:_Global_Average")
    print("_________________________________")
    print("\n")
    #for each fold:
    for fold in range(nfolds):
        train_sel=np.array([x!=fold for x in seqs])
        test_sel=np.array([x==fold for x in seqs])
        train=ratings[train_sel]
        test=ratings[test_sel]

        #First naiv approach... global    
        #calculate model parameters: mean rating over the training set:
        gmr=np.mean(train[:,2])

        #apply the model to the train set:
        err_train[fold]=np.sqrt(np.mean((train[:,2]-gmr)**2))

        #apply the model to the test set:
        err_test[fold]=np.sqrt(np.mean((test[:,2]-gmr)**2))
    
    
    
        #print errors:
        print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))

    #print the final conclusion:
    print("\n")
    print("Mean error on TRAIN: " + str(np.mean(err_train)))
    print("Mean error on  TEST: " + str(np.mean(err_test)))      
    print("\n")           
    print("Global Average :" + str(gmr))
    print("=============================================================")
    print("=============================================================")
    print("\n")       
#######################################################################
#######################################################################

def naive_user():
    
    # Load the data 
    ratings = data.load_ratings()

    ratings_df = pd.DataFrame(ratings, columns=['user_id', 'movie_id', 'rating'],
                              dtype=int)

    #implement the means for each user                          
    mean_user_all = np.mean(ratings_df.groupby(['user_id'])['rating'].mean())

    nfolds=5

    #allocate memory for results:
    err_train=np.zeros(nfolds)
    err_test=np.zeros(nfolds)

    #to make sure you are able to repeat results, set the random seed to something:
    np.random.seed(17)

    seqs=[x%nfolds for x in range(len(ratings))]
    np.random.shuffle(seqs)

    print("Naiv Approach_2_:_User_Average")
    print("_________________________________")
    print("\n")

    #for each fold:
    for fold in range(nfolds):
        train_sel=np.array([x!=fold for x in seqs])
        test_sel=np.array([x==fold for x in seqs])
        
        #make DataFrames for train and test 
        train_df = pd.DataFrame(ratings_df.iloc[train_sel] , 
                            columns=['user_id' , 'movie_id' , 'rating'] ,
                            dtype= int) #.iloc : indexing with np.array in pd.DataFrame)
    
        test_df =pd.DataFrame(ratings_df.iloc[test_sel] , 
                            columns=['user_id' , 'movie_id' , 'rating'] ,
                            dtype= int) 
                            
    
        #Count the occur frequency of each User in the train & test.    
        times_u_train = np.bincount(train_df['user_id'])
        times_u_test = np.bincount(test_df['user_id'])
    
        #Vector of means Implementation for each User
        mean_u_train = np.array(train_df.groupby(['user_id'])['rating'].mean())
    
    
        #After the vector of means Implementation we make equal vectors.
        m_utrain_rep = np.repeat(mean_u_train , times_u_train[1:len(times_u_train)])
        m_utest_rep = np.repeat(mean_u_train , times_u_test[1:len(times_u_test)])
    
        #apply the model to the train set:f you want to see the results for the first Naiv Approach press 1")
        err_train[fold] = np.sqrt(np.mean((train_df.iloc[:,2]-m_utrain_rep)**2))

        #apply the model to the test set:
        err_test[fold] = np.sqrt(np.mean((test_df.iloc[:,2]-m_utest_rep)**2))
    
        #print errors for each fold:
        print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))
        
    #print the final conclusion:

    print("\n")
    print("Mean error on TRAIN: " + str(np.mean(err_train)))
    print("Mean error on  TEST: " + str(np.mean(err_test)))    
    print("\n")
    print("Mean of all user ratings is : " + str( mean_user_all) )
    print("=============================================================")
    print("=============================================================")
    print("\n")
##########################################################################
##########################################################################
    
def naive_item():
     # Load the data 
    ratings = data.load_ratings()

    ratings_df = pd.DataFrame(ratings, columns=['user_id', 'movie_id', 'rating'],
                              dtype=int)   
    ratings_df = ratings_df.sort(['movie_id'])

     # implement the means for each user
    mean_movie_all = np.mean(ratings_df.groupby(['movie_id'])['rating'].mean())

    nfolds=5

    #allocate memory for results:
    err_train=np.zeros(nfolds)
    err_test=np.zeros(nfolds)

    seqs=[x%nfolds for x in range(len(ratings))]
    np.random.shuffle(seqs)

    print("Naiv Approach_3_:_Movie_Average")
    print("_________________________________")
    print("\n")

    for fold in range(nfolds):
        train_sel=np.array([x!=fold for x in seqs])
        test_sel=np.array([x==fold for x in seqs])

        #make DataFrames for train and test
        train_df = pd.DataFrame(ratings_df.iloc[train_sel] ,
                                columns=['user_id' , 'movie_id' , 'rating'] ,
                                dtype= int) #.iloc : indexing with np.array in pd.DataFrame)

        test_df =pd.DataFrame(ratings_df.iloc[test_sel] ,
                                columns=['user_id' , 'movie_id' , 'rating'] ,
                                dtype= int)


        #Count the occur frequency of each User in the train & test.
        times_u_train = np.bincount(train_df['user_id'])
        times_u_test = np.bincount(test_df['user_id'])

        #Vector of means Implementation for each User
        mean_u_train = np.array(train_df.groupby(['user_id'])['rating'].mean())


        #After the vector of means Implementation we make equal vectors.
        m_utrain_rep = np.repeat(mean_u_train , times_u_train[1:len(times_u_train)])
        m_utest_rep = np.repeat(mean_u_train , times_u_test[1:len(times_u_test)])

        #apply the model to the train set:
        err_train[fold] = np.sqrt(np.mean((train_df.iloc[:,2]-m_utrain_rep)**2))

        #apply the model to the test set:
        err_test[fold] = np.sqrt(np.mean((test_df.iloc[:,2]-m_utest_rep)**2))

        #print errors for each fold:
        print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))

    #print the final conclusion:
    print("\n")
    print("Mean error on TRAIN: " + str(np.mean(err_train)))
    print("Mean error on  TEST: " + str(np.mean(err_test)))
    print("\n")
    print("Mean of all movies ratings is : " + str(mean_movie_all))
    print("=============================================================")
    print("\n")
############################################################################
#############################################################################

def round_ratings(ratings):
    maxRating = 5
    minRating = 1
    return (np.array([max(min(x, maxRating), minRating) for x in ratings]))
    
def naive_linear_user_item():
    lm = linear_model.LinearRegression()
    # Predict Y using the linear model with estimated coefficients
    # lm.predict()
    # Grab the linear model function 
    ratings = data.load_ratings()

    avg_ratings_list_train = np.zeros((len(train_set), 2))
    avg_ratings_list_test = np.zeros((len(test_set), 2))

    regression_predictions_train = round_ratings(regr_coeffs[0] * avg_ratings_list_train[:, 0] + regr_coeffs[1] * avg_ratings_list_train[:, 1] + regr_intercept)
    regression_predictions_test = round_ratings(regr_coeffs[0] * avg_ratings_list_test[:, 0] + regr_coeffs[1] * avg_ratings_list_test[:, 1] + regr_intercept)

    regr_error_train = np.sqrt(np.mean((train_set[:, 2] - regression_predictions_train) ** 2))
    regr_error_test = np.sqrt(np.mean((test_set[:, 2] - regression_predictions_test) ** 2))
    lm.fit(data, ratings)

    print("Linear Regression done. Coefficients:", regr_coeffs, regr_intercept)
    print("Training error:", regr_error_train)
    print("Test error:", regr_error_test)


def mf_gradient_descent():
    """
    Matrix factorization with gradient descent
    :param data:
    :param users:
    :param movies:
    :return:
    """
    num_factors = 10
    steps = 2
    learn_rate = 0.005
    regularization = 0.05 # lambda

    nfolds = 5

    ratings = data.load_ratings()

    users = np.max(ratings[:, 0])
    movies = np.max(ratings[:, 1])

    for fold in range(nfolds):

        train_set = np.array([ratings[x] for x in np.arange(len(ratings)) if (x % nfolds) != fold])
        test_set = np.array([ratings[x] for x in np.arange(len(ratings)) if (x % nfolds) == fold])

        # Convert the data set to the IxJ matrix  
        X_data = splitmatrix.Xmatrix(train_set, users, movies)

        X_hat = np.zeros(users, movies) #The matrix of predicted train_set

        E = np.zeros(users, movies) #The error values
        
        # initialize to random matrices
        U = np.random.rand(users, num_factors)
        M = np.random.rand(num_factors, movies)
        
        elapsed = 0

        for step in np.arange(steps):
            start = t.start()
            
            for i in np.arange(len(train_set)):
                
                user_id = train_set[i,0] - 1
                item_id = train_set[i,1] - 1
                actual = train_set[i,2]      

                error = actual - np.sum(U[user_id,:] * M[:,item_id])
                
                # Update U and M by building U_prime and M_prime, which will replace U and M when done with this iteration
                for k in np.arange(num_factors):
                    U[user_id, k] +=  learn_rate * (2 * error * M[k, item_id] - regularization * U[user_id, k])
                    M[k, item_id] += learn_rate * (2 * error * U[user_id, k] - regularization * M[k, item_id])

            elapsed += t.start() - start
            
    
            # Compute intermediate MSE
            X_hat = np.dot(U,M)
            E = X_data - X_hat
            intermediate_error = np.sqrt(np.mean(E[np.where(np.isnan(E) == False)]**2))
            
            print("Iteration", step, "out of", steps, "done. Error:", intermediate_error)
    
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            # Apply U and M one last time and return the result
            X_hat = np.dot(U,M)

    X_train = splitmatrix.Xmatrix(train_set, users, movies)
    X_test = splitmatrix.Xmatrix(test_set, users, movies)
        
    E_train = X_train - X_hat
    E_test = X_test - X_hat
        
    MF_error_train = np.sqrt(np.mean(E_train[np.where(np.isnan(E_train) == False)]**2))
    MF_error_test = np.sqrt(np.mean(E_test[np.where(np.isnan(E_test) == False)]**2))
        
    print('MF training set error:', MF_error_train)
    print('MF test set error:', MF_error_test)



#TODO Matrix factorization with Alternating Least Squares - bonus
def mf_als(data, users, movies, **kwargs):

    return None



if __name__ == "__main__":

     naive_global()
     naive_user()
     naive_item()
#    mf_gradient_descent()




    