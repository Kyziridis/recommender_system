import numpy as np
import pandas as pd

###############################################################################

#==============================================================================
# User data: <UserID::Gender::Age::Occupation::Zip-code>
# Movie data: <MovieID::Title (Year)::Genres>
#==============================================================================

print("\n")

ratings = np.genfromtxt("/home/dead/Documents/Advances in Data Mining/1st project/ml-1m/ratings.dat", 
                        usecols=(0, 1, 2), delimiter='::', dtype='int')


#==============================================================================
# users = np.genfromtxt("/home/dead/Documents/Advances in Data Mining/1st project/ml-1m/users.dat",
#                       usecols=(0,1,2,3,4) , delimiter='::' , dtype='string')
#                       
# movies = np.genfromtxt("/home/dead/Documents/Advances in Data Mining/1st project/ml-1m/users.dat",
#                       usecols=(0,1,2) , delimiter='::' , dtype='string')                      
#==============================================================================



#==============================================================================
#__________Basic Script for 5-fold Cross Validation________
#==============================================================================                        
#This script is for generall global average rating

#split data into 5 train and test folds
nfolds=5

#allocate memory for results:
err_train=np.zeros(nfolds)
err_test=np.zeros(nfolds)

#to make sure you are able to repeat results, set the random seed to something:
np.random.seed(17)

seqs=[x%nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)

#for each fold:
for fold in range(nfolds):
    train_sel=np.array([x!=fold for x in seqs])
    test_sel=np.array([x==fold for x in seqs])
    train=ratings[train_sel]
    test=ratings[test_sel]

###############################################    
#So we have already the train and the test set !!!    
###############################################

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
######################################################################
######################################################################


#==========================================
# #___________NAIV APROACHES_______________
#==========================================

#Implement the first Naiv approach : all ratings for user.
#---------------------------------------------------------
# In this naiv implementation we will use DataFrames in order to
# avoid nested for_loops inside cross validation


#########################################################################
#Using Pandas we make :  | [user_id] | [movie_id] | [rating] |
ratings_df = pd.DataFrame(ratings, columns = ['user_id' , 'movie_id' , 'rating'], 
                          dtype = int)
#########################################################################

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
                            
    # make two vectors st1 and st2 with the unique user_id's
    # in order to check if train_df and test_df include 
    # all the different id's                           
#==============================================================================
#     st1 = list(set(train_df.iloc[:,0]))
#     st2 = list(set(test_df.iloc[:,0]))
#==============================================================================
    
    #Count the occur frequency of each User in the train & test.    
    times_u_train = np.bincount(train_df['user_id'])
    times_u_test = np.bincount(test_df['user_id'])
    
    #Vector of means Implementation for each User
    mean_u_train = np.array(train_df.groupby(['user_id'])['rating'].mean())
        
    
    #After the vector of means Implementation we make equal vectors.
    m_utrain_rep = np.repeat(mean_u_train , times_u_train[1:len(times_u_train)])
    m_utest_rep = np.repeat(mean_u_train , times_u_test[1:len(times_u_test)])
    
#apply the model to the train set:
    err_train[fold]=np.sqrt(np.mean((train_df.iloc[:,2]-m_utrain_rep)**2))

#apply the model to the test set:
    err_test[fold]=np.sqrt(np.mean((test_df.iloc[:,2]-m_utest_rep)**2))
    
#print errors:
    print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))

#print the final conclusion:
print("\n")
print("Mean error on TRAIN: " + str(np.mean(err_train)))
print("Mean error on  TEST: " + str(np.mean(err_test)))    
print("\n")
print("Mean of all user ratings is : " + str( mean_user_all) )
#########################################################
###########################################################

#Implement the third Naiv approach : all ratings for movie
#---------------------------------------------------------
nfolds=5

#allocate memory for results:
err_train=np.zeros(nfolds)
err_test=np.zeros(nfolds)

#to make sure you are able to repeat results, set the random seed to something:
np.random.seed(17)

seqs=[x%nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)
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
                            
    # make two vectors st1 and st2 with the unique user_id's
    # in order to check if train_df and test_df include 
    # all the different id's                           
#==============================================================================
#     st1 = list(set(train_df.iloc[:,0]))
#     st2 = list(set(test_df.iloc[:,0]))
#==============================================================================
    
    #Count the occur frequency of each Movie in the train & test.    
    times_mv_train = np.bincount(train_df['movie_id'])
    times_mv_test = np.bincount(test_df['movie_id'])
    
    #Vector of means Implementation for each Movie
    mean_mv_train = np.array(train_df.groupby(['movie_id'])['rating'].mean())
        
    
    #After the vector of means Implementation we make equal vectors.
    m_mvtrain_rep = np.repeat(mean_mv_train , times_mv_train[times_mv_train])
    m_mvtest_rep = np.repeat(mean_mv_train , times_mv_test)
    
#apply the model to the train set:
    err_train[fold]=np.sqrt(np.mean((train_df.iloc[:,2]-m_mvtrain_rep)**2))

#apply the model to the test set:
    err_test[fold]=np.sqrt(np.mean((test_df.iloc[:,2]-m_mvtest_rep)**2))
    
#print errors:
    print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))

#print the final conclusion:
print("\n")
print("Mean error on TRAIN: " + str(np.mean(err_train)))
print("Mean error on  TEST: " + str(np.mean(err_test)))   










 

















