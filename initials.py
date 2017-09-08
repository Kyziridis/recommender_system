import numpy as np
import pandas as pd

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

#That was the first naiv approach



#=============================================================
#____Make dataframes in order to proceed___
#=============================================================

#Make a dataframe
ratings_df = pd.DataFrame(ratings, columns = ['movie_id' , 'user_id' , 'rating'], 
                          dtype = int)

#Fix the matrix properly
ratings_all = ratings_df.pivot(index = 'user_id', columns ='movie_id', 
                               values = 'rating').fillna(0)
                               
#matrix with 3706x6040  only ratings
R = ratings_all.as_matrix()   

print "Matrix_Dimensions : %dx%d" %(ratings_all.shape[0] , ratings_all.shape[1])
print("\n")
print ratings_all.head(3)

                                            

#Implement the mean per movie... (column)
mean_item = np.zeros(R.shape[1])
for i in range(R.shape[1]):
    mean_item[i]=np.mean(R[:,i])
    
#Implement the mean per user....(row)
mean_user = np.zeros(R.shape[0])
for i in range(R.shape[0]):
    mean_user[i] = np.mean(R[i,:])



print ("\n")    
print 'The mean of some movies is:', mean_item[0:4]
print ("\n")    
print ("=============================================================================")
print ("\n")    
print 'The mean of some users is:', mean_user[0:4]
print ("\n")    


#__________NAIV APROACHES_______________

#Use the first naiv : mean(all ratings)

MeanAll = np.mean(ratings[:,2])
print 'The mean of all of ratings is:' , MeanAll

#Use the second naiv : all ratings for movie



