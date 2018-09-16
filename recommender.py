# import resource
import resource
import numpy as np
import helpers.Time as t
import helpers.load_data as data
import helpers.SplitMatrix as splitmatrix
from sklearn import linear_model
import pandas as pd
import helpers.NumbersRounder as rounder

ratings = data.load_ratings()
nfolds = 5
np.random.seed(17)
seqs = [x % nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)


def global_average():
    # allocate memory for results:
    err_train = np.zeros(nfolds)
    err_test = np.zeros(nfolds)

    print("Naiv Approach_1_:_Global_Average")
    print("_________________________________")
    print("\n")
    start = t.start()

    # for each fold:
    for fold in range(nfolds):
        train_set = np.array([x != fold for x in seqs])
        test_set = np.array([x == fold for x in seqs])

        train = ratings[train_set]
        test = ratings[test_set]

        # First naiv approach... global
        # calculate model parameters: mean rating over the training set:
        gmr = np.mean(train[:, 2])

        # apply the model to the train set:
        err_train[fold] = np.sqrt(np.mean((train[:, 2] - gmr) ** 2))

        # apply the model to the test set:
        err_test[fold] = np.sqrt(np.mean((test[:, 2] - gmr) ** 2))

        # print errors:
        print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))

        elapsed = t.start() - start

        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # print the final conclusion:
    print("\n")
    print("Mean error on TRAIN: " + str(np.mean(err_train)))
    print("Mean error on  TEST: " + str(np.mean(err_test)))
    print("Time: " + str(elapsed % 60) + " seconds")
    print("Memory: " + str(mem_usage) + " kilobytes")
    print("\n")
    print("Global Average :" + str(gmr))
    print("=============================================================")
    print("=============================================================")
    print("\n")


def user_average():
    ratings_df = pd.DataFrame(ratings, columns=['user_id', 'movie_id', 'rating'], dtype=int)

    # implement the means for each user
    mean_user_all = np.mean(ratings_df.groupby(['user_id'])['rating'].mean())

    # allocate memory for results:
    err_train = np.zeros(nfolds)
    err_test = np.zeros(nfolds)

    print("Naiv Approach_2_:_User_Average")
    print("_________________________________")
    print("\n")

    start = t.start()
    # for each fold:
    for fold in range(nfolds):
        train_sel = np.array([x != fold for x in seqs])
        test_sel = np.array([x == fold for x in seqs])

        # make DataFrames for train and test
        train_df = pd.DataFrame(ratings_df.iloc[train_sel],
                                columns=['user_id', 'movie_id', 'rating'],
                                dtype=int)  # .iloc : indexing with np.array in pd.DataFrame)

        test_df = pd.DataFrame(ratings_df.iloc[test_sel],
                               columns=['user_id', 'movie_id', 'rating'],
                               dtype=int)

        # Count the occur frequency of each User in the train & test.
        times_u_train = np.bincount(train_df['user_id'])
        times_u_test = np.bincount(test_df['user_id'])

        # Vector of means Implementation for each User
        mean_u_train = np.array(train_df.groupby(['user_id'])['rating'].mean())

        # After the vector of means Implementation we make equal vectors.
        m_utrain_rep = np.repeat(mean_u_train, times_u_train[1:len(times_u_train)])
        m_utest_rep = np.repeat(mean_u_train, times_u_test[1:len(times_u_test)])

        # apply the model to the train set:f you want to see the results for the first Naiv Approach press 1")
        err_train[fold] = np.sqrt(np.mean((train_df.iloc[:, 2] - m_utrain_rep) ** 2))

        # apply the model to the test set:
        err_test[fold] = np.sqrt(np.mean((test_df.iloc[:, 2] - m_utest_rep) ** 2))

        # print errors for each fold:
        print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))
        elapsed = t.start() - start
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # print the final conclusion:

    print("\n")
    print("Mean error on TRAIN: " + str(np.mean(err_train)))
    print("Mean error on  TEST: " + str(np.mean(err_test)))
    print("Time: " + str(elapsed % 60) + " seconds")
    print("Memory: " + str(mem_usage) + " kilobytes")
    print("\n")
    print("Mean of all user ratings is : " + str(mean_user_all))
    print("=============================================================")
    print("=============================================================")
    print("\n")


def item_average():
    ratings_df = pd.DataFrame(ratings, columns=['user_id', 'movie_id', 'rating'], dtype=int)
    ratings_df = ratings_df.sort(['movie_id'])

    # implement the means for each user
    mean_movie_all = np.mean(ratings_df.groupby(['movie_id'])['rating'].mean())

    # allocate memory for results:
    err_train = np.zeros(nfolds)
    err_test = np.zeros(nfolds)

    seqs = [x % nfolds for x in range(len(ratings))]
    np.random.shuffle(seqs)

    print("Naiv Approach_3_:_Movie_Average")
    print("_________________________________")
    print("\n")

    start = t.start()

    for fold in range(nfolds):
        train_sel = np.array([x != fold for x in seqs])
        test_sel = np.array([x == fold for x in seqs])

        # make DataFrames for train and test
        train_df = pd.DataFrame(ratings_df.iloc[train_sel],
                                columns=['user_id', 'movie_id', 'rating'],
                                dtype=int)  # .iloc : indexing with np.array in pd.DataFrame)

        test_df = pd.DataFrame(ratings_df.iloc[test_sel],
                               columns=['user_id', 'movie_id', 'rating'],
                               dtype=int)

        # Count the occur frequency of each User in the train & test.
        times_u_train = np.bincount(train_df['user_id'])
        times_u_test = np.bincount(test_df['user_id'])

        # Vector of means Implementation for each User
        mean_u_train = np.array(train_df.groupby(['user_id'])['rating'].mean())

        # After the vector of means Implementation we make equal vectors.
        m_utrain_rep = np.repeat(mean_u_train, times_u_train[1:len(times_u_train)])
        m_utest_rep = np.repeat(mean_u_train, times_u_test[1:len(times_u_test)])

        # apply the model to the train set:
        err_train[fold] = np.sqrt(np.mean((train_df.iloc[:, 2] - m_utrain_rep) ** 2))

        # apply the model to the test set:
        err_test[fold] = np.sqrt(np.mean((test_df.iloc[:, 2] - m_utest_rep) ** 2))

        # print errors for each fold:
        print("Fold " + str(fold) + ": RMSE_train=" + str(err_train[fold]) + "; RMSE_test=" + str(err_test[fold]))

        elapsed = t.start() - start
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # print the final conclusion:
    print("\n")
    print("Mean error on TRAIN: " + str(np.mean(err_train)))
    print("Mean error on  TEST: " + str(np.mean(err_test)))
    print("Time: " + str(elapsed % 60) + " seconds")
    print("Memory: " + str(mem_usage) + " kilobytes")
    print("\n")
    print("Mean of all movies ratings is : " + str(mean_movie_all))
    print("=============================================================")
    print("\n")


def user_item_average():

    # allocate memory for results:
    err_train = np.zeros(nfolds)
    err_test = np.zeros(nfolds)
    start = t.start()
    # for each fold:
    for fold in range(nfolds):
        np.random.shuffle(ratings)

        train_set = np.array([x != fold for x in seqs])
        test_set = np.array([x == fold for x in seqs])

        train = ratings[train_set]
        test = ratings[test_set]

        train_avg_rating = np.zeros((len(train), 2))
        test_avg_rating = np.zeros((len(test), 2))

        regr = linear_model.LinearRegression()

        regr.fit(train_avg_rating, train[:, 2])

        train_reg_pre = rounder.rounder(regr.coef_[0] * train_avg_rating[:, 0] + regr.coef_[1] * train_avg_rating[:, 1] + regr.intercept_)
        test_reg_pre = rounder.rounder(regr.coef_[0] * test_avg_rating[:, 0] + regr.coef_[1] * test_avg_rating[:, 1] + regr.intercept_)

        regr_error_train = np.sqrt(np.mean((train[:, 2] - train_reg_pre) ** 2))
        regr_error_test = np.sqrt(np.mean((test[:, 2] - test_reg_pre) ** 2))

        print("Coefficients:", regr.coef_, regr.intercept_)
        elapsed = t.start() - start
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # print errors:
    print("Error on Training set:", regr_error_train)
    print("Error on Test set:", regr_error_test)
    print("Time: " + str(elapsed % 60) + " seconds")
    print("Memory: " + str(mem_usage) + " kilobytes")


def mf_gradient_descent():
    """
    Matrix factorization with gradient descent
    :param data:
    :param users:
    :param movies:
    :return:
    """
    num_factors = 10
    steps = 75
    learn_rate = 0.005
    regularization = 0.05  # lambda

    users = np.max(ratings[:, 0])
    movies = np.max(ratings[:, 1])

    start = t.start()

    for fold in range(nfolds):
        print("fold", fold)

        train_set = np.array([ratings[x] for x in np.arange(len(ratings)) if (x % nfolds) != fold])
        test_set = np.array([ratings[x] for x in np.arange(len(ratings)) if (x % nfolds) == fold])

        # Convert the data set to the IxJ matrix  
        x_data = splitmatrix.split_matrix(train_set, users, movies)

        x_hat = np.zeros(users, movies)  # The matrix of predicted train_set

        E = np.zeros(users, movies)  # The error values

        # initialize to random matrices
        U = np.random.rand(users, num_factors)
        M = np.random.rand(num_factors, movies)

        elapsed = 0

        for step in np.arange(steps):
            start = t.start()

            for idx in np.arange(len(train_set)):

                user_id = train_set[idx, 0] - 1
                item_id = train_set[idx, 1] - 1
                actual = train_set[idx, 2]

                error = actual - np.sum(U[user_id, :] * M[:, item_id])

                # Update U and M
                for k in np.arange(num_factors):
                    U[user_id, k] += learn_rate * (2 * error * M[k, item_id] - regularization * U[user_id, k])
                    M[k, item_id] += learn_rate * (2 * error * U[user_id, k] - regularization * M[k, item_id])

            elapsed += t.start() - start

            x_hat = np.dot(U, M)
            E = x_data - x_hat
            intermediate_error = np.sqrt(np.mean(E[np.where(np.isnan(E) == False)] ** 2))

            print("Iteration", step, "out of", steps, "done. Error:", intermediate_error)

            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            # Apply U and M one last time and return the result

    x_hat = np.dot(U, M)

    x_train = splitmatrix.split_matrix(train_set, users, movies)
    x_test = splitmatrix.split_matrix(test_set, users, movies)

    e_train = x_train - x_hat
    e = x_test - x_hat

    MF_error_train = np.sqrt(np.mean(e_train[np.where(np.isnan(e_train) == False)] ** 2))
    MF_error_test = np.sqrt(np.mean(e[np.where(np.isnan(e) == False)] ** 2))

    print('Error on MF-GD training set :', MF_error_train)
    print('Error on MF-GD test set:', MF_error_test)
    print("Time: " + str(elapsed % 60) + " seconds")
    print("Memory: " + str(mem_usage) + " kilobytes")


if __name__ == "__main__":
    global_average()
    user_average()
    item_average()
    user_item_average()
    mf_gradient_descent()
