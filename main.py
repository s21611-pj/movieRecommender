"""
==========================================
5 Movies Recommender
==========================================

Authors: Paweł Badysiak (s21166), Wojciech Turek (s21611)
How to run:
Install:
    -> pip install numpy
After instalation:
    -> in console open folder with main.py and run command:
        * "python main.py --user 'Paweł Czapiewski' --score-type Euclidean"
        * "python main.py --user 'Paweł Czapiewski' --score-type Pearson"
"""

import argparse
import json
import numpy as np


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user', dest='user', required=True,
                        help='First user')

    parser.add_argument("--score-type", dest="score_type", required=True,
                        choices=['Euclidean', 'Pearson'], help='Similarity metric to be used')
    return parser


def count_common_movies(dataset, user1, user2):
    # Movies rated by both user1 and user2
    common_movies = {}
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1
    return common_movies


# Compute the Euclidean distance score between user1 and user2
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = count_common_movies(dataset, user1, user2)

    # If there are no common movies between the users,
    # then the score is 0
    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


# Compute the Pearson correlation score between user1 and user2
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = count_common_movies(dataset, user1, user2)

    num_ratings = len(common_movies)

    # If there are no common movies between user1 and user2, then the score is 0
    if num_ratings == 0:
        return 0

    # Calculate the sum of ratings of all the common movies
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)


def group_movies_together(movies):
    """
        Method grouping movies by title with average rating
    """

    temporary_database = []
    for movie in movies:
        temporary_list = []
        if movie in temporary_database:
            continue
        for another_movie in movies:
            if movie[0] == another_movie[0]:
                temporary_list.append(another_movie)
        temporary_database = average_movie_rating(temporary_list, temporary_database)
    return temporary_database


def average_movie_rating(temporary_list, temporary_database):
    """
        Method calucalting average movie rating
    """

    average_movie = [temporary_list[0][0], 0]
    for movie in temporary_list:
        average_movie[1] += movie[1]
    average_movie[1] = average_movie[1] / len(temporary_list)
    if average_movie not in temporary_database:
        temporary_database.append(average_movie)
    return temporary_database


def prepare_database():
    """
        Method preparing sorted database from grouped movies with average rating
    """

    database = []
    for another_user in data:
        if another_user == user:
            continue
        score = None
        if score_type == 'Euclidean':
            score = euclidean_score(data, user, another_user)
        if score_type == 'Pearson':
            score = pearson_score(data, user, another_user)
        count = len(count_common_movies(data, user, another_user))

        if count == 0:
            continue

        for movie in data[another_user]:
            if movie not in data[user].keys():
                database.append([movie, score * count * data[another_user][movie]])

    filled_up_database = group_movies_together(database)
    filled_up_database.sort(key=lambda x: x[1])
    return filled_up_database


"""
    Main Driver of application
"""
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user
    score_type = args.score_type

    ratings_file = 'movie_ratings.json'

    with open(ratings_file, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    database = prepare_database()

    print("5 movies you should watch:")
    for movie in database[-5:]:
        print(movie[0])

    print("\n5 movies not fitted to you:")
    for movie in database[:5]:
        print(movie[0])
