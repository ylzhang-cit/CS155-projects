import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prob2utils import *

def read_data(fileName1, fileName2):
    Y = np.loadtxt(fileName1, delimiter='\t', dtype=int)
    movie_lst = []
    names_lst = []
    with open(fileName2, 'r', encoding = "ISO-8859-1") as f:
        for line in f:
            words = line.split()
            M = len(words)
            movie = []
            name = ''
            for i in range(M):
            	if i == 0: # index of the movie
            		movie.append(int(words[i]))
            	elif i < M - 19: # words of the name of a movie
            		name += words[i] + ' '
            	else: # i in range(M - 19, M), last 19 fields
            	    movie.append(int(words[i]))
            names_lst.append(name)
            movie_lst.append(movie)
    return (Y, np.array(movie_lst), np.array(names_lst))

def get_ratings(nums):
    '''
    This function get a numpy array of rating from the numbers of different ratings.
    Input: nums is a 1D numpy array with length 5, nums[i] is the number of rating i+1.
    '''
    lst = []
    for i in range(5):
        lst += [i+1] * int(nums[i])
    return np.array(lst)


if __name__ == '__main__':
    Y, movie_arr, names_arr = read_data('./project3data/data.txt', \
                                './project3data/movies.txt')
    N = len(names_arr)
    M = np.max(Y[:, 0])
    # ratings_arr[i,j] is the number of rating j for the i-th movie (j=1,2,3,4,5)
    # ratings_arr[i,6] is the number of ratings for the i-th movie
    # ratings_arr[i,0] is average rating for the i-th movie
    ratings_arr = np.zeros((N, 7))
    for sample in Y:
    	movie_index = sample[1] - 1
    	rating = sample[2]
    	ratings_arr[movie_index][rating] += 1
    ratings_arr[:,6] = np.sum(ratings_arr[:,1:6], axis=1)
    ratings_arr[:,0] = np.dot(ratings_arr[:,1:6], np.arange(1, 6)) / ratings_arr[:,6]
    # select the 10 most populart moives and best movies
    popu10 = np.argsort(-ratings_arr[:,6])[:10]
    best10 = np.argsort(-ratings_arr[:,0])[:10]
    # For each genre, find the movies and their ratings
    genres_name = ['Unknown', 'Action', 'Adventure', 'Animation', 'Childrens',\
     'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',\
      'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    genres_ratings_lst = [[] for _ in range(19)]
    genres_category = [set() for _ in range(19)]
    movie_genres_lst = [set() for _ in range(N)]
    for (i, j, Yij) in Y:
        for k in range(19):
            if movie_arr[j-1, k+1] == 1:
                movie_genres_lst[j-1].add(genres_name[k])
                genres_ratings_lst[k].append(Yij)
                genres_category[k].add(j-1)
    # find the 10 most popular movies in each category except the unknown genre
    genre_popu10 = np.zeros((19, 10), dtype=int)
    for k in range(1, 19):
        k_indices = np.array(list(genres_category[k]))
        genre_popu10[k] = k_indices[np.argsort(-ratings_arr[k_indices,6])[:10]]

    # Basic visualization
    # plot all ratings in the MovieLens Dataset
    plt.figure()
    plt.hist(Y[:,2], range=(0.5,5.5), align='left', facecolor='green', alpha=0.75)
    plt.xlabel('rating')
    plt.ylabel('numbers of rating')
    plt.title('All ratings in the MovieLens Dataset')
    plt.grid(True)
    plt.savefig('./result/all_ratings.png')
    plt.close()
    # print and plot the 10 most popular moives
    print('\n\nThe 10 most popular movies and their numbers of ratings:')
    for n, index in enumerate(popu10):
        print('{:60}{:4.0f}'.format(names_arr[index], ratings_arr[index,6]))
        plt.figure()
        plt.hist(get_ratings(ratings_arr[index,1:6]), \
            range=(0.5, 5.5), align='left', facecolor='green', alpha=0.75)
        plt.xlabel('rating')
        plt.ylabel('numbers of rating')
        plt.title(names_arr[index])
        plt.grid(True)
        plt.savefig('./result/popular_moive%d.png' % n)
        plt.close()
    print('\n\n')
    # print and plot the 10 best moives
    print('The 10 best movies and their average rating:')
    for n, index in enumerate(best10):
        print('{:60}{:.2f}'.format(names_arr[index], ratings_arr[index,0]))
        plt.figure()
        plt.hist(get_ratings(ratings_arr[index,1:6]), \
            range=(0.5, 5.5), align='left', facecolor='green', alpha=0.75)
        plt.xlabel('rating')
        plt.ylabel('numbers of rating')
        plt.title(names_arr[index])
        plt.grid(True)
        plt.savefig('./result/best_moive%d.png' % n)
        plt.close()
    print('\n\n')
    # plot all ratings of action, adventure, western movies
    genres = [1, 2, 17] # Action, Adventure, War
    for genre in genres:
        plt.figure()
        plt.hist(genres_ratings_lst[genre], \
            range=(0.5, 5.5), align='left', facecolor='green', alpha=0.75)
        plt.xlabel('rating')
        plt.ylabel('numbers of rating')
        plt.title('All ratings of %s movies' % genres_name[genre])
        plt.grid(True)
        plt.savefig('./result/genre%d.png' % genre)
        plt.close()

    # Matrix factorization visualization
    K = 20
    eta = 0.02
    reg = 0.0
    (U, V, Ein) = train_model(M, N, K, eta, reg, Y)
    V = (V.T - np.mean(V, axis=1)).T # each row of V has zero mean
    A, S, B = np.linalg.svd(V)
    A12 = A[:, 0:2]
    VV = np.dot(A12.T, V)
    # 10 random movies
    rand10 = np.random.choice(N, 10, replace=False)
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(VV[0,rand10], VV[1,rand10], '*')
    for index in rand10:
        g = '\n(' + ', '.join(movie_genres_lst[index]) + ')'
        plt.text(VV[0,index], VV[1,index], names_arr[index] + g)
    plt.title('10 random movies')
    plt.margins(0.1)
    fig.set_size_inches(12,12)
    plt.savefig('./result/10movies_random.png')
    plt.close()
    # the 10 most popular movies
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(VV[0, popu10], VV[1,popu10], '*')
    for index in popu10:
        g = '\n(' + ', '.join(movie_genres_lst[index]) + ')'
        plt.text(VV[0,index], VV[1,index], names_arr[index] + g)
    plt.title('The 10 most popular movies')
    plt.margins(0.1)
    fig.set_size_inches(12,12)
    plt.savefig('./result/10movies_popular.png')
    plt.close()
    # the 10 best movies
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(VV[0, best10], VV[1,best10], '*')
    for index in best10:
        g = '\n(' + ', '.join(movie_genres_lst[index]) + ')'
        plt.text(VV[0,index], VV[1,index], names_arr[index] + g)
    plt.title('The 10 best movies')
    plt.margins(0.1)
    fig.set_size_inches(12,12)
    plt.savefig('./result/10movies_best.png')
    plt.close()
    # print and plot the 10 most popular movies in 3 different genres
    genres = [1, 2, 17] # Action, Adventure, War
    for genre in genres:
        print('The 10 most popular %s movies and their numbers of ratings:' % genres_name[genre])
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(VV[0, genre_popu10[genre]], VV[1,genre_popu10[genre]], '*')
        for index in genre_popu10[genre]:
            print('{:60}{:4.0f}'.format(names_arr[index], ratings_arr[index,6]))
            g = '\n(' + ', '.join(movie_genres_lst[index]) + ')'
            plt.text(VV[0,index], VV[1,index], names_arr[index] + g)
        print('\n\n')
        plt.title('The 10 most popular %s movies' % genres_name[genre])
        plt.margins(0.1)
        fig.set_size_inches(12,12)
        plt.savefig('./result/10movies_%s.png' % genres_name[genre])
        plt.close()



    