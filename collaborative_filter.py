import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error


def data_movies():
    url_movies = 'C:/Users/prisc/OneDrive/Área de Trabalho/Identifying-gray-sheep/movie_lens-100k/movies.csv'
    df_movies = pd.read_csv(url_movies, sep="|", names= [0,1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    df_movies = df_movies.iloc[:, 0:2]
    df_movies = df_movies.rename(columns={0: 'id_movie', 1: 'title_movie'})
    df_movies = df_movies.set_index('id_movie')
    return df_movies

def data_ratings():
    url_ratings = 'C:/Users/prisc/OneDrive/Área de Trabalho/Identifying-gray-sheep/movie_lens-100k/ratings.csv'
    df_ratings = pd.read_csv(url_ratings, names = ['id_user', 'id_movie', 'ratings', 'timestemp'], sep="\t")
    df_ratings = df_ratings[['id_user', 'id_movie', 'ratings']]    
    return df_ratings

def data_frame_recommender():
    df_movies =data_movies()
    df_ratings =data_ratings()
    df = pd.merge(df_movies, df_ratings, on='id_movie' )
    df_recommender = df.pivot_table(index='id_user', columns=['id_movie'], values='ratings').fillna(0) # adicionar columns=['title_movie', 'id_movie']
    df_recommender.to_csv("_users_x_movies.csv", sep="\t")
    return df_recommender

def calcular_media(index):
    url = 'C:/Users/prisc/OneDrive/Área de Trabalho/Identifying-gray-sheep/saidas_detector_greysheep/_similarity_users.csv'
    df_similarity = pd.read_csv(url, delimiter='\t')
    df_similarity.set_index('index', inplace=True)
    df_similarity = df_similarity.loc[index].to_frame()
    df_similarity.to_csv("_df_similarity.csv", sep="\t")
    return df_similarity

def prediction_rating(index, movie, n_neighbors):
    # url = 'C:/Users/prisc/OneDrive/Área de Trabalho/Identifying-gray-sheep/_users_x_movies.csv'
    # df = pd.read_csv(url, delimiter='\t')
    df = data_frame_recommender()
    df_similarity = calcular_media(index+1) # matriz de similaridade - uso para previsão
    knn_model = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=n_neighbors)
    ## cenário 2    
    # df_r = remove_exemplas_grey()
    # knn_model.fit(df_r)
    knn_model.fit(df)
    print()
    id_user = index
    id_movie = movie
    distance, index_knn = knn_model.kneighbors(df.iloc[id_user].values.reshape(1, -1), n_neighbors=n_neighbors)
    user = df.index[index] # o user recebe 1
    # Média das classificações do usuário alvo
    ratings_target_user = df.loc[user].to_frame() # todas as avaliações do usuário alvo
    user_ratings = ratings_target_user[(ratings_target_user > 0).all(axis=1)] # todas as avaliações != 0 do usuário alvo
    mean_ratings_user = user_ratings.mean()

    # Avaliações K-vizinhos-próximo != 0
    df_ratings_knns = pd.DataFrame()
    P_a_t =0
    sum_sim =0
    for i in range(1, n_neighbors):    # O loop precisa ser do tamanho de n_neighbors 
        id_user_knn = df.index[index_knn.flatten()[i]] # index_knn posição - id_user_knn id
        ratings_user_knn = df.loc[id_user_knn].to_frame() # avaliações do vizinho semelhante        
        sim = df_similarity.iloc[id_user_knn-1, 0] # similaridade entre usuário alvo e o k vizinho
        rat_user_knn_item =ratings_user_knn.loc[id_movie] # classificação do k vizinho no filme alvo
        mean_rat_knn_item = ratings_user_knn.mean() # avaliação média do k vizinho sobre todos os filmes

        # Ajusta a previsão
        P_a_t += sim.item() * (rat_user_knn_item.item() - mean_rat_knn_item.item())
        sum_sim += sim.item()
        df_ratings_knns = pd.concat([df_ratings_knns, ratings_user_knn], axis=1)   
    P_a_t /= sum_sim
    P_a_t += mean_ratings_user[index+1] # mean_ratings_user 
    df_ratings_knn = pd.merge(ratings_target_user, df_ratings_knns, on='id_movie').sort_values(by=id_user_knn, ascending=False)
    # #  print(df_ratings_knn) os filmes que foram avaliados pelo usuário alvo e o vizinho mais prximo
    df_ratings_knn = df_ratings_knn[(df_ratings_knn[id_user_knn] > 0) & (df_ratings_knn[user]==0)].reset_index()
    # print(f"A classificação prevista P_a,t é: {P_a_t:.2f}")
    if P_a_t > 5:
        P_a_t = 5
    P_a_t = float("{:.2f}".format(P_a_t))
    return P_a_t

def predict_user_movie(index):
    df = data_frame_recommender()
    user = df.index[index]
    ratings_target_user = df.loc[user].to_frame() # todas as avaliações do usuário alvo
    user_ratings = ratings_target_user[(ratings_target_user > 1).all(axis=1)]
    list_ratings = []
    list_predict = []
    i = 0
    for row in user_ratings.itertuples():
        i +=1
        list_ratings.append(row[1])
        list_predict.append(prediction_rating(index, row[0], 20))
        if i == 5:
            break
    print(f'lista de previsões: {list_predict}')
    print(f'lista de notas: {list_ratings}')
    erro_mean = mean_absolute_error(list_ratings, list_predict)
    print(f"O erro médio absoluto do usuário {user}, foi: {erro_mean:.2f}")
    return erro_mean

# # # CENÁRIO 01
# # x = grey_sheep()
# # ale = x.sample(3)
# # media = 0

# # for index in ale.itertuples():
# #     index_int = int(index[0])   
# #     media += predict_user_movie(index_int-1)
# # media = media/3
# # print(f"A média  do erro médio absoluto grupo white de usuário, foi: {media:.2f}")


# # CENÁRIO 02
# media = 0
# vet = [43, 75, 49]
# for index in vet:
#     index_int = int(index)  
#     media += predict_user_movie(index_int-1)
# media = media/3
# print(f"A média  do erro médio absoluto grupo white de usuário, foi: {media:.2f}")