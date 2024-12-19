from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from numpy import where
import numpy as np
import sys

sys.setrecursionlimit(1500)

def data_frame():
    url = 'C:/........................../Identifying-gray-sheep/movie_lens-100k/ratings.csv'  
    colunas = ['id_user', 'id_movie', 'ratings', 'timestemp']
    X = pd.read_csv(url, names=colunas, skiprows=0, delimiter='\t', )
    X = X.drop(columns='timestemp')
    N_users = sorted(set(X.loc[:, 'id_user']))
    M_movies = sorted(set(X.loc[:, 'id_movie']))
    table = pd.DataFrame(data=0, columns=M_movies, index=N_users)
    for row in X.itertuples():
        table.at[row.id_user, row.id_movie] = row.ratings
    table.to_csv("saidas_detector_greysheep/_data_table.csv", sep="\t")
    return table

def cosine_similarity_users():    
    url = 'C:/........................../Identifying-gray-sheep/saidas_detector_greysheep/_data_table.csv'
    df = pd.read_csv(url, delimiter='\t')
    similarity = cosine_similarity(df, df)
    df =pd.DataFrame(similarity, index=pd.RangeIndex(start=1, stop=944, name='index'), 
                     columns=pd.RangeIndex(start=1, stop=944, name='columns'))
    df.to_csv("saidas_detector_greysheep/_similarity_users.csv", sep="\t")    
    return df

def statistics_descriptive():
    url = 'C:/........................../Identifying-gray-sheep/saidas_detector_greysheep/_similarity_users.csv'
    df = pd.read_csv(url, delimiter='\t')
    df = df.drop(columns='index')
    skewness = pd.DataFrame(df.skew()).T
    df = df.describe()
    df = pd.concat([df, skewness], ignore_index=True)
    df = df.rename(index={0: 'count', 1: 'mean', 2: 'std', 3: 'min',4: '25%', 5: '50%', 6: '75%', 7: 'max', 8: 'skewness'})
    df = df.drop(index={'count', 'min', 'max'})
    df.to_csv("saidas_detector_greysheep/_statistics_descriptive.csv", sep="\t" )
    return df

def select_candidates_grey_sheep():
    url = 'C:/........................../Identifying-gray-sheep/saidas_detector_greysheep/_statistics_descriptive.csv'
    df = pd.read_csv(url, delimiter='\t', index_col=0)
    q1 = df.loc['25%'].describe()
    q2 = df.loc['50%'].describe()
    mean = df.loc['mean'].describe()
    skewness_q3 = df.loc['75%'].skew()
    df = df.T
    samples = []
    for row in df.itertuples():
        if row[3] < q1['25%'] and row[4] < q2['25%'] and row[1] < mean['25%']:            
            samples.append(row)
    for row in samples:
        if row[6] > skewness_q3:      
            samples.remove(row)
    df = pd.DataFrame(samples)
    df = df.set_index('Index')
    df.to_csv("saidas_detector_greysheep/_candidates_grey_sheep.csv", sep="\t")   
    return df

def select_candidates_white_sheep():
    url = 'C:/........................../Identifying-gray-sheep/saidas_detector_greysheep/_statistics_descriptive.csv'
    df = pd.read_csv(url, delimiter='\t', index_col=0)
    mean = df.loc['mean'].describe()
    samples = []
    df = df.T
    for row in df.itertuples():
        if row[1] > mean['75%']:
            samples.append(row)
    df = pd.DataFrame(samples)
    df = df.set_index('Index')
    df.to_csv("saidas_detector_greysheep/_candidates_white_sheep.csv", sep="\t")
    return df

def outliers_lof():
    df = pd.concat([select_candidates_white_sheep(), select_candidates_grey_sheep()])
    lof = LocalOutlierFactor(n_neighbors=50, contamination=0.24) # define k vizinhos e o limite
    y_pred = lof.fit_predict(df)
    lofs = where(y_pred ==- 1) # posições do index das anomalias
    list_outliers = []
    for line in np.nditer(lofs):
        list_outliers.append(df.iloc[line]) 
    df = pd.DataFrame(list_outliers)
    df.to_csv("saidas_detector_greysheep/lof/_outliers_k=50_contamination=0.24.csv", sep="\t")
    return df

def grey_sheep():    
    url = 'C:/........................../Identifying-gray-sheep/saidas_detector_greysheep/lof/_outliers_k=50_contamination=0.24.csv'
    df = pd.read_csv(url, delimiter='\t', index_col=0)
    indice_df = pd.DataFrame(df.index)
    url = 'C:/........................../Identifying-gray-sheep/saidas_detector_greysheep/_data_table.csv'
    grey_sheep = pd.read_csv(url, delimiter='\t', index_col=0)
    list_outliers = []
    for line in indice_df.itertuples():
        list_outliers.append(grey_sheep.iloc[line[1]-1])    
    df = pd.DataFrame(list_outliers)
    df.to_csv("saidas_detector_greysheep/ratings_grey_sheep/ratings_group_grey_sheep.csv", sep="\t")
    return df

def white_sheep():    
    url = 'C:/........................../Identifying-gray-sheep/saidas_detector_greysheep/_candidates_white_sheep.csv'
    df = pd.read_csv(url, delimiter='\t', index_col=0)
    indice_df = pd.DataFrame(df.index)
    url = 'C:/........................../Identifying-gray-sheep/saidas_detector_greysheep/_data_table.csv'
    grey_sheep = pd.read_csv(url, delimiter='\t', index_col=0)
    list_outliers = []
    for line in indice_df.itertuples():
        list_outliers.append(grey_sheep.iloc[line[1]-1])    
    df = pd.DataFrame(list_outliers)
    df.to_csv("saidas_detector_greysheep/ratings_white_sheep/ratings_group_white_sheep.csv", sep="\t")
    return df

def other_sheep():
    url = 'C:/........................../Identifying-gray-sheep/saidas_detector_greysheep/_data_table.csv'
    df = pd.read_csv(url, delimiter='\t', index_col=0)    
    df_2 = pd.concat([grey_sheep(), white_sheep()])
    df = df[~df.isin(df_2.to_dict(orient='list')).all(axis=1)]
    df.to_csv("saidas_detector_greysheep/ratings_other_sheep/ratings_group_other_sheep.csv", sep="\t" )
    return df

def remove_candidates_grey():
    grey_sheep_r = grey_sheep()
    url = 'C:/........................../Identifying-gray-sheep/saidas_detector_greysheep/_data_table.csv'
    df = pd.read_csv(url, delimiter='\t', index_col=0) 
    for i in grey_sheep_r.index:
        df = df.drop(index=int(i))
    df.to_csv("saidas_detector_greysheep/ratings_scenario_two/ratings_REMOVE_grey_sheep.csv", sep="\t" )
    return df

