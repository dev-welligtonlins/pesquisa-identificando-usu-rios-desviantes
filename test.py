import detector_grey_sheep
# from collaborative_filter import data_movies, data_ratings
import collaborative_filter

## Organização dos dados 
# X_df = detector_grey_sheep.data_frame()

## Similaridade entre os usuários
# X_df = detector_grey_sheep.cosine_similarity_users()

## Representação estatística dos usuários
# X_df = detector_grey_sheep.statistics_descriptive()

## Classificando candidatos grey sheep users
# X_df = detector_grey_sheep.select_candidates_grey_sheep()

## Classificando candidatos wite sheep users
# X_df = detector_grey_sheep.select_candidates_grey_sheep()

## Detectando outliers - LOF
# X_df = detector_grey_sheep.outliers_lof()

## Ratings do grupo GREY SHEEP
# X_df = detector_grey_sheep.grey_sheep()

## Ratings do grupo WHITE SHEEP
# X_df = detector_grey_sheep.white_sheep()

## Ratings do grupo OTHER SHEEP
# X_df = detector_grey_sheep.other_sheep()

## Ratings REMOVENDO GREY SHEEP
# X_df = detector_grey_sheep.remove_candidates_grey()

### COLLABORATIVE FILTER

## Data movies
# X_df = collaborative_filter.data_movies()

## Data user
# X_df = collaborative_filter.data_ratings()

## Data Frame USERS x MOVIES
# X_df = collaborative_filter.data_frame_recommender()

## Similaridade do usuário alvo
# X_df = collaborative_filter.calcular_media(1)

## Previsão de avaliação
X_df = collaborative_filter.prediction_rating(0, 1, 5)

