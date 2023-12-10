# 연극 뮤지컬 concat
import pandas as pd


def concat_play_musical(play, musical):
  try:
    playdf = pd.read_excel(play)
    musicaldf = pd.read_excel(musical)
  except:
    playdf = pd.read_csv(play)
    musicaldf = pd.read_csv(musical)
  df = pd.concat([playdf, musicaldf])
  df = df.drop(['Unnamed: 0'], axis=1)
  df = df.reset_index(drop=True)
  return df

# 파일 경로를 함수화하면, 모든 코드가 복잡해지므로 전역변수화 시켰음

play_top = "데이터\연극\연극_TOP77_장소병합.xlsx"
musical_top = "데이터\뮤지컬\뮤지컬_TOP77.xlsx"
play_top_star = "데이터\연극\playTop77_star.xlsx"
musical_top_star = "데이터\뮤지컬\musicalTop77_star.xlsx"
play_actor_lank = "데이터\연극\연극 배우 랭킹.csv"
musical_actor_lank = "데이터\뮤지컬\뮤지컬 배우 랭킹.csv"
play_top_cast = "데이터\연극\playTop77_casting.xlsx"
musical_top_cast = "데이터\뮤지컬\musicalTop77_casting.xlsx"

play_top_poster = '데이터\연극\연극_TOP77_장소병합 (+포스터).xlsx'
musical_top_poster = '데이터\뮤지컬\뮤지컬_TOP77 (+포스터).xlsx'
play_stat = '데이터\연극\연극_통계_e.xlsx'
musical_stat = '데이터\뮤지컬\뮤지컬_통계_e.xlsx'

top_lank = concat_play_musical(play_top, musical_top)
top_star = concat_play_musical(play_top_star, musical_top_star)
top_cast = concat_play_musical(play_top_cast, musical_top_cast)
actor_lank = concat_play_musical(play_actor_lank, musical_actor_lank)

top_poster = concat_play_musical(play_top_poster, musical_top_poster)
top_poster = top_poster.drop('Image', axis=1)

stat = concat_play_musical(play_stat, musical_stat)
stat = stat.drop('Title', axis=1)

# 장르
import os
import pandas as pd
import seaborn as sns
import numpy as np
import random
from matplotlib import pyplot as plt
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

def genre_similarlity(top70_df, title_name):
  df=top70_df[['Title', '장르']]
  from sklearn.feature_extraction.text import TfidfVectorizer

  #df 장르에 존재하는 공백 형식이 갖추어지지 않아있어서 공백있는거 공백 제거하기
  df['장르']=df['장르'].str.replace(' ', '').str.replace(',', ', ')

  #그 다음 ,를 기준으로 장르 나눠주기
  df['genres_literal']=df['장르'].str.split(',')
  df['genres_literal'] = df['genres_literal'].apply(lambda x : (' ').join(x))
  #df
  tfidf_vect = TfidfVectorizer()
  genre_mat1=tfidf_vect.fit_transform(df['genres_literal'])

  from sklearn.metrics.pairwise import cosine_similarity

  genre_sim1 = cosine_similarity(genre_mat1, genre_mat1)#똑같은 메트릭스를 두개 집어넣는다.

  genre_sim_sorted_ind1 = genre_sim1.argsort()[:, ::-1] #내림차순으로 sorting해서 보여줘.

  # 인자로 입력된 movies_df DataFrame에서 'title' 컬럼이 입력된 title_name 값인 DataFrame 추출
  title_movie = df[df['Title'] == title_name]

  # title_name을 가진 DataFrame의 index 객체를 ndarray로 반환하고
  # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n 개의 index 추출
  title_index = title_movie.index.values
  similar_indexes = genre_sim_sorted_ind1[title_index, :]  # Include one extra for the input movie

  # 추출된 top_n+1 index들 출력. top_n+1 index는 2차원 데이터임.
  # dataframe에서 index로 사용하기 위해서 1차원 array로 변경
  similar_indexes = similar_indexes.reshape(-1)

  # 유사한 영화 중에 유사도가 높은 순으로 정렬한 인덱스에 대응되는 유사도 추출
  similar_scores = genre_sim1[title_index, similar_indexes]

  # DataFrame으로 만들고 유사도가 높은 순으로 정렬
  similar_movie_df = df.iloc[similar_indexes]
  similar_movie_df['Cosine_Similarity'] = similar_scores  # Add cosine similarity scores to the DataFrame

  # Round the cosine similarity values to two decimal places
  similar_movie_df['Cosine_Similarity'] = similar_movie_df['Cosine_Similarity'].round(2)

  similar_movie_df = similar_movie_df[['Title', 'Cosine_Similarity']]

  # Exclude the input movie from the result
  similar_movie_df = similar_movie_df[similar_movie_df['Title'] != title_name]

  return similar_movie_df

def genre_sim(title):
  similar_df = genre_similarlity(top_lank, title)
  return similar_df

# 줄거리 + 이미지 
import nltk
#nltk.download('punkt')

import torch
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
from konlpy.tag import Okt, Kkma, Komoran
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import networkx
from tqdm.autonotebook import tqdm
from tqdm import tqdm, tqdm_notebook

import seaborn as sns
import matplotlib.pyplot as plt

from gensim.models import word2vec
import itertools

from PIL import Image
import os
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset

# import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from numpy.linalg import norm
import math
import time
import pickle
import unicodedata


def load_pickle(filename):
    with open(filename, 'rb') as f:
        res = pickle.load(f)
    return res

location = '데이터/'
plot_image_v = load_pickle(location+'sbert_vgg16_vecs.pickle')
top_stat = pd.concat([plot_image_v, stat], axis=1)

def plot_image_sim(title):
  target_index = top_poster[top_poster['Title'] == title].index[0]

  # 코사인 유사도
  cosine_sim = cosine_similarity(top_stat)

  # 특정 작품과 다른 작품들 간의 유사도
  target_similarities = cosine_sim[target_index]

  plot_image_weight = pd.DataFrame([top_poster['Title'], target_similarities], index=['Title', 'Similarity']).T

  return plot_image_weight

# 별점
# user_ratings : 별점리스트, review_count: 전체 행개수
def calculate_weighted_rating (user_ratings, review_count, base_weight=1.0):

  # 가중치 계산
  weight = base_weight*( 1.0 + review_count)    # 사람 수가 적을수록 가중치 감소/ 기본 가중치값 먼저 필요

  # 평균 별점 계산
  average_rating = sum(user_ratings)/len(user_ratings)   # 작품명 같은 것끼리 계산

  # 가중 평균 별점 계산
  weighted_rating = average_rating*weight

  return weighted_rating

def star_weight():
  import pandas as pd
  star_df = top_star
  star = star_df['Star']
  num_play = star_df.shape[0]

  # 일단 작품이름 같은 것끼리 묶어야함
  # 묶은 것에서 user_rating 리스트 만들기
  # 묶은 것 아이템 개수=작품별 리뷰 개수 계산
  # for문 이용해서 작품 하나씩 들어가서 돌려야함.
  # 결과  --> 작품: 가중치 이런식으로 아웃풋.

  # 제목별로 별점 그룹화
  grouped_title = star_df.groupby('Title')['Star'].apply(list).reset_index()
  type(grouped_title)

  # 별점 가중치 함수 데이터에 적용

  review_count = 100
  weighted_ratings = {}

  for row in grouped_title.itertuples():
    title = row.Title
    user_ratings = row.Star

    # 가중 평균 계산 함수 호출
    weighted_rating = calculate_weighted_rating(user_ratings, review_count=len(user_ratings))
    weighted_ratings[title] = weighted_rating
  '''
  for title, weighted_rating in weighted_ratings.items():
      print(f"Title: {title}, Weighted Rating: {weighted_rating}")
  '''
  return weighted_ratings

weighted_ratings = star_weight()

# 배우
import pandas as pd
def actor_lanking(actor_csv):
  actor_csv = actor_csv.reset_index()
  actor_csv.columns = ["index", "actor_name", "masterpiece"]

  # 랭킹에 있는 배우들의 작품만 따로 추출 (작품명이 \n으로 이어진 하나의 문자열이기 때문에, 리스트화 시킬 예정)
  actor_mp = actor_csv['masterpiece'].to_list()

  actor_dic = actor_csv.set_index('index').T.to_dict()  # 딕셔너리 형태로 변환

  # key: actor_name, value: index, masterpiece
  for key, value in actor_dic.items():
    temp = value['masterpiece']
    temp2 = temp.split('\n')
    value['masterpiece'] = temp2
  return actor_dic

def cast_actor(inter_actor):
  inter_actor = inter_actor[['Title', 'cast_act']]

  # 문자열 형태로 되어 있는 cast_act 열을 리스트화 시켜서 다시 집어 넣을 예정
  inter_actor_cast = inter_actor['cast_act']

  inter_list = []
  for i in range(len(inter_actor)):
    string = inter_actor_cast[i]
    ch_string = string.split(', ')
    inter_list.append(ch_string)

  # Title 기준 딕셔너리화
  inter_dic = inter_actor.set_index('Title').T.to_dict()

  # key: Title, cast_act
  i = 0
  for key, value in inter_dic.items():
    value['cast_act'] = inter_list[i]
    i+=1
  return inter_dic, inter_actor

# 배우 이름 찾기 (랭킹 도출) 함수
def find_value(dictionary, finding_value):
    for key, value in dictionary.items():
        if (finding_value.find(value['actor_name']) > -1):
            return key

def search_actor(inter_dic, actor_dic):
  actor_mp = {}
  i = 0
  for key1, value1 in inter_dic.items():
      for name in value1['cast_act']:
          index = find_value(actor_dic, name)
          if index is not None:
              for key2, value2 in actor_dic.items():
                  if key2 == index:
                    # 여기서부터 짰음
                      recent_performances = value2['masterpiece']

                      for title in recent_performances:
                        title = title.split('(')
                        title = title[0]

                        # 인터파크 제목에서 최근 공연을 찾음
                        temp = key1.find(title)
                        if temp > -1:
                          #print(f"배우 {name}의 최근 공연 '{key1}'이 인터파크 제목에 포함됨.")
                          actor_mp[i] = {'actor' : name, 'Title' : key1}
                          i+=1
  return actor_mp

def actor_name_dic(actor_dic, inter_dic, SA):
  dic = {}
  for key1, value1 in actor_dic.items():
    dic[key1] = value1['actor_name']


  index = 200  # 순위권 이후 배우 추가용
  for key1, value1 in inter_dic.items():
    for i in value1['cast_act']:  # 공연별 캐스팅 배우 딕셔너리화
      ret = 0 # 순위권 배우 이름 중복 추가 방지용
      for key2, value2 in SA.items(): # 순위권 배우 판별
        if (value2['Title'] == key1 and value2['actor'] == i) or (i == '해당없음'): # 순위권 배우거나, 배우명이 없는 경우 추가 제외
          ret = 1
          break
      if(ret == 0):
        for ran in range(100, index):  # 동명이인 찾기
          if(dic[ran] == i ):
            ret = 1
            break
          else:
            ret = 0

      if(ret == 0): # 그 외 전부 추가
        dic[index] = i
        index += 1
  dic_t = {v:k for k,v in dic.items()}
  return dic, dic_t

def change_index(search, dic):
  ret_dic = {}
  for key, value in search.items():
    index = dic[value['actor']]
    ret_dic[index] = value['Title']
  return ret_dic

def cast_list(dic1, dic2, dic_t):
  cast_dic = {}
  for key1, value1 in dic2.items():
    cast_act = value1['cast_act']
    cast_index = []
    # 배우 인덱스 저장
    for name in cast_act:
      cast_index.append(dic_t.get(name))
    cast_dic[key1] = cast_index
  return cast_dic

def actor_rate(cast_dic, title_name, ci):
  score_list = []
  cast_list = cast_dic[title_name]
  for key, value in cast_dic.items():
    score = 0
    for index in cast_list:
      # 이 공연의 배우가 있을 경우 추천점수 +1점
      if((key != title_name) and (index in value)):
        score += 1
      # 유명한 배우가 있을 경우 추천점수 +2.5점
    for name, title in ci.items():
      if(key == title):
        score += 2.5
    score_list.append(score)
  return score_list

actor_dic = actor_lanking(actor_lank)
inter_dic, inter_actor = cast_actor(top_cast)
SA = search_actor(inter_dic, actor_dic)
pt_actor_dic, pt_actor_dic_t = actor_name_dic(actor_dic, inter_dic, SA)
ci = change_index(SA, pt_actor_dic_t) # 랭킹배우의 이름->인덱스

# 가중치 총합
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((0,5))

def weight_calc(weighted_ratings):
  weighted_df = pd.DataFrame(weighted_ratings, index = ['StarScore']).T
  weighted_df[:] = scaler.fit_transform(weighted_df[:])
  return weighted_df

def castscore_calc(title, actor_dic, inter_dic, actor_dic_t, ci, inter_actor):
  cast_dic = cast_list(actor_dic, inter_dic, actor_dic_t)
  score_list = actor_rate(cast_dic, title, ci)
  inter_actor['actor_score'] = score_list
  casting = inter_actor.sort_values(by=['actor_score'], ascending=False)
  casting = casting[['Title', 'actor_score']]
  casting = casting.reset_index(drop=True)

  cast = {}
  for i in range(len(casting)):
    key = casting['Title'][i]
    value = casting['actor_score'][i]
    cast[key] = value
    castscore_df = pd.DataFrame(cast, index = ['CastScore']).T
    castscore_df[:] = scaler.fit_transform(castscore_df[:])

  return castscore_df

def genre_calc(title, genre_sim):
  genre_weight = genre_sim.rename(columns = {'Title' : 'Title', 'Cosine_Similarity' : 'GenreScore'})
  genre_weight = genre_weight.set_index(keys=['Title'])
  genre_weight[:] = scaler.fit_transform(genre_weight[:])
  return genre_weight

def plot_image_calc(title):
  plot_image_weight = plot_image_sim(title)
  pi_weight = plot_image_weight.rename(columns = {'Title':'Title', 'Similarity':'PIScore'})
  pi_weight = pi_weight.set_index(keys=['Title'])
  pi_weight = pi_weight.sort_values(by=['PIScore'], ascending=False)
  pi_weight = pi_weight.iloc[1:]
  pi_weight[:] = scaler.fit_transform(pi_weight[:])
  return pi_weight

# 줄거리+이미지 / 장르
def merge_calc_pi(weighted_df, castscore_df, pi_df, genre_df, pm = 0):
  func_list = []
  w_df = pd.concat([weighted_df, castscore_df, pi_df, genre_df], axis=1, join='inner')
  for row in w_df.iterrows():
    #print(row[1]['Weight'])
    ss = row[1]['StarScore']
    cs = row[1]['CastScore']
    ps = row[1]['PIScore']
    gs = row[1]['GenreScore']
    if (pm == 0): # 연극일 경우
      func = gs * 0.35 + ps * 0.3 + cs * 0.2 + ss * 0.15
    else:  # 뮤지컬일 경우
      func = cs * 0.35 + gs * 0.3 + ps * 0.2 + ss * 0.15
    func_list.append([row[0], func])
    #print(f"Title: {row[0]}, Weighted Rating: {func}")
  return func_list

def score_pg(title):
  weighted_df = weight_calc(weighted_ratings)
  castscore_df = castscore_calc(title, pt_actor_dic, inter_dic, pt_actor_dic_t, ci, inter_actor)
  pi_df = plot_image_calc(title)
  genre_sm = genre_sim(title)
  genre_df = genre_calc(title, genre_sm)
  func_list = merge_calc_pi(weighted_df, castscore_df, pi_df, genre_df)
  play_score = pd.DataFrame(func_list, columns=['Title', 'Score']).sort_values('Score', ascending=False)
  return play_score

# 협업 필터링
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def get_rmse(R, P, Q, non_zeros):
    error = 0
    # 두개의 분해된 행렬 P와 Q.T의 내적 곱으로 예측 R 행렬 생성
    full_pred_matrix = np.dot(P, Q.T)

    # 실제 R 행렬에서 널이 아닌 값의 위치 인덱스 추출하여 실제 R 행렬과 예측 행렬의 RMSE 추출
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]

    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]

    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)

    return rmse

def matrix_factorization(R, K, steps=200, learning_rate=0.01, r_lambda = 0.01):
    num_users, num_items = R.shape
    # P와 Q 매트릭스의 크기를 지정하고 정규분포를 가진 랜덤한 값으로 입력합니다.
    np.random.seed(1)
    P = np.random.normal(scale=1./K, size=(num_users, K))
    Q = np.random.normal(scale=1./K, size=(num_items, K))

    break_count = 0

    # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트 객체에 저장.
    non_zeros = [ (i, j, R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j] > 0 ]

    # SGD기법으로 P와 Q 매트릭스를 계속 업데이트.
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구함
            eij = r - np.dot(P[i, :], Q[j, :].T)
            # Regularization을 반영한 SGD 업데이트 공식 적용
            P[i,:] = P[i,:] + learning_rate*(eij * Q[j, :] - r_lambda*P[i,:])
            Q[j,:] = Q[j,:] + learning_rate*(eij * P[i, :] - r_lambda*Q[j,:])

        rmse = get_rmse(R, P, Q, non_zeros)
        '''
        if (step % 10) == 0 :
            print("### iteration step : ", step," rmse : ", rmse)
        '''

    return P, Q

# 성능평가를 위해 return 값을 unseen_list > movies_list로 변경
def get_unseen_movies(ratings_matrix, userId):
    # userId로 입력받은 사용자의 모든 영화정보 추출하여 Series로 반환함.
    # 반환된 user_rating 은 영화명(title)을 index로 가지는 Series 객체임.
    user_rating = ratings_matrix.loc[userId,:]

    # user_rating이 0보다 크면 기존에 관람한 영화임. 대상 index를 추출하여 list 객체로 만듬
    already_seen = user_rating[ user_rating > 0].index.tolist()

    # 모든 영화명을 list 객체로 만듬.
    movies_list = ratings_matrix.columns.tolist()

    # list comprehension으로 already_seen에 해당하는 movie는 movies_list에서 제외함.
    unseen_list = [ movie for movie in movies_list if movie not in already_seen]

    return movies_list

def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    # 예측 평점 DataFrame에서 사용자id index와 unseen_list로 들어온 영화명 컬럼을 추출하여
    # 가장 예측 평점이 높은 순으로 정렬함.
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_movies

total_star = top_star.reset_index(drop=True)
# 동일한 ID가 동일한 Title에 대해서 남긴 평점은 평균값으로 처리
pivot_df = total_star.pivot_table(index='ID', columns='Title', values='Star', aggfunc='mean')
# *** 아이디 삭제
pivot_df = pivot_df.drop(pivot_df.index[0])
pivot_df = pivot_df.fillna(0)
P, Q = matrix_factorization(pivot_df.values, K=50, steps=200, learning_rate=0.01, r_lambda = 0.01)
pred_matrix = np.dot(P, Q.T)
ratings_pred_matrix = pd.DataFrame(data=pred_matrix, index= pivot_df.index, columns = pivot_df.columns)

# 메인
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 컨텐츠 (장르 / 줄거리+이미지)
# 여기만 바뀜 score(title) ==> score_pg(title)
def score_list(high_rated_titles):
  # 사용자의 가장 높은 평점 작품 : score 함수 => 합산
  scaler = MinMaxScaler((0,0.7))
  scaler_10 = MinMaxScaler((0,10))
  df = pd.DataFrame()
  for title in high_rated_titles:
    score_df = score_pg(title)
    df = pd.concat([df, score_df])
  df1 = df.groupby('Title')['Score'].sum()
  df1 = df1.sort_values(axis=0, ascending=False)
  df = df1.to_frame()
  df[:] = scaler.fit_transform(df[:])
  #df['Score'] = list(map(lambda x: 10-x, range(10)))
  return df

# 협업
def collab_list(userid):
  scaler = MinMaxScaler((0,0.3))
  scaler_10 = MinMaxScaler((0,10))
  # user의 가장 높은 평점 찾기, 해당 타이틀 추출
  total_star = top_star.reset_index(drop=True)
  watched_titles = total_star[total_star['ID'] == userid]
  max_rating = watched_titles['Star'].max()
  high_rated_titles = watched_titles[watched_titles['Star']== max_rating]['Title'].unique()
  # 협업필터링
  unseen_list = get_unseen_movies(pivot_df, userid)
  predicted_movies = recomm_movie_by_userid(ratings_pred_matrix, userid, unseen_list, top_n=130)

  ## 평점 데이타를 DataFrame으로 생성.
  recomm_movies = pd.DataFrame(data=predicted_movies.values,index=predicted_movies.index,columns=['pred_score'])
  recomm_movies_idx  = recomm_movies.reset_index()
  recomm_movies_idx = recomm_movies_idx.set_index(keys=['Title'])
  recomm_movies_idx[:] = scaler.fit_transform(recomm_movies_idx[:])

  #recomm_movies_idx['score'] = list(map(lambda x: 10-x, range(10)))
  return recomm_movies_idx, high_rated_titles


def mat_content_image(userid):
  recomm_movies_idx, high_rated_titles = collab_list(userid)
  # 컨텐츠 기반 합산
  scores_df = score_list(high_rated_titles)

  total_df = pd.merge(scores_df, recomm_movies_idx, left_index=True, right_index=True, how='outer')
  total_df['score'] = total_df['Score'] + total_df['pred_score']
  total_df = total_df[['score']]

  total_score = total_df.sort_values('score', ascending=False).head(10)

  return total_score

def find_title(top_lank, title_input):
  title_str = title_input.replace(" ", "")
  title_strip = top_lank['Title'].str.replace(" ", "")
  title_index = title_strip[title_strip.str.contains(title_str)].index[0]
  return top_lank.loc[title_index]['Title']

def predict(userid):
   result = mat_content_image(userid)
   df_1 = pd.read_csv('데이터\연극+뮤지컬_TOP77.csv')
   df_1 = df_1.drop(df_1.columns[0], axis=1)
   df_1['Image'] = [unicodedata.normalize('NFC', filename) for filename in df_1['Image']]
   result['Image'] = result.index.map(lambda x: df_1[df_1['Title'] == find_title(df_1, x)]['Image'].iloc[0])
   return result

#import joblib
#joblib.dump(model, './model_test_id.pkl')
