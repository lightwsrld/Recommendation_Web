import pandas as pd
import unicodedata
import pickle
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def image_plus(title_name):
    result = []

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

    play_top_poster = '데이터\연극\연극_TOP77_장소병합 (+포스터).xlsx'
    musical_top_poster = '데이터\뮤지컬\뮤지컬_TOP77 (+포스터).xlsx'

    df = concat_play_musical(play_top_poster, musical_top_poster)
    df['Image'] = [unicodedata.normalize('NFC', filename) for filename in df['Image']]
    location = '데이터/'

    if df["Title"].isin([title_name]).any():
        target_idx = df[df["Title"] == title_name].index.to_list()[0]

        with open(location+'image+plot_vecs.pickle', 'rb') as f:
            review_vecs = pickle.load(f)
        with open(location+'genres_vecs.pickle', 'rb') as f:
            gen_vecs = pickle.load(f)
        with open(location+'demo_vecs.pickle', 'rb') as f:
            demo_vecs = pickle.load(f)

        review_numpy = np.squeeze(np.array(review_vecs))
        review_numpy = pd.DataFrame(review_numpy)
        gen_numpy = np.squeeze(np.array(gen_vecs))
        gen_numpy = pd.DataFrame(gen_numpy)

        x1 = pd.concat([review_numpy, gen_numpy], axis=1)
        x2 = pd.concat([x1, demo_vecs], axis=1)

        dist_mtx = euclidean_distances(x2, x2)

        title_name = df['Image'].iloc[target_idx]

        close_list = dist_mtx[target_idx].argsort()[1:11]
        # result.append("가장 가까운 이미지")
        # result.append("======================")
        # target을 포함해 target과 가장 가까운 것 10개

        data = []
        for i, idx in enumerate(close_list):
            tit = df['Title'].iloc[idx]
            img = df['Image'].iloc[idx]
            data.append([i, tit, img, dist_mtx[target_idx][idx]])

        result = pd.DataFrame(data, columns=['Index', 'Title', 'Image', 'Distance']).set_index('Index')
        
    return result