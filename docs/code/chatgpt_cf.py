import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'ad1': [1, 0, 1, 0],
    'ad2': [1, 1, 0, 0],
    'ad3': [0, 1, 1, 1],
    'ad4': [0, 0, 1, 1],
}

users = ['userA', 'userB', 'userC', 'userD']
df = pd.DataFrame(data, index=users)

print(df)

item_similarity = cosine_similarity(df.T)  
item_similarity_df = pd.DataFrame(item_similarity, index=df.columns, columns=df.columns)

print(item_similarity_df.round(2))


user = 'userA'
clicked_ads = df.loc[user]
print(f"clicked_ads: {clicked_ads}")
unseen_ads = clicked_ads[clicked_ads == 0].index.tolist()  
print(f"\n{user} {unseen_ads}")


recommend_scores = {}
for ad in unseen_ads:

    sim_scores = item_similarity_df.loc[ad, clicked_ads[clicked_ads == 1].index]
    print(f" {ad} sim_scores: {sim_scores}")
    recommend_scores[ad] = sim_scores.sum()
    print(f" {ad} recommend_scores: {recommend_scores[ad]}")


recommendation = sorted(recommend_scores.items(), key=lambda x: x[1], reverse=True)
print(f"\n {user}:")
for ad, score in recommendation:
    print(f"{ad}: {score:.2f}")