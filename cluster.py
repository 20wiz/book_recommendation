import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

epochs = 10
embedding_size = 64
data_used = 10000

# Load and preprocess the dataset
users = pd.read_csv('.\\data\\Users.csv').head(data_used)
books = pd.read_csv('.\\data\\Books.csv').head(data_used)
ratings = pd.read_csv('.\\data\\Ratings.csv').head(data_used)

# Map user and book identifiers to indices
user_ids = ratings['User-ID'].unique().tolist()
book_isbns = ratings['ISBN'].unique().tolist()

user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
book_to_index = {isbn: idx for idx, isbn in enumerate(book_isbns)}

ratings['user'] = ratings['User-ID'].map(user_to_index)
ratings['book'] = ratings['ISBN'].map(book_to_index)

#print head of ratings
# print(ratings.head())

user_raiting_counts = ratings['User-ID'].value_counts()
top_5_users = user_raiting_counts.nlargest(5)
print('top 5 rating users')
print(top_5_users)

# Normalize ratings to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
ratings['Book-Rating'] = scaler.fit_transform(ratings[['Book-Rating']])


# # 사용자별 평가(rating) 수 계산
# user_rating_counts = ratings.groupby('user')['Book-Rating'].count().values.reshape(-1, 1)

# # KMeans 클러스터링 수행 (클러스터 개수는 3으로 설정했지만, 필요에 따라 변경 가능)
# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(user_rating_counts)

# # 클러스터링 결과를 사용자별 평가 수와 함께 시각화
# plt.figure(figsize=(10, 6))
# plt.scatter(user_rating_counts, np.zeros_like(user_rating_counts), c=clusters, cmap='viridis', alpha=0.5)
# plt.xlabel('Number of Ratings')
# plt.title('User Clustering based on Number of Ratings')
# plt.grid(True)
# plt.show()


# 사용자-책 상호작용 행렬 생성 (사용자가 해당 책에 평가했으면 1, 아니면 0)
user_book_matrix = ratings.pivot_table(index='user', columns='book', values='Book-Rating', fill_value=0)

# KMeans 클러스터링 수행 (클러스터 개수는 3으로 설정, 필요시 변경 가능)
kmeans = KMeans(n_clusters=300, random_state=42)
clusters = kmeans.fit_predict(user_book_matrix)

# 클러스터 결과를 사용자 인덱스에 추가
user_book_matrix['cluster'] = clusters

# 클러스터링 결과 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(x=user_book_matrix.index, y=clusters, hue=clusters, palette='viridis', alpha=0.7)
plt.xlabel('User Index')
plt.ylabel('Cluster')
plt.title('User Clustering Based on Book Ratings')
plt.grid(True)
plt.show()