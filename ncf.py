import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# 하이퍼파라미터 설정
epochs = 10
embedding_size = 64
data_used = 10000

# 데이터 로드 및 전처리
users = pd.read_csv('.\\data\\Users.csv').head(data_used)
books = pd.read_csv('.\\data\\Books.csv', dtype={'Year-Of-Publication': str}).head(data_used)
ratings = pd.read_csv('.\\data\\Ratings.csv').head(data_used)

# 사용자 및 책 ID를 인덱스로 매핑
user_ids = ratings['User-ID'].unique().tolist()
book_isbns = ratings['ISBN'].unique().tolist()

user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
book_to_index = {isbn: idx for idx, isbn in enumerate(book_isbns)}

ratings['user'] = ratings['User-ID'].map(user_to_index)
ratings['book'] = ratings['ISBN'].map(book_to_index)

# 사용자-책 상호작용 행렬 생성 (평점이 없으면 NaN)
user_book_matrix = ratings.pivot_table(index='user', columns='book', values='Book-Rating')

# NaN 값을 0으로 대체 (또는 다른 방법으로 처리 가능)
user_book_matrix = user_book_matrix.fillna(0)

# 데이터셋 분할
X = ratings[['user', 'book']].values
y = ratings['Book-Rating'].values

# MinMaxScaler를 사용하여 y를 0과 1 사이로 스케일링
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_users = torch.LongTensor(X_train[:, 0])
train_books = torch.LongTensor(X_train[:, 1])
train_ratings = torch.FloatTensor(y_train)

test_users = torch.LongTensor(X_test[:, 0])
test_books = torch.LongTensor(X_test[:, 1])
test_ratings = torch.FloatTensor(y_test)

# DataLoader 생성
train_dataset = TensorDataset(train_users, train_books, train_ratings)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 모델 정의
class NCF(nn.Module):
    def __init__(self, num_users, num_books, embedding_size):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.book_embedding = nn.Embedding(num_books, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, book):
        user_embedded = self.user_embedding(user)
        book_embedded = self.book_embedding(book)
        x = torch.cat([user_embedded, book_embedded], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 모델 초기화
num_users = len(user_ids)
num_books = len(book_isbns)
model = NCF(num_users, num_books, embedding_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    all_predictions = []
    all_targets = []
    
    for batch_users, batch_books, batch_ratings in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_users, batch_books).squeeze()
        loss = criterion(predictions, batch_ratings)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Store predictions and targets for accuracy and f1 score calculation
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_targets.extend(batch_ratings.detach().cpu().numpy())
    
    train_losses.append(epoch_loss / len(train_loader))
    
    # Convert targets and predictions to binary
    binary_targets = [1 if t >= 0.5 else 0 for t in all_targets]
    binary_predictions = [1 if p >= 0.5 else 0 for p in all_predictions]
    
    # Calculate accuracy and f1 score
    accuracy = accuracy_score(binary_targets, binary_predictions)
    f1 = f1_score(binary_targets, binary_predictions)
    
 # Calculate RMSE, MAE, and correlation
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets, all_predictions)
    correlation = np.corrcoef(all_targets, all_predictions)[0, 1]
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, Correlation: {correlation:.4f}')