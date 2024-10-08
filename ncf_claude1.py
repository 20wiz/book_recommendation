import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

data_used = 10000
# Load and preprocess the dataset
users = pd.read_csv('.\\book\\data\\Users.csv').head(data_used)
books = pd.read_csv('.\\book\\data\\Books.csv').head(data_used)
ratings = pd.read_csv('.\\book\\data\\Ratings.csv').head(data_used)

# Map user and book identifiers to indices
user_ids = ratings['User-ID'].unique().tolist()
book_isbns = ratings['ISBN'].unique().tolist()

user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
book_to_index = {isbn: idx for idx, isbn in enumerate(book_isbns)}

ratings['user'] = ratings['User-ID'].map(user_to_index)
ratings['book'] = ratings['ISBN'].map(book_to_index)

#print head of ratings
print(ratings.head())

# Normalize ratings to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
ratings['Book-Rating'] = scaler.fit_transform(ratings[['Book-Rating']])

# Split the data
X = ratings[['user', 'book']].values
y = ratings['Book-Rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model parameters
num_users = len(user_ids)
num_books = len(book_isbns)
embedding_size = 100

# Improved Neural Collaborative Filtering Model
class ImprovedNCF(nn.Module):
    def __init__(self, num_users, num_books, embedding_size):
        super(ImprovedNCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.book_embedding = nn.Embedding(num_books, embedding_size)
        
        self.fc1 = nn.Linear(embedding_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
        
    def forward(self, user, book):
        user_embedded = self.user_embedding(user)
        book_embedded = self.book_embedding(book)
        
        concat = torch.cat([user_embedded, book_embedded], dim=-1)
        x = self.activation(self.fc1(concat))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        output = torch.sigmoid(self.output(x))  # Constrain output to [0, 1]
        return output

# Custom loss function
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, pred, target):
        weights = torch.exp(target)  # Give more weight to higher ratings
        loss = weights * (pred - target) ** 2
        return loss.mean()

# Initialize the model, loss function, and optimizer
model = ImprovedNCF(num_users, num_books, embedding_size)
criterion = WeightedMSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Prepare data for training
train_users = torch.LongTensor(X_train[:, 0])
train_books = torch.LongTensor(X_train[:, 1])
train_ratings = torch.FloatTensor(y_train)

test_users = torch.LongTensor(X_test[:, 0])
test_books = torch.LongTensor(X_test[:, 1])
test_ratings = torch.FloatTensor(y_test)

# Create DataLoader for batch processing
train_dataset = TensorDataset(train_users, train_books, train_ratings)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training the model
epochs = 50
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_users, batch_books, batch_ratings in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_users, batch_books).squeeze()
        loss = criterion(predictions, batch_ratings)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_users, test_books).squeeze()
        test_loss = criterion(test_predictions, test_ratings)
        test_losses.append(test_loss.item())
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

# Final evaluation
model.eval()
with torch.no_grad():
    test_predictions = model(test_users, test_books).squeeze()
    test_loss = criterion(test_predictions, test_ratings)
    print(f'Final Test Loss: {test_loss.item():.4f}')
    
    # Convert predictions back to original scale
    test_predictions_np = scaler.inverse_transform(test_predictions.numpy().reshape(-1, 1)).flatten()
    test_ratings_np = scaler.inverse_transform(test_ratings.numpy().reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(test_ratings_np, test_predictions_np))
    mae = mean_absolute_error(test_ratings_np, test_predictions_np)
    correlation = np.corrcoef(test_ratings_np, test_predictions_np)[0, 1]
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'Correlation: {correlation:.4f}')

# Visualize the test results
plt.figure(figsize=(10, 6))
plt.scatter(test_ratings_np, test_predictions_np, alpha=0.5)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.plot([0, 10], [0, 10], 'r--')  # Perfect prediction line
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()

# Plot rating distribution
plt.figure(figsize=(10, 6))
plt.hist(test_ratings_np, bins=20, alpha=0.5, label='Actual Ratings')
plt.hist(test_predictions_np, bins=20, alpha=0.5, label='Predicted Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Actual and Predicted Ratings')
plt.legend()
plt.show()