import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset

# hyperparameters
epochs = 10
embedding_size = 64
data_used = 10000

# data loading and preprocessing
users = pd.read_csv('.\\data\\Users.csv').head(data_used)
books = pd.read_csv('.\\data\\Books.csv', dtype={'Year-Of-Publication': str}).head(data_used)
ratings = pd.read_csv('.\\data\\Ratings.csv').head(data_used)


user_ids = ratings['User-ID'].unique().tolist()
book_isbns = ratings['ISBN'].unique().tolist()

user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
book_to_index = {isbn: idx for idx, isbn in enumerate(book_isbns)}

ratings['user'] = ratings['User-ID'].map(user_to_index)
ratings['book'] = ratings['ISBN'].map(book_to_index)

# user-book interaction matrix
user_book_matrix = ratings.pivot_table(index='user', columns='book', values='Book-Rating')

# Nan values are replaced with -1
user_book_matrix = user_book_matrix.fillna(-1)

# data split
X = ratings[['user', 'book']].values
y = ratings['Book-Rating'].values

# MinMaxScaler is used to scale y between 0 and 1
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_users = torch.LongTensor(X_train[:, 0])
train_books = torch.LongTensor(X_train[:, 1])
train_ratings = torch.FloatTensor(y_train)

test_users = torch.LongTensor(X_test[:, 0])
test_books = torch.LongTensor(X_test[:, 1])
test_ratings = torch.FloatTensor(y_test)

# DataLoader 
train_dataset = TensorDataset(train_users, train_books, train_ratings)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# model definition
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

# model initialization
num_users = len(user_ids)
num_books = len(book_isbns)
model = NCF(num_users, num_books, embedding_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
train_losses = []
test_losses = []
accuracies = []
rmses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    all_predictions = []
    all_targets = []
    
    for batch_users, batch_books, batch_ratings in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_users, batch_books).squeeze()  # (batch_size, 1) -> (batch_size,)
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
    
    accuracies.append(accuracy)
    rmses.append(rmse)
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}')

# visualize metrics
plt.figure(figsize=(6, 5))

# Accuracy plot
# plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), accuracies, marker='o', label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Precision, Recall, F1 Score
precision = precision_score(binary_targets, binary_predictions)
recall = recall_score(binary_targets, binary_predictions)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(binary_targets, all_predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print(f'Hyperparameters:')
print(f'  Epochs: {epochs}')
print(f'  Embedding Size: {embedding_size}')
print(f'  Data Used: {data_used}')
print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]}')