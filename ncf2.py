import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

# hyperparameters and constants
EPOCHS = 10
EMBEDDING_SIZE = 64
DATA_USED = 100000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DATA_PATH = './data/'

# Data loading and preprocessing function
def load_and_preprocess_data(data_used, data_path):
    # users = pd.read_csv(f'{data_path}Users.csv').head(data_used)
    # books = pd.read_csv(f'{data_path}Books.csv', dtype={'Year-Of-Publication': str}).head(data_used)
    ratings = pd.read_csv(f'{data_path}Ratings.csv').head(data_used)
    
    # Exclude ratings with a value of 0
    ratings = ratings[ratings['Book-Rating'] != 0]
    
    # Map user and book IDs
    user_ids = ratings['User-ID'].unique().tolist()
    book_isbns = ratings['ISBN'].unique().tolist()
    
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    book_to_index = {isbn: idx for idx, isbn in enumerate(book_isbns)}
    
    ratings['user'] = ratings['User-ID'].map(user_to_index)
    ratings['book'] = ratings['ISBN'].map(book_to_index)
    
    # Create user-book interaction matrix (optional)
    # user_book_matrix = ratings.pivot_table(index='user', columns='book', values='Book-Rating')
    # user_book_matrix = user_book_matrix.fillna(-1)
    
    # Split dataset
    X = ratings[['user', 'book']].values
    y = ratings['Book-Rating'].values
    
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, len(user_ids), len(book_isbns)

# model definition
class NCF(nn.Module):
    def __init__(self, num_users, num_books, embedding_size):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.book_embedding = nn.Embedding(num_books, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)  # dropout layer
    
    def forward(self, user, book):
        user_embedded = self.user_embedding(user)
        book_embedded = self.book_embedding(book)
        x = torch.cat([user_embedded, book_embedded], dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # regression task
        return x.squeeze()  # output shape: (batch_size,)

# visualization function
def plot_metrics(epochs, train_losses, rmses, maes, r2_scores):
    plt.figure(figsize=(18, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Train Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    
    # RMSE plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), rmses, marker='o', label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # data loading and preprocessing
    X_train, X_test, y_train, y_test, num_users, num_books = load_and_preprocess_data(DATA_USED, DATA_PATH)
    
    # tensor conversion
    train_users = torch.LongTensor(X_train[:, 0])
    train_books = torch.LongTensor(X_train[:, 1])
    train_ratings = torch.FloatTensor(y_train)
    
    test_users = torch.LongTensor(X_test[:, 0])
    test_books = torch.LongTensor(X_test[:, 1])
    test_ratings = torch.FloatTensor(y_test)
    
    # DataLoader 
    train_dataset = TensorDataset(train_users, train_books, train_ratings)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # model initialization
    model = NCF(num_users, num_books, EMBEDDING_SIZE).to(device)
    criterion = nn.MSELoss()  # 회귀 손실 함수로 변경
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # initialize lists to store metrics
    train_losses = []
    rmses = []
    maes = []
    r2_scores = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch_users, batch_books, batch_ratings in train_loader:
            batch_users = batch_users.to(device)
            batch_books = batch_books.to(device)
            batch_ratings = batch_ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_users, batch_books)
            loss = criterion(predictions, batch_ratings)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_users.size(0)
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(batch_ratings.detach().cpu().numpy())
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        # calculate metrics
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        rmses.append(rmse)
        maes.append(mae)
        r2_scores.append(r2)
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
    
    # visualize metrics
    plot_metrics(EPOCHS, train_losses, rmses, maes, r2_scores)
    
    #  test set evaluation
    model.eval()
    with torch.no_grad():
        test_users = test_users.to(device)
        test_books = test_books.to(device)
        test_ratings = test_ratings.to(device)
        
        test_predictions = model(test_users, test_books).cpu().numpy()
        test_targets = test_ratings.cpu().numpy()
        
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
        test_mae = mean_absolute_error(test_targets, test_predictions)
        test_r2 = r2_score(test_targets, test_predictions)
        
        print(f'Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}')
    
    print_hyperparameters()

def print_hyperparameters():
    print(f'Hyperparameters:')
    print(f'  Epochs: {EPOCHS}')
    print(f'  Embedding Size: {EMBEDDING_SIZE}')
    print(f'  Data Used: {DATA_USED}')
    print(f'  Batch Size: {BATCH_SIZE}')
    print(f'  Learning Rate: {LEARNING_RATE}')

if __name__ == "__main__":
    main()
