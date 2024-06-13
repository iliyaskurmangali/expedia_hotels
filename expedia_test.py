import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Load train data (replace with your actual data loading)
train = pd.read_csv('/Users/iliyask/Desktop/Developer/hotel_recommendation/data/expedia-hotel-recommendations/train.csv', nrows=100000)

# Function to convert date object into relevant attributes
def convert_date_into_days(df):
    df['srch_ci'] = pd.to_datetime(df['srch_ci'])
    df['srch_co'] = pd.to_datetime(df['srch_co'])
    df['date_time'] = pd.to_datetime(df['date_time'])

    df['stay_dur'] = (df['srch_co'] - df['srch_ci']).dt.days
    df['no_of_days_bet_booking'] = (df['srch_ci'] - df['date_time']).dt.days

    df['Cin_day'] = df["srch_ci"].dt.day
    df['Cin_month'] = df["srch_ci"].dt.month
    df['Cin_year'] = df["srch_ci"].dt.year

convert_date_into_days(train)

# Handling missing values
train['orig_destination_distance'].fillna(train['orig_destination_distance'].median(), inplace=True)
train.dropna(inplace=True)
train = train[train['is_booking'] == 1]

# Keep only selected columns
selected_columns = ['no_of_days_bet_booking', 'stay_dur', 'orig_destination_distance', 'Cin_day', 'Cin_month','is_mobile', 'is_package',
       'channel']
train_cleaned = train[selected_columns]

# Scale the data
scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_cleaned), columns=train_cleaned.columns)

# Apply PCA
pca = PCA()
X_proj = pca.fit_transform(train_scaled)

# Visualize PCA results
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(X_proj).corr(), cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap of PCA Components')
plt.show()

# Explained variance ratio
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(pca.explained_variance_ratio_, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('% explained variance')
plt.title('Explained Variance by Principal Component')

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative explained variance')
plt.title('Cumulative Explained Variance')
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_proj)

# Visualize KMeans clustering
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=kmeans.labels_, cmap='viridis', s=20)
plt.title('KMeans Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 2, 2)
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=train['hotel_cluster'], cmap='viridis', s=20)
plt.title('True Hotel Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

# Evaluate K-Nearest Neighbors with PCA
pca3 = PCA(n_components=3)
X_proj3 = pca3.fit_transform(train_scaled)

print("Accuracy using 3 PCs:")
print(cross_val_score(KNeighborsClassifier(), X_proj3, train['hotel_cluster'], cv=5).mean())

print("\nAccuracy using all initial features:")
print(cross_val_score(KNeighborsClassifier(), train_scaled, train['hotel_cluster'], cv=5).mean())
