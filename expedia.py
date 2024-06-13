import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Loading 100k data rows
# Load train data
train = pd.read_csv('/Users/iliyask/Desktop/Developer/hotel_recommendation/data/expedia-hotel-recommendations/train.csv', nrows=100000)

# Function to convert date object into relevant attributes
def convert_date_into_days(df):
    df['srch_ci'] = pd.to_datetime(df['srch_ci'])
    df['srch_co'] = pd.to_datetime(df['srch_co'])
    df['date_time'] = pd.to_datetime(df['date_time'])

    df['stay_dur'] = (df['srch_co'] - df['srch_ci']).astype('timedelta64[ns]')
    df['no_of_days_bet_booking'] = (df['srch_ci'] - df['date_time']).astype('timedelta64[ns]')

    # For hotel check-in
    # Month, Year, Day
    df['Cin_day'] = df["srch_ci"].apply(lambda x: x.day)
    df['Cin_month'] = df["srch_ci"].apply(lambda x: x.month)
    df['Cin_year'] = df["srch_ci"].apply(lambda x: x.year)

convert_date_into_days(train)

# Check the percentage of Nan in dataset
train.isnull().sum().sort_values(ascending=False)/len(train)

sns.histplot(train['orig_destination_distance'],kde=True)

#Using Median: Since the data is skewed, the median is a better measure to use for imputation.
train['orig_destination_distance'].fillna(train['orig_destination_distance'].median(), inplace=True)

train.dropna(inplace=True)

#let's have a data only hotel that were booked
train=train[train['is_booking']==1]


# Drop unnecessary columns
columns_to_drop = ['date_time', 'srch_ci', 'srch_co', 'user_id','site_name','hotel_cluster',]
train_cleaned = train.drop(columns=columns_to_drop)

# Convert timedelta columns to float (days)
train_cleaned['stay_dur'] = train_cleaned['stay_dur'].dt.days
train_cleaned['no_of_days_bet_booking'] = train_cleaned['no_of_days_bet_booking'].dt.days

# Fill any remaining missing values if necessary
train_cleaned.fillna(train_cleaned.median(), inplace=True)

# Scale the data
scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_cleaned), columns=train_cleaned.columns)

# Apply KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
train_scaled['cluster'] = kmeans.fit_predict(train_scaled)
print(train_scaled)
# Use PCA for dimensionality reduction to 10 components
pca = PCA(n_components=10)
train_pca = pd.DataFrame(pca.fit_transform(train_scaled.drop(columns=['cluster'])),
                         columns=[f'PCA{i+1}' for i in range(10)])
train_pca['cluster'] = train_scaled['cluster']

# Calculate the distance to cluster centroids
centroids = kmeans.cluster_centers_
train_pca['distance_to_centroid'] = train_scaled.apply(lambda row: np.linalg.norm(row.drop('cluster') - centroids[int(row['cluster'])]), axis=1)

# Set a threshold for anomalies
threshold = train_pca['distance_to_centroid'].mean() + 3 * train_pca['distance_to_centroid'].std()
anomalies = train_pca[train_pca['distance_to_centroid'] > threshold]

# Plot the clusters and anomalies using the first three principal components in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster in train_pca['cluster'].unique():
    cluster_data = train_pca[train_pca['cluster'] == cluster]
    ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'], cluster_data['PCA3'],
               label=f'Cluster {cluster}', alpha=0.6)

# Plot anomalies
ax.scatter(anomalies['PCA1'], anomalies['PCA2'], anomalies['PCA3'],
           color='red', label='Anomalies', s=50)

ax.set_title('Clusters and Anomalies Visualization using the first three PCA components')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.legend()
plt.show()
