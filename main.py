import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_and_save(input_file, output_file):
    # Load data
    df = pd.read_csv(input_file)

    # Extract features
    if 'id' in df.columns:
        X = df.drop(columns=['id']).values
    else:
        X = df.values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Number of clusters = 4n - 1
    n_dimensions = X.shape[1]
    n_clusters = 4 * n_dimensions - 1

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Prepare output DataFrame
    if 'id' in df.columns:
        submission = pd.DataFrame({'id': df['id'], 'label': labels})
    else:
        submission = pd.DataFrame({'id': range(len(labels)), 'label': labels})

    # Save result
    submission.to_csv(output_file, index=False)
    print(f"Saved clustering results to {output_file}")

if __name__ == "__main__":
    cluster_and_save('public_data.csv', 'b10603034_public.csv')
    cluster_and_save('private_data.csv', 'b10603034_private.csv')
