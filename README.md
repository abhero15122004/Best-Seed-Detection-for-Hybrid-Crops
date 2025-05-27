# Best-Seed-Detection-for-Hybrid-Crops
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

plt.style.use('ggplot')
sns.set_theme()

def load_and_preprocess(file_path):
    # Load the datasets
    df = pd.read_csv(file_path)

    # Handle missing data (drop or fill)
    df.fillna(method='ffill', inplace=True)

    return df

# Load Capsicum and Chilli datasets
capsicum_df = load_and_preprocess('/content/capsicum_hybrid_species_dataset.csv')
chilli_df = load_and_preprocess('/content/chilli_hybrid_species_dataset.csv')

scaler = MinMaxScaler()

def normalize_features(df, features, scaler):
    df_num = df.copy()

    # Handle categorical features
    for feature in features:
        if feature in df_num.columns and df_num[feature].dtype == 'object':
            df_num[feature] = df_num[feature].map({'Low': 1, 'Medium': 2, 'High': 3})

    # Normalize numerical features
    df_num[features] = scaler.fit_transform(df_num[features].astype(float))

    return df_num

# Important features for hybrid crop prediction
important_features = ['HeatLevel (SHU)', 'Yield (kg/plant)', 'DiseaseResistance', 'DroughtTolerance']

capsicum_normalized = normalize_features(capsicum_df, important_features, scaler)
chilli_normalized = normalize_features(chilli_df, important_features, scaler)

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(capsicum_normalized[important_features].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation in Capsicum Dataset")
plt.show()

# Pairplot for Capsicum
sns.pairplot(capsicum_normalized[important_features])
plt.show()

# Distribution of Hybrid Scores
def plot_hybrid_score_distribution(df_normalized, title):
    df_normalized['HybridScore'] = df_normalized[important_features].sum(axis=1)
    plt.figure(figsize=(10, 6))
    sns.histplot(df_normalized['HybridScore'], bins=20, kde=True)
    plt.title(f'Hybrid Score Distribution: {title}')
    plt.xlabel('Hybrid Score')
    plt.ylabel('Frequency')
    plt.show()

plot_hybrid_score_distribution(capsicum_normalized, "Capsicum")
plot_hybrid_score_distribution(chilli_normalized, "Chilli")

# Find the best species based on HybridScore
def calculate_hybrid_scores(df_normalized, important_features):
    df_normalized['HybridScore'] = df_normalized[important_features].sum(axis=1)
    return df_normalized

capsicum_normalized = calculate_hybrid_scores(capsicum_normalized, important_features)
chilli_normalized = calculate_hybrid_scores(chilli_normalized, important_features)

def find_best_species(df_normalized):
    best_species = df_normalized.loc[df_normalized['HybridScore'].idxmax()]
    return best_species

best_capsicum_species = find_best_species(capsicum_normalized)
best_chilli_species = find_best_species(chilli_normalized)

print(f"Best Capsicum species: {best_capsicum_species['SpeciesID']} with score: {best_capsicum_species['HybridScore']}")
print(f"Best Chilli species: {best_chilli_species['SpeciesID']} with score: {best_chilli_species['HybridScore']}")

# Plot species details
def plot_species_details(df, species_id):
    species_details = df[df['SpeciesID'] == species_id]
    print(species_details)

plot_species_details(capsicum_df, best_capsicum_species['SpeciesID'])
plot_species_details(chilli_df, best_chilli_species['SpeciesID'])

# Clustering species
def cluster_species(df_normalized, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_normalized['Cluster'] = kmeans.fit_predict(df_normalized[important_features])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_normalized, x='HeatLevel (SHU)', y='Yield (kg/plant)', hue='Cluster', palette='viridis')
    plt.title('Species Clustering')
    plt.show()

cluster_species(capsicum_normalized)
cluster_species(chilli_normalized)

# PCA plot for hybrid crops
def plot_pca(df_normalized, important_features):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_normalized[important_features])

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=df_normalized['HybridScore'], palette='coolwarm', size=df_normalized['HybridScore'])
    plt.title('PCA Plot for Species')
    plt.show()

plot_pca(capsicum_normalized, important_features)
plot_pca(chilli_normalized, important_features)

# Deep Learning model for predicting individual hybrid parameters
def build_dl_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(len(important_features), activation='linear')  # Predicts individual parameters
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Prepare combined data from Capsicum and Chilli for hybrid prediction
combined_df = pd.concat([capsicum_normalized[important_features], chilli_normalized[important_features]])

X = combined_df
y = combined_df[important_features]

# Split data (could add train/test split here if needed)
input_dim = X.shape[1]
model = build_dl_model(input_dim)

# Train the DL model
model.fit(X, y, epochs=100, batch_size=32, verbose=1)

# Predict individual parameters for the best hybrid crop
predicted_parameters = model.predict(X)
best_hybrid_parameters = pd.DataFrame(predicted_parameters, columns=important_features)

# Show predictions for individual hybrid crop features
print("Predicted parameters for the best hybrid crop:")
print(best_hybrid_parameters.iloc[best_capsicum_species.name])

# Generate a report or export the results to a CSV
best_hybrid_report = pd.DataFrame([best_capsicum_species, best_chilli_species])
best_hybrid_report.to_csv('best_hybrid_species_report.csv', index=False)
