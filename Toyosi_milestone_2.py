from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Breast Cancer dataset from scikit-learn
data = load_breast_cancer()
features = data.data  # Features (independent variables)
labels = data.target  # Labels (target variable)

# Optional: Convert the dataset to a pandas DataFrame for better readability
df = pd.DataFrame(data=features, columns=data.feature_names)

# Standardize the features to have a mean of 0 and variance of 1 (important for PCA)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA to reduce dimensions to 2 principal components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

# Create a DataFrame for the two principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Label'] = labels  # Optionally add labels for visualization

# Plot the two principal components with a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Label'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2 Component PCA')
plt.show()

# Split the PCA-transformed data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(principal_components, labels, test_size=0.3, random_state=42)

# Initialize and train a logistic regression model using the training set
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate and print the accuracy of the logistic regression model
accuracy = accuracy_score(y_test, predictions)
print(f'Logistic Regression Accuracy: {accuracy:.2f}')
