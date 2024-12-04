import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the dataset
cleaned_file_path = "Updated_AirQualityUCI.csv"
cleaned_data = pd.read_csv(cleaned_file_path)

# Define features and target
selected_features = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']
X = cleaned_data[selected_features]
y = cleaned_data['Air_Quality_Level']

# # Handle skewness with Power Transformation
# power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
# X_transformed = power_transformer.fit_transform(X)

# Handle skewness with Log Transformation
for feature in selected_features:
    if (X[feature] <= 0).any():
        X[feature] = X[feature] - X[feature].min() + 1
        X[feature] = np.log(X[feature])
X_transformed = np.log1p(X)

# Handle outliers using clipping (Winsorization)
X_clipped = np.clip(X_transformed, np.percentile(X_transformed, 1, axis=0), np.percentile(X_transformed, 99, axis=0))

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clipped)

# Perform dimensionality reduction with PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
X_reduced = pca.fit_transform(X_scaled)
print(f"Reduced Dimensions: {X_reduced.shape[1]} components retained.")

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_reduced, y)
print("\nClass Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train and evaluate KNN with different values of k
k_values = range(1, 12)
training_accuracies = []
testing_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knn.fit(X_train, y_train)

    # Compute training accuracy
    train_acc = accuracy_score(y_train, knn.predict(X_train))
    training_accuracies.append(train_acc)

    # Compute testing accuracy
    test_acc = accuracy_score(y_test, knn.predict(X_test))
    testing_accuracies.append(test_acc)

# Plot training vs testing accuracy for different k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, training_accuracies, label='Training Accuracy', marker='o')
plt.plot(k_values, testing_accuracies, label='Testing Accuracy', marker='s')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN: Training vs Testing Accuracy (Comprehensive Preprocessing)')
plt.legend()
plt.grid(True)
plt.show()

# Train the optimal KNN model
# optimal_k = k_values[testing_accuracies.index(max(testing_accuracies))]
optimal_k = 5
print(f"\nOptimal Number of Neighbors (k): {optimal_k}")

knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = knn.predict(X_test)
print("\nKNN Performance After Comprehensive Preprocessing:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test, display_labels=['Good', 'Moderate', 'Poor'], cmap='Blues')
plt.title("KNN Confusion Matrix After Comprehensive Preprocessing")
plt.show()

# Cross-Validation
cv_scores = cross_val_score(knn, X_resampled, y_resampled, cv=5)
print("\nCross-Validation Scores:", cv_scores)
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.2f}")

# Generate the classification report as a dictionary
report = classification_report(y_test, y_pred, target_names=['Good', 'Moderate', 'Poor'], output_dict=True)

# Convert the dictionary to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Remove 'accuracy', 'macro avg', and 'weighted avg' entries if present
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

# Round the values for better display
report_df = report_df.round(2)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='Blues', fmt='g')
plt.title('Classification Report Heatmap')
plt.ylabel('Classes')
plt.xlabel('Metrics')
plt.show()

# Perform PCA to reduce features to 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)  # Use standardized data

# Create a scatter plot
plt.figure(figsize=(8, 6))
for label in sorted(y.unique()):
    plt.scatter(
        X_2d[y == label, 0],
        X_2d[y == label, 1],
        label=f"Class {label}",
        alpha=0.6
    )

# Customize the plot
plt.title("2D Projection of Air Quality Levels")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.legend(title="Air Quality Level")
plt.grid(True)
plt.show()























# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler, PowerTransformer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Load the dataset
# cleaned_file_path = "Updated_AirQualityUCI.csv"  # Dataset with Air_Quality_Level
# cleaned_data = pd.read_csv(cleaned_file_path)

# # Define features and target
# selected_features = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']
# X = cleaned_data[selected_features]
# y = cleaned_data['Air_Quality_Level']


# # Apply Yeo- Johnson Transformation
# power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
# X_transformed = power_transformer.fit_transform(X)


# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_transformed)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Train and evaluate KNN with different values of k
# k_values = range(1, 21)
# training_accuracies = []
# testing_accuracies = []

# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)

#     # Compute training accuracy
#     train_acc = accuracy_score(y_train, knn.predict(X_train))
#     training_accuracies.append(train_acc)

#     # Compute testing accuracy
#     test_acc = accuracy_score(y_test, knn.predict(X_test))
#     testing_accuracies.append(test_acc)

# # Plot training vs testing accuracy for different k values
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, training_accuracies, label='Training Accuracy', marker='o')
# plt.plot(k_values, testing_accuracies, label='Testing Accuracy', marker='s')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Accuracy')
# plt.title('KNN: Training vs Testing Accuracy (after skewness correction)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Train the optimal KNN model (choose k where testing accuracy peaks)
# optimal_k = k_values[testing_accuracies.index(max(testing_accuracies))]
# print(f"\nOptimal Number of Neighbors (k): {optimal_k}")

# knn = KNeighborsClassifier(n_neighbors=optimal_k)
# knn.fit(X_train, y_train)

# #  Make predictions and evaluate
# y_pred = knn.predict(X_test)
# print("\nKNN Performance:")
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test, display_labels=['Good', 'Moderate', 'Poor'], cmap='Blues')
# plt.title("KNN Confusion Matrix")
# plt.show()
