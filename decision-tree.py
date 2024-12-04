from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns



cleaned_file_path = "data/Cleaned_AirQualityUCI.csv"
cleaned_data = pd.read_csv(cleaned_file_path)

selected_features = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'T', 'RH', 'AH']

print("\nDataset Columns:")
print(cleaned_data.columns.tolist())
missing_features = [feature for feature in selected_features if feature not in cleaned_data.columns]
if missing_features:
    raise ValueError(f"The following features are missing from the dataset: {missing_features}")

stats = cleaned_data[selected_features].describe().round(1)
print(stats)

def define_air_quality(row):
    if all(row[feature] < stats.loc['50%', feature] for feature in selected_features):
        return "Good"
    elif any(stats.loc['50%', feature] <= row[feature] < stats.loc['75%', feature] for feature in selected_features):
        return "Moderate"
    elif any(row[feature] >= stats.loc['75%', feature] for feature in selected_features):
        return 'Poor'
    else:
        return "Good"

cleaned_data['Air_Quality_Level'] = cleaned_data.apply(define_air_quality, axis=1)

print("\nAir Quality Level Distribution:")
print(cleaned_data['Air_Quality_Level'].value_counts())

updated_file_path = "data/Updated_AirQualityUCI.csv"
cleaned_data.to_csv(updated_file_path, index=False)
print(f"\nUpdated dataset saved to {updated_file_path}")


X = cleaned_data[selected_features]
y = cleaned_data['Air_Quality_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree = DecisionTreeClassifier(max_depth=9, min_samples_split=10, min_samples_leaf=5, random_state=42, class_weight='balanced')
decision_tree.fit(X_train, y_train)


y_pred = decision_tree.predict(X_test)

print("\nDecision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(decision_tree, X_test, y_test, display_labels=['Good', 'Moderate', 'Poor'], cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

cv_scores = cross_val_score(decision_tree, X, y, cv=5) 
print("\nCross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean().round(2))


plt.figure(figsize=(20, 10))
plot_tree(decision_tree, feature_names=selected_features, class_names=['Good', 'Moderate', 'Poor'], filled=True, impurity=True, fontsize=8)
plt.title("Decision Tree Visualization")
plt.show()


max_depths = range(1, 21) 
training_accuracies = []
testing_accuracies = []

for depth in max_depths:
    decision_tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    decision_tree.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, decision_tree.predict(X_train))
    training_accuracies.append(train_acc)

    test_acc = accuracy_score(y_test, decision_tree.predict(X_test))
    testing_accuracies.append(test_acc)

plt.figure(figsize=(10, 6))
plt.plot(max_depths, training_accuracies, label='Training Accuracy', marker='o')
plt.plot(max_depths, testing_accuracies, label='Testing Accuracy', marker='s')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Training vs Testing Accuracy')
plt.legend()
plt.grid(True)
plt.show()


report = classification_report(y_test, y_pred, target_names=['Good', 'Moderate', 'Poor'], output_dict=True)

report_df = pd.DataFrame(report).transpose().round(2)

report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap="Blues", cbar_kws={"shrink": 0.8})
plt.title("Classification Report Heatmap (Decision Tree)")
plt.xlabel("Metrics")
plt.ylabel("Classes")
plt.tight_layout()
plt.show()