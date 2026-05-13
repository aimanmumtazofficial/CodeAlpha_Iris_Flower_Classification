# ================================================
#   Iris Flower Classification
#   CodeAlpha Data Science Internship - Task 1
#   Student: Aiman | ID: CA/DF1/54987
# ================================================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("   Iris Flower Classification - CodeAlpha Task 1")
print("=" * 55)
print("Step 1: All libraries imported successfully.")

# ------------------------------------------------
# Step 2: Load the Dataset
# ------------------------------------------------
df = pd.read_csv('Iris.csv')

print("\nStep 2: Dataset loaded successfully.")
print(f"   Total Rows   : {df.shape[0]}")
print(f"   Total Columns: {df.shape[1]}")
print(f"   Columns      : {list(df.columns)}")

# ------------------------------------------------
# Step 3: Explore the Dataset
# ------------------------------------------------
print("\n" + "=" * 55)
print("Step 3: Dataset Exploration")
print("=" * 55)

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

print("\nSpecies Count:")
print(df['Species'].value_counts())

print("\nChecking for Missing Values:")
print(df.isnull().sum())
if df.isnull().sum().sum() == 0:
    print("No missing values found in the dataset.")

# ------------------------------------------------
# Step 4: Visualization - Species Distribution
# ------------------------------------------------
print("\nStep 4: Generating visualizations...")

colors = ['#FF6B9D', '#C78DF5', '#85D4E3']

plt.figure(figsize=(8, 5))
df['Species'].value_counts().plot(kind='bar', color=colors, edgecolor='black')
plt.title('Iris Species Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Species', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('graph1_species_count.png', dpi=150)
plt.show()
plt.close()
print("   Graph 1 saved: graph1_species_count.png")

# ------------------------------------------------
# Step 5: Visualization - Box Plot (Before Handling Outliers)
# ------------------------------------------------
features     = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
species_list = df['Species'].unique()

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Feature Distribution by Species (Before Handling Outliers)',
             fontsize=14, fontweight='bold')

for i, feature in enumerate(features):
    ax = axes[i // 2, i % 2]
    data_to_plot = [df[df['Species'] == sp][feature].values for sp in species_list]
    bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.4)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(feature, fontweight='bold', fontsize=11, pad=10)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Setosa', 'Versicolor', 'Virginica'], fontsize=10, rotation=0)
    ax.set_ylabel('cm', fontsize=10)

plt.subplots_adjust(hspace=0.4, wspace=0.35)
plt.savefig('graph2_boxplot_before_outliers.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 2 saved: graph2_boxplot_before_outliers.png")

# ------------------------------------------------
# Step 6: Handle Outliers using IQR Method (per Species)
# ------------------------------------------------
print("\n" + "=" * 55)
print("Step 6: Handling Outliers using IQR Method")
print("=" * 55)

print(f"Rows before removing outliers: {len(df)}")

# IQR is applied per species so each species is cleaned independently
clean_frames = []
for species in df['Species'].unique():
    subset = df[df['Species'] == species].copy()
    for feature in features:
        Q1  = subset[feature].quantile(0.25)
        Q3  = subset[feature].quantile(0.75)
        IQR = Q3 - Q1
        lb  = Q1 - 1.5 * IQR
        ub  = Q3 + 1.5 * IQR
        subset = subset[(subset[feature] >= lb) & (subset[feature] <= ub)]
    clean_frames.append(subset)

df = pd.concat(clean_frames).reset_index(drop=True)

print(f"Rows after removing outliers : {len(df)}")
print(f"Outliers removed             : {150 - len(df)}")
print("Outliers handled successfully.")

# ------------------------------------------------
# Step 7: Visualization - Correlation Heatmap
# ------------------------------------------------
plt.figure(figsize=(8, 6))
numeric_df = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdPu',
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('graph3_heatmap.png', dpi=150)
plt.show()
plt.close()
print("   Graph 3 saved: graph3_heatmap.png")

# ------------------------------------------------
# Step 8: Visualization - Pairplot
# ------------------------------------------------
plot_df = df[['SepalLengthCm', 'SepalWidthCm',
              'PetalLengthCm', 'PetalWidthCm', 'Species']]
colors_map = {
    'Iris-setosa'     : '#FF6B9D',
    'Iris-versicolor' : '#C78DF5',
    'Iris-virginica'  : '#85D4E3'
}
g = sns.pairplot(plot_df, hue='Species', palette=colors_map,
                 diag_kind='hist', height=2.3)
g.fig.suptitle('Pairplot - All Features by Species',
               y=1.02, fontsize=13, fontweight='bold')
plt.savefig('graph4_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Graph 4 saved: graph4_pairplot.png")

# ------------------------------------------------
# Step 9: Prepare Data for the Model
# ------------------------------------------------
print("\n" + "=" * 55)
print("Step 9: Data Preparation")
print("=" * 55)

# Separate features and target variable
# We drop 'Id' column since it is not a useful feature for classification
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Encode species names into numeric labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Species label encoding:")
for i, name in enumerate(le.classes_):
    print(f"   {name} -> {i}")

# Split dataset: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nData split complete.")
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing samples  : {X_test.shape[0]}")

# ------------------------------------------------
# Step 10: Train the Model
# ------------------------------------------------
print("\n" + "=" * 55)
print("Step 10: Model Training")
print("=" * 55)

# Random Forest is a good choice here because it handles
# multi-class classification well and is resistant to overfitting
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

print("Random Forest model trained successfully.")
print(f"   Number of trees: {model.n_estimators}")

# ------------------------------------------------
# Step 11: Evaluate the Model
# ------------------------------------------------
print("\n" + "=" * 55)
print("Step 11: Model Evaluation")
print("=" * 55)

y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ------------------------------------------------
# Step 12: Confusion Matrix
# ------------------------------------------------
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Species', fontsize=12)
plt.ylabel('Actual Species', fontsize=12)
plt.xticks(rotation=15)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('graph5_confusion_matrix.png', dpi=150)
plt.show()
plt.close()
print("Graph 5 saved: graph5_confusion_matrix.png")

# ------------------------------------------------
# Step 13: Feature Importance
# ------------------------------------------------
importances   = model.feature_importances_
feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

plt.figure(figsize=(8, 5))
bar_colors = ['#85D4E3', '#85D4E3', '#FF6B9D', '#FF6B9D']
bars = plt.barh(feature_names, importances, color=bar_colors, edgecolor='black')
for bar, imp in zip(bars, importances):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
             f'{imp:.3f}', va='center', fontweight='bold')
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('graph6_feature_importance.png', dpi=150)
plt.show()
plt.close()

print("\nFeature Importance Scores:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"   {name:20s}: {imp:.4f}")

# ------------------------------------------------
# Step 14: Predict a New Flower
# ------------------------------------------------
print("\n" + "=" * 55)
print("Step 14: Predicting a New Flower")
print("=" * 55)

# Change these measurement values to test with different flowers
new_flower = pd.DataFrame({
    'SepalLengthCm': [5.1],
    'SepalWidthCm' : [3.5],
    'PetalLengthCm': [1.4],
    'PetalWidthCm' : [0.2]
})

prediction     = model.predict(new_flower)
predicted_name = le.inverse_transform(prediction)[0]
probabilities  = model.predict_proba(new_flower)[0]

print("Input Measurements:")
print(f"   Sepal Length : {new_flower['SepalLengthCm'][0]} cm")
print(f"   Sepal Width  : {new_flower['SepalWidthCm'][0]} cm")
print(f"   Petal Length : {new_flower['PetalLengthCm'][0]} cm")
print(f"   Petal Width  : {new_flower['PetalWidthCm'][0]} cm")
print(f"\nPredicted Species: {predicted_name}")
print("\nPrediction Confidence:")
for species, prob in zip(le.classes_, probabilities):
    bar = '█' * int(prob * 30)
    print(f"   {species:22s}: {bar} {prob * 100:.1f}%")

# ------------------------------------------------
# Final Summary
# ------------------------------------------------
print("\n" + "=" * 55)
print("   Project Complete!")
print("=" * 55)
print(f"Dataset        : Iris.csv (150 rows, 6 columns)")
print(f"Outliers       : Handled using IQR Method (per species)")
print(f"Model          : Random Forest Classifier")
print(f"Accuracy       : {accuracy * 100:.2f}%")
print(f"Graphs saved   : 6 PNG files")
print("=" * 55)