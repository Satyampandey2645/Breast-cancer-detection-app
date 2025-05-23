# Import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Load the Kaggle dataset (replace with your file path)
df = pd.read_csv('Cancer_Data.csv')  # Kaggle filename is usually 'data.csv'

# Data Cleaning
print("\n=== Initial Data Exploration ===")
print("Original shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# Drop unnecessary columns
df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')

# Convert diagnosis to binary (M=1, B=0)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])  # 1=Malignant, 0=Benign

# Visualize the target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosis', data=df)
plt.title('Diagnosis Distribution (0 = Benign, 1 = Malignant)')
plt.show()

# Prepare features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Evaluation Function
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
    
    print(f"\n=== {model_name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# 1. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
evaluate_model(rf, X_test_scaled, y_test, "Random Forest")

# 2. Support Vector Machine
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
evaluate_model(svm, X_test_scaled, y_test, "SVM")

# 3. Neural Network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n=== Neural Network Training ===")
history = model.fit(X_train_scaled, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

# Evaluate NN
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nNeural Network Test Accuracy: {test_acc:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.legend()
plt.show()

# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Top 10 Important Features (Random Forest)")
plt.bar(range(10), importances[indices][:10], align="center")
plt.xticks(range(10), X.columns[indices][:10], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Save Models
joblib.dump(rf, 'breast_cancer_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModels saved to disk: 'breast_cancer_rf_model.pkl' and 'scaler.pkl'")

# Prediction Function
def predict_breast_cancer(feature_dict):
    """
    Input: Dictionary of features (keys = column names)
    Output: Prediction (0/1), Probability, and Interpretation
    """
    model = joblib.load('breast_cancer_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Convert dict to dataframe row
    features = pd.DataFrame([feature_dict])
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Predict
    pred = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0][1]
    
    diagnosis = "Malignant" if pred == 1 else "Benign"
    confidence = prob if pred == 1 else (1 - prob)
    
    return {
        'prediction': pred,
        'probability': float(confidence),
        'diagnosis': diagnosis,
        'confidence': f"{confidence*100:.1f}%"
    }

# Example Usage
if __name__ == "__main__":
    # Create example input (using mean values from dataset)
    example_input = {
        'radius_mean': 17.99,
        'texture_mean': 10.38,
        'perimeter_mean': 122.8,
        'area_mean': 1001.0,
        'smoothness_mean': 0.1184,
        'compactness_mean': 0.2776,
        'concavity_mean': 0.3001,
        'concave points_mean': 0.1471,
        'symmetry_mean': 0.2419,
        'fractal_dimension_mean': 0.07871
        # Add all 30 features in production
    }
    
    result = predict_breast_cancer(example_input)
    print("\nExample Prediction Result:")
    for k, v in result.items():
        print(f"{k:>12}: {v}")