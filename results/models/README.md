# 🤖 Models Directory

This directory contains all trained machine learning models and preprocessing objects.

## 📁 Structure

```
models/
├── decision_tree.pkl         # Trained Decision Tree model
├── naive_bayes.pkl          # Trained Naive Bayes model
├── logistic_regression.pkl  # Trained Logistic Regression model
├── knn.pkl                  # Trained k-NN model
├── random_forest.pkl        # Trained Random Forest model (BEST)
├── scaler.pkl               # Feature scaler (StandardScaler)
└── label_encoder.npy        # Label encoder for class names
```

## 🎯 Model Performance

| Model | File Size | Accuracy | Best For |
|-------|-----------|----------|----------|
| **Random Forest** | **11MB** | **62.3%** | **Best overall performance** |
| **k-NN** | **160KB** | **61.2%** | **Simple, interpretable** |
| **Logistic Regression** | **4KB** | **58.3%** | **Fast, linear** |
| **Decision Tree** | **8KB** | **56.1%** | **Interpretable** |
| **Naive Bayes** | **4KB** | **12.1%** | **Probabilistic** |

## 🔧 Using the Models

### **Load and Use a Model**:
```python
import joblib
import numpy as np

# Load the best model
model = joblib.load('results/models/random_forest.pkl')

# Load the scaler
scaler = joblib.load('results/models/scaler.pkl')

# Load the label encoder
label_encoder = np.load('results/models/label_encoder.npy', allow_pickle=True)

# Prepare new data (example)
new_data = [[0.5, 0.3, 0.8, 0.2, 0.1, 0.9, 0.4, 0.6]]  # 8 features

# Scale the data
scaled_data = scaler.transform(new_data)

# Make prediction
prediction = model.predict(scaled_data)

# Convert back to class name
class_name = label_encoder[prediction[0]]
print(f"Predicted class: {class_name}")
```

### **Load All Models**:
```python
import joblib

models = {}
model_names = ['decision_tree', 'naive_bayes', 'logistic_regression', 'knn', 'random_forest']

for name in model_names:
    models[name] = joblib.load(f'results/models/{name}.pkl')

# Use any model
predictions = models['random_forest'].predict(scaled_data)
```

## 📊 Model Details

### **Random Forest** (`random_forest.pkl`) - **BEST MODEL**
- **Type**: Ensemble of 300 decision trees
- **Size**: 11MB (largest model)
- **Performance**: 62.3% accuracy, 85.4% ROC-AUC
- **Features**: Feature importance, robust to overfitting
- **Use Case**: Best overall performance, production deployment

### **k-NN** (`knn.pkl`)
- **Type**: k-Nearest Neighbors classifier
- **Size**: 160KB (stores training data)
- **Performance**: 61.2% accuracy, 83.9% ROC-AUC
- **Features**: Distance-based, interpretable
- **Use Case**: Simple classification, interpretable results

### **Logistic Regression** (`logistic_regression.pkl`)
- **Type**: Linear classifier with regularization
- **Size**: 4KB (just coefficients)
- **Performance**: 58.3% accuracy, 83.3% ROC-AUC
- **Features**: Fast, linear decision boundary
- **Use Case**: Fast predictions, linear relationships

### **Decision Tree** (`decision_tree.pkl`)
- **Type**: Single decision tree
- **Size**: 8KB (tree structure)
- **Performance**: 56.1% accuracy, 76.7% ROC-AUC
- **Features**: Interpretable, feature importance
- **Use Case**: Understanding decision rules, feature analysis

### **Naive Bayes** (`naive_bayes.pkl`)
- **Type**: Probabilistic classifier
- **Size**: 4KB (statistics only)
- **Performance**: 12.1% accuracy, 73.2% ROC-AUC
- **Features**: Fast, probabilistic outputs
- **Use Case**: Baseline model, probability estimates

## 🔧 Preprocessing Objects

### **Scaler** (`scaler.pkl`)
- **Type**: StandardScaler (mean=0, std=1)
- **Purpose**: Feature standardization
- **Required for**: k-NN, Logistic Regression
- **Usage**: `scaler.transform(new_data)`

### **Label Encoder** (`label_encoder.npy`)
- **Type**: NumPy array of class names
- **Purpose**: Convert numeric predictions to class names
- **Classes**: ['MIT', 'NUC', 'CYT', 'ME1', 'EXC', 'ME2', 'ME3', 'VAC', 'POX', 'ERL']
- **Usage**: `label_encoder[prediction]`

## 🚀 Production Deployment

### **Recommended Model**: Random Forest
```python
# Production-ready prediction function
def predict_protein_localization(features):
    """
    Predict protein localization from 8 features.
    
    Args:
        features: List of 8 numerical features [mcg, gvh, alm, mit, erl, pox, vac, nuc]
    
    Returns:
        str: Predicted localization class
    """
    import joblib
    import numpy as np
    
    # Load model and preprocessing objects
    model = joblib.load('results/models/random_forest.pkl')
    scaler = joblib.load('results/models/scaler.pkl')
    label_encoder = np.load('results/models/label_encoder.npy', allow_pickle=True)
    
    # Preprocess and predict
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    
    return label_encoder[prediction]
```

## 📝 Notes

- **File Formats**: Models saved as pickle files (.pkl)
- **Compatibility**: Requires same scikit-learn version
- **Reproducibility**: All models use random_state=42
- **Size**: Random Forest is largest due to 300 trees
- **Loading**: Use `joblib.load()` for models, `np.load()` for encoder
- **Scaling**: Always scale features before prediction
- **Classes**: 10 possible protein localization sites

