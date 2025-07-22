import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the penguins dataset from seaborn"""
    print("Loading penguins dataset...")
    df = sns.load_dataset('penguins')
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    return df

def preprocess_data(df):
    """
    Preprocess the data:
    - Handle missing values
    - Apply one-hot encoding to categorical features
    - Apply label encoding to target variable
    """
    print("\nPreprocessing data...")
    
    # Handle missing values by dropping rows with any missing values
    df_clean = df.dropna()
    print(f"Dataset shape after removing missing values: {df_clean.shape}")
    
    # Separate features and target
    X = df_clean.drop('species', axis=1)
    y = df_clean['species']
    
    # Apply one-hot encoding to categorical features (sex, island)
    categorical_features = ['sex', 'island']
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    print(f"Features after one-hot encoding: {list(X_encoded.columns)}")
    
    # Apply label encoding to target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Target classes: {label_encoder.classes_}")
    print(f"Target distribution: {np.bincount(y_encoded)}")
    
    return X_encoded, y_encoded, label_encoder

def split_data(X, y):
    """Split data into training and test sets with stratification"""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training target distribution: {np.bincount(y_train)}")
    print(f"Test target distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train XGBoost classifier with parameters to prevent overfitting"""
    print("\nTraining XGBoost model...")
    
    # Initialize XGBoost classifier with overfitting prevention parameters
    xgb_classifier = xgb.XGBClassifier(
        max_depth=3,           # Limit tree depth to prevent overfitting
        n_estimators=100,      # Number of boosting rounds
        learning_rate=0.1,     # Step size shrinkage
        subsample=0.8,         # Fraction of samples for each tree
        colsample_bytree=0.8,  # Fraction of features for each tree
        random_state=42,
        eval_metric='mlogloss'  # Multiclass log loss
    )
    
    # Train the model
    xgb_classifier.fit(X_train, y_train)
    
    print("Model training completed!")
    return xgb_classifier

def evaluate_model(model, X_train, X_test, y_train, y_test, label_encoder):
    """Evaluate the model on both training and test sets"""
    print("\nEvaluating model...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
    train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
    
    # Calculate metrics for test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    
    print("="*50)
    print("TRAINING SET PERFORMANCE:")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"F1-Score (Macro): {train_f1_macro:.4f}")
    print(f"F1-Score (Weighted): {train_f1_weighted:.4f}")
    
    print("\nTEST SET PERFORMANCE:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1-Score (Macro): {test_f1_macro:.4f}")
    print(f"F1-Score (Weighted): {test_f1_weighted:.4f}")
    print("="*50)
    
    # Detailed classification report for test set
    print("\nDETAILED CLASSIFICATION REPORT (Test Set):")
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    
    # Confusion matrix
    print("CONFUSION MATRIX (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)
    
    # Feature importance
    print("\nFEATURE IMPORTANCE:")
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df)
    
    return {
        'train_accuracy': train_accuracy,
        'train_f1_macro': train_f1_macro,
        'train_f1_weighted': train_f1_weighted,
        'test_accuracy': test_accuracy,
        'test_f1_macro': test_f1_macro,
        'test_f1_weighted': test_f1_weighted
    }

def save_model(model, label_encoder, metrics, feature_names):
    """Save the trained model and associated metadata"""
    print("\nSaving model...")
    
    # Create app/data directory if it doesn't exist
    os.makedirs('app/data', exist_ok=True)
    
    # Save the XGBoost model
    model_path = 'app/data/model.json'
    model.save_model(model_path)
    
    # Save label encoder and metadata
    metadata = {
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'feature_names': feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
        'model_metrics': metrics,
        'model_params': model.get_params()
    }
    
    # Save metadata as JSON
    import json
    metadata_path = 'app/data/model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

def main():
    """Main training pipeline"""
    print("Starting penguin species classification training pipeline...")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Preprocess data
    X, y, label_encoder = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, label_encoder)
    
    # Save model
    save_model(model, label_encoder, metrics, X.columns)
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    
    # Check for overfitting
    train_test_diff = metrics['train_accuracy'] - metrics['test_accuracy']
    if train_test_diff > 0.1:
        print(f"\nWARNING: Potential overfitting detected!")
        print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Difference: {train_test_diff:.4f}")
    else:
        print(f"\nGood generalization! Train-test accuracy difference: {train_test_diff:.4f}")

if __name__ == "__main__":
    main()