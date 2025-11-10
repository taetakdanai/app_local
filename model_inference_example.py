"""
Model Loading and Inference Example
====================================

This script demonstrates how to load and use the exported battery RUL prediction models.

Author: Battery RUL Prediction Team
Date: November 10, 2025
"""

import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path


class BatteryRULPredictor:
    """
    Battery RUL Prediction wrapper for easy model loading and inference.
    """
    
    def __init__(self, model_type='zenodo'):
        """
        Initialize the predictor.
        
        Parameters:
        -----------
        model_type : str
            Either 'zenodo' or 'nasa' to specify which model to load
        """
        self.model_type = model_type.lower()
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.feature_info = None
        self.models_dir = Path(__file__).parent / 'models'
        
        self._load_model()
    
    def _load_model(self):
        """Load the model, preprocessor, and metadata."""
        print(f"Loading {self.model_type.upper()} model...")
        
        try:
            if self.model_type == 'zenodo':
                # Load Zenodo model
                self.model = joblib.load(self.models_dir / 'zenodo_best_model_latest.pkl')
                self.preprocessor = joblib.load(self.models_dir / 'zenodo_preprocessor_latest.pkl')
                
                with open(self.models_dir / 'zenodo_model_metadata_latest.json', 'r') as f:
                    self.metadata = json.load(f)
                
                with open(self.models_dir / 'zenodo_feature_info_latest.json', 'r') as f:
                    self.feature_info = json.load(f)
                    
            elif self.model_type == 'nasa':
                # Load NASA fine-tuned model
                self.model = joblib.load(self.models_dir / 'nasa_finetuned_model_latest.pkl')
                # NASA model includes preprocessing in pipeline
                
                with open(self.models_dir / 'nasa_model_metadata_latest.json', 'r') as f:
                    self.metadata = json.load(f)
                
                with open(self.models_dir / 'nasa_feature_info_latest.json', 'r') as f:
                    self.feature_info = json.load(f)
            else:
                raise ValueError("model_type must be 'zenodo' or 'nasa'")
            
            print(f"✓ Model loaded successfully!")
            self._print_model_info()
            
        except FileNotFoundError as e:
            print(f"✗ Error: Model files not found. Please run the training notebooks first.")
            print(f"  Missing file: {e}")
            raise
    
    def _print_model_info(self):
        """Print model information."""
        print(f"\n{'='*60}")
        print(f"Model: {self.metadata['model_name']}")
        print(f"Type: {self.metadata['model_type']}")
        print(f"Dataset: {self.metadata['dataset']}")
        
        if self.model_type == 'zenodo':
            metrics = self.metadata['metrics']
            print(f"\nPerformance Metrics:")
            print(f"  R² Score: {metrics['r2_score']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  MAE: {metrics['mae']:.2f}")
        else:
            print(f"\nPerformance on NASA Test Set:")
            nasa_metrics = self.metadata['metrics']['nasa_test']
            print(f"  R² Score: {nasa_metrics['r2_score']:.4f}")
            print(f"  RMSE: {nasa_metrics['rmse']:.2f}")
            print(f"  MAE: {nasa_metrics['mae']:.2f}")
            
            print(f"\nPerformance on Zenodo Test Set:")
            zenodo_metrics = self.metadata['metrics']['zenodo_test']
            print(f"  R² Score: {zenodo_metrics['r2_score']:.4f}")
            print(f"  RMSE: {zenodo_metrics['rmse']:.2f}")
        
        print(f"\nRequired Features: {self.get_required_features_count()}")
        print(f"Confidence Level: {self.metadata['deployment_notes']['confidence_level']}")
        print(f"{'='*60}\n")
    
    def get_required_features(self):
        """Get list of required features."""
        if self.model_type == 'zenodo':
            return self.feature_info['feature_names']
        else:
            return self.feature_info['common_features']
    
    def get_required_features_count(self):
        """Get number of required features."""
        return len(self.get_required_features())
    
    def predict(self, X):
        """
        Predict RUL for input data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features. Must contain all required features.
        
        Returns:
        --------
        np.ndarray
            Predicted RUL values
        """
        # Validate input
        required_features = self.get_required_features()
        missing_features = set(required_features) - set(X.columns)
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only required features in correct order
        X_subset = X[required_features].copy()
        
        # Predict
        if self.model_type == 'zenodo':
            # Zenodo model requires separate preprocessing
            X_processed = self.preprocessor.transform(X_subset)
            predictions = self.model.predict(X_processed)
        else:
            # NASA model includes preprocessing
            predictions = self.model.predict(X_subset)
        
        return predictions
    
    def predict_with_uncertainty(self, X, return_std=True):
        """
        Predict RUL with uncertainty estimates (for RandomForest models).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        return_std : bool
            Whether to return standard deviation
        
        Returns:
        --------
        predictions : np.ndarray
            Mean predictions
        std : np.ndarray (optional)
            Standard deviation of predictions across trees
        """
        required_features = self.get_required_features()
        X_subset = X[required_features].copy()
        
        if self.model_type == 'zenodo':
            X_processed = self.preprocessor.transform(X_subset)
            
            # Get predictions from all trees
            if hasattr(self.model, 'estimators_'):
                tree_predictions = np.array([tree.predict(X_processed) 
                                            for tree in self.model.estimators_])
                predictions = tree_predictions.mean(axis=0)
                
                if return_std:
                    std = tree_predictions.std(axis=0)
                    return predictions, std
                return predictions
            else:
                return self.model.predict(X_processed)
        else:
            # For NASA model with pipeline
            if hasattr(self.model.named_steps['rf'], 'estimators_'):
                X_processed = self.model.named_steps['preprocess'].transform(X_subset)
                tree_predictions = np.array([tree.predict(X_processed) 
                                            for tree in self.model.named_steps['rf'].estimators_])
                predictions = tree_predictions.mean(axis=0)
                
                if return_std:
                    std = tree_predictions.std(axis=0)
                    return predictions, std
                return predictions
            else:
                return self.model.predict(X_subset)
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance from the model.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
        
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        if self.model_type == 'zenodo':
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = self.get_required_features()
            else:
                print("Model does not support feature importances")
                return None
        else:
            # NASA model with pipeline
            if hasattr(self.model.named_steps['rf'], 'feature_importances_'):
                importances = self.model.named_steps['rf'].feature_importances_
                feature_names = self.get_required_features()
            else:
                print("Model does not support feature importances")
                return None
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(importances)],
            'Importance': importances
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        return importance_df


def example_usage_zenodo():
    """Example: Using Zenodo model."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Using Zenodo Model")
    print("="*60)
    
    # Initialize predictor
    predictor = BatteryRULPredictor(model_type='zenodo')
    
    # Example: Create sample data (replace with your actual data)
    required_features = predictor.get_required_features()
    n_samples = 5
    
    # Create random sample data (for demonstration only)
    X_sample = pd.DataFrame(
        np.random.randn(n_samples, len(required_features)),
        columns=required_features
    )
    
    print(f"Sample data shape: {X_sample.shape}")
    
    # Make predictions
    predictions = predictor.predict(X_sample)
    
    print(f"\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: RUL = {pred:.2f}")
    
    # Get predictions with uncertainty
    if hasattr(predictor.model, 'estimators_'):
        predictions, std = predictor.predict_with_uncertainty(X_sample)
        print(f"\nPredictions with Uncertainty:")
        for i, (pred, uncertainty) in enumerate(zip(predictions, std)):
            print(f"  Sample {i+1}: RUL = {pred:.2f} ± {uncertainty:.2f}")
    
    # Get feature importance
    importance_df = predictor.get_feature_importance(top_n=5)
    if importance_df is not None:
        print(f"\nTop 5 Important Features:")
        print(importance_df.to_string(index=False))


def example_usage_nasa():
    """Example: Using NASA fine-tuned model."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Using NASA Fine-tuned Model")
    print("="*60)
    
    # Initialize predictor
    predictor = BatteryRULPredictor(model_type='nasa')
    
    # Example: Create sample data
    required_features = predictor.get_required_features()
    n_samples = 3
    
    # Create random sample data (for demonstration only)
    X_sample = pd.DataFrame(
        np.random.randn(n_samples, len(required_features)),
        columns=required_features
    )
    
    print(f"Sample data shape: {X_sample.shape}")
    
    # Make predictions
    predictions = predictor.predict(X_sample)
    
    print(f"\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"  Battery {i+1}: RUL = {pred:.2f}")
    
    # Get feature importance
    importance_df = predictor.get_feature_importance(top_n=5)
    if importance_df is not None:
        print(f"\nTop 5 Important Features:")
        print(importance_df.to_string(index=False))


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Battery RUL Prediction - Model Inference Examples")
    print("="*60)
    
    # Example 1: Zenodo model
    try:
        example_usage_zenodo()
    except Exception as e:
        print(f"\n✗ Could not run Zenodo example: {e}")
    
    # Example 2: NASA model
    try:
        example_usage_nasa()
    except Exception as e:
        print(f"\n✗ Could not run NASA example: {e}")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nNote: Replace sample data with your actual battery measurements.")
    print("Refer to models/README.md for detailed usage instructions.\n")
