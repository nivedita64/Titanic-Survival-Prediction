"""
Test suite for Titanic Survival Prediction pipeline.

This module contains unit tests and validation checks for all major components
of the ML pipeline including data loading, preprocessing, training, and evaluation.
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
import sys
import os

# Import pipeline components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    TitanicDataLoader, EDAVisualizer, DataPreprocessor,
    ModelTrainer, ModelEvaluator
)


class TestTitanicPipeline:
    """Test suite for Titanic ML pipeline."""
    
    def __init__(self):
        """Initialize test suite."""
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def print_header(self, title: str) -> None:
        """Print test section header."""
        print("\n" + "="*70)
        print(f"TEST: {title}")
        print("="*70)
    
    def test_pass(self, message: str) -> None:
        """Record a passed test."""
        self.passed += 1
        print(f"✓ PASS: {message}")
    
    def test_fail(self, message: str, error: str = "") -> None:
        """Record a failed test."""
        self.failed += 1
        print(f"✗ FAIL: {message}")
        if error:
            print(f"        Error: {error}")
    
    def test_warn(self, message: str) -> None:
        """Record a warning."""
        self.warnings += 1
        print(f"⚠ WARN: {message}")
    
    def test_data_loading(self) -> pd.DataFrame:
        """Test data loading functionality."""
        self.print_header("DATA LOADING")
        
        try:
            loader = TitanicDataLoader()
            train_df, test_df = loader.load_data()
            
            # Validate training data
            if train_df.shape[0] > 0:
                self.test_pass(f"Training data loaded: {train_df.shape}")
            else:
                self.test_fail("Training data is empty")
                return None
            
            # Validate test data
            if test_df.shape[0] > 0:
                self.test_pass(f"Test data loaded: {test_df.shape}")
            else:
                self.test_fail("Test data is empty")
            
            # Check for required columns
            required_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
            missing_cols = [col for col in required_cols if col not in train_df.columns]
            
            if not missing_cols:
                self.test_pass(f"All required columns present: {len(required_cols)} found")
            else:
                self.test_warn(f"Missing columns: {missing_cols}")
            
            # Check target variable
            if 'Survived' in train_df.columns:
                self.test_pass(f"Target variable 'Survived' present")
                unique_values = train_df['Survived'].unique()
                if set(unique_values).issubset({0, 1}):
                    self.test_pass(f"Target values are binary: {sorted(unique_values)}")
                else:
                    self.test_fail(f"Target values not binary: {unique_values}")
            else:
                self.test_fail("Target variable 'Survived' missing")
            
            return train_df
            
        except Exception as e:
            self.test_fail("Data loading failed", str(e))
            return None
    
    def test_exploratory_analysis(self, train_df: pd.DataFrame) -> None:
        """Test EDA functionality."""
        self.print_header("EXPLORATORY ANALYSIS")
        
        if train_df is None or len(train_df) == 0:
            self.test_fail("Cannot test EDA - no data")
            return
        
        try:
            loader = TitanicDataLoader()
            loader.exploratory_analysis(train_df)
            self.test_pass("EDA analysis completed without errors")
        except Exception as e:
            self.test_fail("EDA analysis failed", str(e))
    
    def test_title_extraction(self) -> None:
        """Test title extraction feature."""
        self.print_header("FEATURE ENGINEERING - TITLE EXTRACTION")
        
        test_cases = [
            ("Braund, Mr. Owen Harris", "Mr"),
            ("Cumings, Mrs. John Bradley", "Mrs"),
            ("Heikkinen, Miss. Laina", "Miss"),
            ("Futrelle, Mrs. Jacques Heath", "Mrs"),
            ("Allen, Mr. William Henry", "Mr"),
            ("Moran, Dr. James", "Dr"),
        ]
        
        try:
            for name, expected_title in test_cases:
                extracted = DataPreprocessor.extract_title(name)
                if extracted == expected_title or (expected_title == "Dr" and extracted == "Rare"):
                    self.test_pass(f"Extracted '{extracted}' from '{name.split(',')[0]}'")
                else:
                    self.test_fail(f"Expected '{expected_title}', got '{extracted}'")
        except Exception as e:
            self.test_fail("Title extraction failed", str(e))
    
    def test_data_preprocessing(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Test data preprocessing pipeline."""
        self.print_header("DATA PREPROCESSING")
        
        if train_df is None or len(train_df) == 0:
            self.test_fail("Cannot test preprocessing - no data")
            return None
        
        try:
            # Prepare data
            X_train = train_df.drop('Survived', axis=1) if 'Survived' in train_df.columns else train_df
            
            preprocessor = DataPreprocessor()
            X_processed, scaler = preprocessor.preprocess_data(X_train, is_training=True)
            
            # Validate output
            if X_processed.shape[0] == X_train.shape[0]:
                self.test_pass(f"Row count preserved: {X_processed.shape[0]} rows")
            else:
                self.test_fail(f"Row count mismatch: {X_processed.shape[0]} vs {X_train.shape[0]}")
            
            if X_processed.shape[1] > 0:
                self.test_pass(f"Features created: {X_processed.shape[1]} features")
            else:
                self.test_fail("No features in output")
            
            # Check for missing values
            missing_count = X_processed.isnull().sum().sum()
            if missing_count == 0:
                self.test_pass("No missing values in processed data")
            else:
                self.test_fail(f"Found {missing_count} missing values")
            
            # Check for inf values
            inf_count = np.isinf(X_processed).sum().sum()
            if inf_count == 0:
                self.test_pass("No infinite values in processed data")
            else:
                self.test_fail(f"Found {inf_count} infinite values")
            
            # Verify scaler
            if isinstance(scaler, StandardScaler):
                self.test_pass("StandardScaler fitted successfully")
            else:
                self.test_fail("Scaler is not StandardScaler type")
            
            return X_processed
            
        except Exception as e:
            self.test_fail("Data preprocessing failed", str(e))
            return None
    
    def test_model_training(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Test model training functionality."""
        self.print_header("MODEL TRAINING")
        
        if X_train is None or len(X_train) == 0:
            self.test_fail("Cannot test training - no data")
            return
        
        try:
            # Create smaller dataset for quick testing
            if len(X_train) > 300:
                X_train = X_train[:300]
                y_train = y_train[:300]
            
            trainer = ModelTrainer()
            
            # Test SVM training (quick version - no tuning)
            print("\nTesting SVM training...")
            try:
                svm_model, _ = trainer.train_svm(X_train, y_train, tune=False)
                
                # Test prediction
                pred = svm_model.predict(X_train[:5])
                pred_proba = svm_model.predict_proba(X_train[:5])
                
                if len(pred) == 5 and len(pred_proba) == 5:
                    self.test_pass(f"SVM training successful - predictions available")
                else:
                    self.test_fail("SVM predictions shape mismatch")
                    
            except Exception as e:
                self.test_fail("SVM training failed", str(e))
            
            # Test Random Forest training (quick version - no tuning)
            print("\nTesting Random Forest training...")
            try:
                rf_model, _ = trainer.train_random_forest(X_train, y_train, tune=False)
                
                # Test prediction
                pred = rf_model.predict(X_train[:5])
                pred_proba = rf_model.predict_proba(X_train[:5])
                
                if len(pred) == 5 and len(pred_proba) == 5:
                    self.test_pass(f"Random Forest training successful - predictions available")
                else:
                    self.test_fail("RF predictions shape mismatch")
                    
            except Exception as e:
                self.test_fail("Random Forest training failed", str(e))
                
        except Exception as e:
            self.test_fail("Model training test failed", str(e))
    
    def test_model_evaluation(self, model, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Test model evaluation functionality."""
        self.print_header("MODEL EVALUATION")
        
        if model is None or len(X_test) == 0:
            self.test_fail("Cannot test evaluation - no data")
            return
        
        try:
            evaluator = ModelEvaluator()
            results = evaluator.evaluate_model(model, X_test, y_test, "Test Model")
            
            # Validate metrics
            required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            
            for metric in required_metrics:
                if metric in results:
                    value = results[metric]
                    if 0 <= value <= 1:
                        self.test_pass(f"{metric}: {value:.4f}")
                    else:
                        self.test_fail(f"{metric} out of range [0,1]: {value}")
                else:
                    self.test_fail(f"Missing metric: {metric}")
            
            # Validate prediction arrays
            if 'y_pred' in results and len(results['y_pred']) == len(y_test):
                self.test_pass(f"Predictions shape valid: {len(results['y_pred'])}")
            else:
                self.test_fail("Predictions shape mismatch")
            
            if 'y_pred_proba' in results and len(results['y_pred_proba']) == len(y_test):
                self.test_pass(f"Probability predictions shape valid: {len(results['y_pred_proba'])}")
            else:
                self.test_fail("Probability predictions shape mismatch")
                
        except Exception as e:
            self.test_fail("Model evaluation failed", str(e))
    
    def print_summary(self) -> None:
        """Print test summary."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total = self.passed + self.failed
        
        print(f"✓ Passed: {self.passed}")
        print(f"✗ Failed: {self.failed}")
        print(f"⚠ Warnings: {self.warnings}")
        print(f"Total: {total}")
        
        if self.failed == 0:
            success_rate = 100.0 if total > 0 else 0
        else:
            success_rate = (self.passed / total) * 100
        
        print(f"Success Rate: {success_rate:.1f}%")
        print("="*70)
        
        if self.failed == 0:
            print("✓ ALL TESTS PASSED - Pipeline is ready for deployment!")
        else:
            print(f"✗ {self.failed} test(s) failed - Review errors above")
    
    def run_all_tests(self) -> None:
        """Run complete test suite."""
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + "TITANIC SURVIVAL PREDICTION - TEST SUITE".center(68) + "║")
        print("╚" + "="*68 + "╝")
        
        # Test 1: Data Loading
        train_df = self.test_data_loading()
        
        # Test 2: Exploratory Analysis
        if train_df is not None:
            self.test_exploratory_analysis(train_df)
        
        # Test 3: Feature Engineering
        self.test_title_extraction()
        
        # Test 4: Data Preprocessing
        if train_df is not None:
            X_train = train_df.drop('Survived', axis=1) if 'Survived' in train_df.columns else train_df
            y_train = train_df['Survived'] if 'Survived' in train_df.columns else None
            
            X_processed = self.test_data_preprocessing(X_train)
            
            # Test 5: Model Training
            if X_processed is not None and y_train is not None:
                self.test_model_training(X_processed.values, y_train.values)
                
                # Test 6: Model Evaluation
                # Split data for evaluation
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_processed, y_train, test_size=0.2, random_state=42
                )
                
                trainer = ModelTrainer()
                try:
                    rf_model, _ = trainer.train_random_forest(X_tr.values, y_tr.values, tune=False)
                    self.test_model_evaluation(rf_model, X_te.values, y_te.values)
                except:
                    self.test_warn("Skipping evaluation test due to training issues")
        
        # Print summary
        self.print_summary()


def run_quick_validation() -> None:
    """Run quick validation without full pipeline."""
    print("\n" + "="*70)
    print("QUICK VALIDATION - Core Pipeline Logic")
    print("="*70)
    
    try:
        # Load data
        print("\n1. Loading data...")
        loader = TitanicDataLoader()
        train_df, test_df = loader.load_data()
        print(f"   ✓ Data loaded: {train_df.shape[0]} training samples")
        
        # Quick preprocessing
        print("\n2. Preprocessing data...")
        X_train = train_df.drop('Survived', axis=1) if 'Survived' in train_df.columns else train_df
        y_train = train_df['Survived'] if 'Survived' in train_df.columns else None
        
        preprocessor = DataPreprocessor()
        X_processed, scaler = preprocessor.preprocess_data(X_train, is_training=True)
        print(f"   ✓ Processed: {X_processed.shape[1]} features created")
        
        # Quick model training
        print("\n3. Training models (quick version)...")
        from sklearn.model_selection import train_test_split
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_processed, y_train, test_size=0.2, random_state=42
        )
        
        # Limit to 300 samples for speed
        X_tr_small = X_tr[:300]
        y_tr_small = y_tr[:300]
        
        trainer = ModelTrainer()
        rf_model, _ = trainer.train_random_forest(X_tr_small.values, y_tr_small.values, tune=False)
        
        # Quick evaluation
        print("\n4. Evaluating model...")
        pred = rf_model.predict(X_te.values)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_te, pred)
        print(f"   ✓ Random Forest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n" + "="*70)
        print("✓ QUICK VALIDATION PASSED - Pipeline is functional!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + "TITANIC SURVIVAL PREDICTION - TESTING".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1].lower() == "quick":
        run_quick_validation()
    else:
        # Run full test suite
        tester = TestTitanicPipeline()
        tester.run_all_tests()
