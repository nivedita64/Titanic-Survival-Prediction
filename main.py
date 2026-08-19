"""
Titanic Survival Prediction - Production-Ready ML Pipeline

This module implements a complete binary classification pipeline for predicting
passenger survival on the Titanic dataset. It includes:
1. Exploratory Data Analysis (EDA) with comprehensive visualizations
2. Advanced data preprocessing with missing value imputation
3. Feature engineering (Title extraction, binning, family metrics)
4. Support Vector Machine (SVM) with hyperparameter tuning
5. Random Forest Classifier with optimized hyperparameters
6. Comprehensive model evaluation and comparison

Target Accuracy: ~78.9%+ on test set

Author: Data Scientist/ML Engineer
Version: 1.0
"""

import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING AND EXPLORATORY DATA ANALYSIS
# ============================================================================

class TitanicDataLoader:
    """
    Load and explore the Titanic dataset with comprehensive EDA visualizations.
    """
    
    @staticmethod
    def load_data(train_path: str = 'train.csv', test_path: str = 'test.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load Titanic dataset from CSV files.
        
        Args:
            train_path: Path to training data (default: 'train.csv')
            test_path: Path to test data (default: 'test.csv')
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print("Loading Titanic dataset...")
        
        # For demonstration, create sample data if files don't exist
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print(f"✓ Dataset loaded successfully")
        except FileNotFoundError:
            print("⚠ Dataset files not found. Creating sample Titanic dataset...")
            train_df, test_df = TitanicDataLoader._create_sample_data()
        
        print(f"Training set shape: {train_df.shape}")
        print(f"Test set shape: {test_df.shape}")
        
        return train_df, test_df
    
    @staticmethod
    def _create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create sample Titanic dataset for demonstration."""
        np.random.seed(42)
        
        n_train = 891
        n_test = 418
        
        # Create realistic sample names
        first_names = ['Owen', 'John', 'Laina', 'Jacques', 'William', 'Mary', 'James', 'Anna', 'Charles', 'Jane']
        last_names = ['Braund', 'Cumings', 'Heikkinen', 'Futrelle', 'Allen', 'Palsson', 'Billiard', 'Hall', 'McCarthy', 'Anderson']
        titles = ['Mr', 'Mrs', 'Miss', 'Master']
        
        def generate_names(n):
            names = []
            for i in range(n):
                first = np.random.choice(first_names)
                last = np.random.choice(last_names)
                title = np.random.choice(titles)
                name = f"{last}, {title} {first}"
                names.append(name)
            return names
        
        # Create training data
        train_data = {
            'PassengerId': range(1, n_train + 1),
            'Survived': np.random.binomial(1, 0.38, n_train),
            'Pclass': np.random.choice([1, 2, 3], n_train),
            'Name': generate_names(n_train),
            'Sex': np.random.choice(['male', 'female'], n_train),
            'Age': np.random.normal(30, 15, n_train),
            'SibSp': np.random.poisson(0.5, n_train),
            'Parch': np.random.poisson(0.4, n_train),
            'Ticket': [f'T{i}' for i in range(n_train)],
            'Fare': np.random.exponential(30, n_train),
            'Cabin': np.random.choice([None, 'C', 'E', 'F', 'G', 'D', 'A', 'B'], n_train),
            'Embarked': np.random.choice(['S', 'C', 'Q', None], n_train)
        }
        train_df = pd.DataFrame(train_data)
        
        # Create test data
        test_data = {
            'PassengerId': range(n_train + 1, n_train + n_test + 1),
            'Pclass': np.random.choice([1, 2, 3], n_test),
            'Name': generate_names(n_test),
            'Sex': np.random.choice(['male', 'female'], n_test),
            'Age': np.random.normal(30, 15, n_test),
            'SibSp': np.random.poisson(0.5, n_test),
            'Parch': np.random.poisson(0.4, n_test),
            'Ticket': [f'T{i}' for i in range(n_train, n_train + n_test)],
            'Fare': np.random.exponential(30, n_test),
            'Cabin': np.random.choice([None, 'C', 'E', 'F', 'G', 'D', 'A', 'B'], n_test),
            'Embarked': np.random.choice(['S', 'C', 'Q', None], n_test)
        }
        test_df = pd.DataFrame(test_data)
        
        return train_df, test_df
    
    @staticmethod
    def exploratory_analysis(df: pd.DataFrame) -> None:
        """
        Perform comprehensive exploratory data analysis.
        
        Args:
            df: Training DataFrame
        """
        print("\n" + "="*70)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        print("\nDataset Info:")
        print(f"  Shape: {df.shape}")
        print(f"  Total samples: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        print("\nMissing Values:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        for col in missing[missing > 0].index:
            print(f"  {col}: {missing[col]} ({missing_pct[col]:.1f}%)")
        
        print("\nBasic Statistics:")
        print(df.describe())
        
        print("\nSurvival Distribution:")
        if 'Survived' in df.columns:
            survival_counts = df['Survived'].value_counts()
            survival_pct = (survival_counts / len(df)) * 100
            print(f"  Died (0): {survival_counts[0]} ({survival_pct[0]:.1f}%)")
            print(f"  Survived (1): {survival_counts[1]} ({survival_pct[1]:.1f}%)")


class EDAVisualizer:
    """
    Create comprehensive EDA visualizations for the Titanic dataset.
    """
    
    @staticmethod
    def create_visualizations(df: pd.DataFrame) -> None:
        """
        Create multiple EDA visualization plots.
        
        Args:
            df: Training DataFrame
        """
        print("\nGenerating EDA visualizations...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Survival by Sex
        ax1 = plt.subplot(2, 3, 1)
        survival_sex = df.groupby('Sex')['Survived'].value_counts(normalize=True).unstack()
        survival_sex.plot(kind='bar', ax=ax1, color=['#d62728', '#2ca02c'])
        ax1.set_title('Survival Rate by Sex', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Sex')
        ax1.set_ylabel('Proportion')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
        ax1.legend(['Died', 'Survived'])
        ax1.grid(True, alpha=0.3)
        
        # 2. Survival by Passenger Class
        ax2 = plt.subplot(2, 3, 2)
        survival_class = df.groupby('Pclass')['Survived'].value_counts(normalize=True).unstack()
        survival_class.plot(kind='bar', ax=ax2, color=['#d62728', '#2ca02c'])
        ax2.set_title('Survival Rate by Passenger Class', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Passenger Class')
        ax2.set_ylabel('Proportion')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.legend(['Died', 'Survived'])
        ax2.grid(True, alpha=0.3)
        
        # 3. Age Distribution by Survival
        ax3 = plt.subplot(2, 3, 3)
        df[df['Survived'] == 0]['Age'].hist(bins=30, label='Died', ax=ax3, alpha=0.6, color='#d62728')
        df[df['Survived'] == 1]['Age'].hist(bins=30, label='Survived', ax=ax3, alpha=0.6, color='#2ca02c')
        ax3.set_title('Age Distribution by Survival', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Age')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Fare Distribution by Survival
        ax4 = plt.subplot(2, 3, 4)
        df[df['Survived'] == 0]['Fare'].hist(bins=30, label='Died', ax=ax4, alpha=0.6, color='#d62728')
        df[df['Survived'] == 1]['Fare'].hist(bins=30, label='Survived', ax=ax4, alpha=0.6, color='#2ca02c')
        ax4.set_title('Fare Distribution by Survival', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Fare')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Survival by Family Size
        ax5 = plt.subplot(2, 3, 5)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        family_survival = df.groupby('FamilySize')['Survived'].mean()
        family_survival.plot(kind='bar', ax=ax5, color='#1f77b4')
        ax5.set_title('Survival Rate by Family Size', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Family Size')
        ax5.set_ylabel('Survival Rate')
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
        ax5.grid(True, alpha=0.3)
        
        # 6. Correlation Heatmap (Numerical Features)
        ax6 = plt.subplot(2, 3, 6)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax6,
                   cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
        ax6.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
        print("✓ EDA visualizations saved as 'eda_visualizations.png'")
        plt.show()


# ============================================================================
# DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================

class DataPreprocessor:
    """
    Handle missing values, feature engineering, and data scaling.
    """
    
    @staticmethod
    def extract_title(name: str) -> str:
        """
        Extract title from passenger name.
        
        Args:
            name: Passenger name string
            
        Returns:
            Extracted title (e.g., 'Mr', 'Mrs', 'Miss', 'Master', 'Rare')
        """
        try:
            # Try standard format: "LastName, FirstName Title."
            if ',' in name and '.' in name:
                title = name.split(',')[1].split('.')[0].strip()
            else:
                # Fallback for different formats
                return 'Mr'  # Default title for sample data
            
            # Map rare titles
            rare_titles = ['Dona', 'Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 
                           'Major', 'Rev', 'Sir', 'Jonkheer']
            
            if title in rare_titles:
                return 'Rare'
            elif title in ['Mlle', 'Ms']:
                return 'Miss'
            elif title == 'Mme':
                return 'Mrs'
            else:
                return title
        except:
            return 'Mr'  # Default title on any parsing error
    
    @staticmethod
    def preprocess_data(
        df: pd.DataFrame,
        is_training: bool = True,
        scaler: StandardScaler = None
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Comprehensive data preprocessing and feature engineering.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data
            scaler: Pre-fitted scaler (for test data)
            
        Returns:
            Tuple of (processed_df, scaler)
        """
        print("\n" + "="*70)
        print("DATA PREPROCESSING & FEATURE ENGINEERING")
        print("="*70)
        
        df = df.copy()
        
        # 1. Handle Missing Values
        print("\n1. Handling missing values...")
        
        # Age: Impute by group (Pclass + Sex)
        if df['Age'].isnull().sum() > 0:
            age_by_group = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
            df['Age'] = df['Age'].fillna(age_by_group)
            print(f"   ✓ Age: Imputed {df['Age'].isnull().sum()} missing values with grouped median")
        
        # Embarked: Impute with mode
        if df['Embarked'].isnull().sum() > 0:
            embarked_mode = df['Embarked'].mode()[0]
            df['Embarked'] = df['Embarked'].fillna(embarked_mode)
            print(f"   ✓ Embarked: Imputed with mode '{embarked_mode}'")
        
        # Fare: Impute with median
        if df['Fare'].isnull().sum() > 0:
            fare_median = df['Fare'].median()
            df['Fare'] = df['Fare'].fillna(fare_median)
            print(f"   ✓ Fare: Imputed with median {fare_median:.2f}")
        
        # 2. Feature Engineering
        print("\n2. Engineering new features...")
        
        # Title extraction
        df['Title'] = df['Name'].apply(DataPreprocessor.extract_title)
        print(f"   ✓ Title: Extracted from names ({df['Title'].nunique()} unique titles)")
        
        # Family Size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        print(f"   ✓ FamilySize: Created (SibSp + Parch + 1)")
        
        # IsAlone flag
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        print(f"   ✓ IsAlone: Created flag for single travelers")
        
        # Age binning
        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                             labels=['Child', 'Teen', 'Young', 'Adult', 'Senior'])
        print(f"   ✓ AgeBin: Binned age into 5 categories")
        
        # Fare binning
        df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')
        print(f"   ✓ FareBin: Binned fare into quantile-based categories")
        
        # 3. Feature Selection (Drop unnecessary columns)
        print("\n3. Selecting features...")
        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        print(f"   ✓ Dropped non-predictive columns: {', '.join(drop_cols)}")
        
        # 4. Encoding Categorical Variables
        print("\n4. Encoding categorical variables...")
        
        categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # One-Hot Encoding
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"   ✓ One-Hot Encoded: {', '.join(categorical_cols)}")
        print(f"   ✓ Features after encoding: {df.shape[1]} columns")
        
        # 5. Scale Numerical Features
        print("\n5. Scaling numerical features...")
        
        numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'FamilySize']
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if is_training:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            print(f"   ✓ StandardScaler fitted and applied to {len(numerical_cols)} features")
        else:
            df[numerical_cols] = scaler.transform(df[numerical_cols])
            print(f"   ✓ StandardScaler applied using training set statistics")
        
        print(f"\n✓ Final feature shape: {df.shape}")
        
        return df, scaler


# ============================================================================
# MODEL TRAINING AND HYPERPARAMETER TUNING
# ============================================================================

class ModelTrainer:
    """
    Train and tune SVM and Random Forest models.
    """
    
    @staticmethod
    def train_svm(X_train: np.ndarray, y_train: np.ndarray, tune: bool = True) -> Tuple[SVC, Dict[str, Any]]:
        """
        Train SVM model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            tune: Whether to perform GridSearchCV
            
        Returns:
            Tuple of (best_model, best_params)
        """
        print("\n" + "="*70)
        print("TRAINING SVM MODEL")
        print("="*70)
        
        if tune:
            print("\nPerforming GridSearchCV for SVM hyperparameters...")
            
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
            
            svm = SVC(probability=True, random_state=42)
            grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
            
            print(f"\n✓ Best SVM parameters: {best_params}")
            print(f"✓ Best CV Accuracy: {best_cv_score:.4f}")
        else:
            print("\nTraining SVM with default parameters...")
            best_model = SVC(probability=True, random_state=42)
            best_model.fit(X_train, y_train)
            best_params = best_model.get_params()
            
            # Calculate CV score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
            print(f"✓ Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return best_model, best_params
    
    @staticmethod
    def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, tune: bool = True) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        Train Random Forest model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            tune: Whether to perform GridSearchCV
            
        Returns:
            Tuple of (best_model, best_params)
        """
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*70)
        
        if tune:
            print("\nPerforming GridSearchCV for Random Forest hyperparameters...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
            
            print(f"\n✓ Best Random Forest parameters: {best_params}")
            print(f"✓ Best CV Accuracy: {best_cv_score:.4f}")
        else:
            print("\nTraining Random Forest with default parameters...")
            best_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            best_model.fit(X_train, y_train)
            best_params = best_model.get_params()
            
            # Calculate CV score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
            print(f"✓ Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return best_model, best_params


# ============================================================================
# MODEL EVALUATION AND COMPARISON
# ============================================================================

class ModelEvaluator:
    """
    Evaluate and compare model performance with comprehensive metrics.
    """
    
    @staticmethod
    def evaluate_model(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n{model_name} - Test Set Performance:")
        print("-" * 70)
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        return {
            'model': model,
            'model_name': model_name,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    @staticmethod
    def plot_confusion_matrices(results_list: List[Dict[str, Any]], y_test: np.ndarray) -> None:
        """
        Plot confusion matrices for multiple models.
        
        Args:
            results_list: List of model evaluation results
            y_test: True test labels
        """
        fig, axes = plt.subplots(1, len(results_list), figsize=(14, 4))
        
        if len(results_list) == 1:
            axes = [axes]
        
        for idx, results in enumerate(results_list):
            cm = confusion_matrix(y_test, results['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f"{results['model_name']} - Confusion Matrix", 
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrices saved as 'confusion_matrices.png'")
        plt.show()
    
    @staticmethod
    def plot_roc_curves(results_list: List[Dict[str, Any]], y_test: np.ndarray) -> None:
        """
        Plot ROC curves for multiple models.
        
        Args:
            results_list: List of model evaluation results
            y_test: True test labels
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#1f77b4', '#ff7f0e']
        
        for idx, results in enumerate(results_list):
            fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
            roc_auc = results['roc_auc']
            
            ax.plot(fpr, tpr, color=colors[idx], lw=2.5,
                   label=f"{results['model_name']} (AUC = {roc_auc:.4f})")
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curves saved as 'roc_curves.png'")
        plt.show()
    
    @staticmethod
    def plot_feature_importance(rf_model: RandomForestClassifier, feature_names: List[str]) -> None:
        """
        Plot feature importance from Random Forest model.
        
        Args:
            rf_model: Trained Random Forest model
            feature_names: List of feature names
        """
        # Get feature importance
        importance = rf_model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importance)[-15:]  # Top 15 features
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(indices)), importance[indices], color='#1f77b4')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Top 15 Most Important Features - Random Forest', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance plot saved as 'feature_importance.png'")
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function orchestrating the complete ML pipeline.
    """
    
    print("\n" + "="*70)
    print("TITANIC SURVIVAL PREDICTION - PRODUCTION PIPELINE")
    print("="*70)
    
    # =====================================================================
    # Step 1: Load Data and EDA
    # =====================================================================
    loader = TitanicDataLoader()
    train_df, test_df = loader.load_data()
    
    # Basic exploration
    loader.exploratory_analysis(train_df)
    
    # Visualizations
    visualizer = EDAVisualizer()
    visualizer.create_visualizations(train_df)
    
    # =====================================================================
    # Step 2: Prepare Data
    # =====================================================================
    
    # Separate features and target
    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']
    
    # Preprocess training data
    preprocessor = DataPreprocessor()
    X_train_processed, scaler = preprocessor.preprocess_data(X_train, is_training=True)
    
    # Preprocess test data
    X_test_processed, _ = preprocessor.preprocess_data(test_df, is_training=False, scaler=scaler)
    
    # Features for importance plot
    feature_names = X_train_processed.columns.tolist()
    
    # Train-test split for evaluation
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nTrain set size: {X_train_split.shape[0]}")
    print(f"Test set size: {X_test_split.shape[0]}")
    print(f"Total features: {X_train_split.shape[1]}")
    
    # =====================================================================
    # Step 3: Train Models
    # =====================================================================
    trainer = ModelTrainer()
    
    # Train SVM
    svm_model, svm_params = trainer.train_svm(X_train_split, y_train_split, tune=True)
    
    # Train Random Forest
    rf_model, rf_params = trainer.train_random_forest(X_train_split, y_train_split, tune=True)
    
    # =====================================================================
    # Step 4: Evaluate Models
    # =====================================================================
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    evaluator = ModelEvaluator()
    
    svm_results = evaluator.evaluate_model(svm_model, X_test_split, y_test_split, "SVM")
    rf_results = evaluator.evaluate_model(rf_model, X_test_split, y_test_split, "Random Forest")
    
    # =====================================================================
    # Step 5: Visualizations
    # =====================================================================
    print("\n" + "="*70)
    print("GENERATING EVALUATION PLOTS")
    print("="*70)
    
    results_list = [svm_results, rf_results]
    
    evaluator.plot_confusion_matrices(results_list, y_test_split)
    evaluator.plot_roc_curves(results_list, y_test_split)
    evaluator.plot_feature_importance(rf_model, feature_names)
    
    # =====================================================================
    # Step 6: Summary Report
    # =====================================================================
    print("\n" + "="*70)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*70)
    
    print("\nModel Comparison:")
    print("-" * 70)
    print(f"{'Metric':<20} {'SVM':<20} {'Random Forest':<20}")
    print("-" * 70)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    for metric in metrics:
        svm_val = svm_results[metric]
        rf_val = rf_results[metric]
        print(f"{metric.replace('_', ' ').title():<20} {svm_val:<20.4f} {rf_val:<20.4f}")
    
    # Best model
    if rf_results['accuracy'] > svm_results['accuracy']:
        best_model = rf_results
        best_model_name = "Random Forest"
    else:
        best_model = svm_results
        best_model_name = "SVM"
    
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print("="*70)
    print(f"Accuracy: {best_model['accuracy']*100:.2f}%")
    print(f"ROC-AUC:  {best_model['roc_auc']:.4f}")
    print("="*70)
    
    return {
        'svm_model': svm_model,
        'rf_model': rf_model,
        'svm_results': svm_results,
        'rf_results': rf_results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    results = main()
