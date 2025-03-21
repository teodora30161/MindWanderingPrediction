import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
import os
from datetime import datetime
import joblib
from sklearn.metrics import f1_score


class MindWanderingClassifier:
    """Class for mind wandering classification using XGBoost."""
    
    def __init__(self, random_state=42, results_dir="results"):
        """Initialize the classifier with parameters and directories."""
        self.random_state = random_state
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize models and metrics containers
        self.best_model_orig = None
        self.best_model_smote = None
        self.features = None
        self.target = None
        self.engineered_features = []

    def compute_metrics(self, model, X_test, y_test):
        """Compute various performance metrics for a model."""
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
            
        # ROC and PR curves
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
            
        # Balanced accuracy
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10, strategy='uniform')

        return {
                'y_pred': y_pred,
                'y_probs': y_probs,
                'precision': precision,
                'recall': recall,
                'pr_auc': pr_auc,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'balanced_accuracy': bal_acc,
                'f1_score': f1,
                'calib_prob_true': prob_true,
                'calib_prob_pred': prob_pred
            }
        
        
    def load_data(self, file_path, target='ON_OFF', group_column='subj'):
        """Load and prepare the dataset."""
        print(f"Loading data from {file_path}...")
        self.df = pd.read_csv(file_path)
        self.target = target
        self.group_column = group_column
        print(f"Loaded {len(self.df)} records.")
        return self
        
    def select_features(self, features=None):
        """Select features for the model."""
        if features is None:
            features = ['BV', 'AE', 'pupil_size_median']
        self.features = features
        print(f"Selected features: {self.features}")
        return self
        
    def engineer_features(self):
        """Create features from the ones we selected."""
        print("Engineering additional features...")
        
        # Feature interactions (with safeguards against divide by zero)
        if 'BV' in self.features and 'AE' in self.features:
            self.df['BV_AE'] = self.df['BV'] * self.df['AE']
            self.engineered_features.append('BV_AE')
            
        if 'BV' in self.features and 'pupil_size_median' in self.features:
            self.df['BV_ratio_pupil'] = self.df['BV'] / (self.df['pupil_size_median'] + 1e-6)
            self.engineered_features.append('BV_ratio_pupil')
            
        if 'AE' in self.features and 'pupil_size_median' in self.features:
            self.df['AE_pupil'] = self.df['AE'] / (self.df['pupil_size_median'] + 1e-6)
            self.engineered_features.append('AE_pupil')
        
        # Add all features interaction to features list
        self.features = self.features + self.engineered_features
        print(f"Added engineered features: {self.engineered_features}")
        return self
        
    def clean_data(self):
        """Clean data by removing NaNs and encoding target variable."""
        print("Cleaning data...")
        all_needed_cols = self.features + [self.target, self.group_column]
        self.df_clean = self.df.dropna(subset=all_needed_cols).copy()
        
        # Checking target needs encoding
        if self.df_clean[self.target].dtype == 'object':
            print(f"Encoding target variable '{self.target}'...")
            self.df_clean[self.target] = self.df_clean[self.target].map({'ON': 0, 'OFF': 1})
        
        print(f"Clean data shape: {self.df_clean.shape}")
        return self
        
    def prepare_datasets(self, test_size=0.2):
        """Prepare train and test datasets."""
        print("Preparing datasets...")
        self.X = self.df_clean[self.features].values
        self.y = self.df_clean[self.target].values
        self.groups = self.df_clean[self.group_column].values
        
        # Train-Test Split
        self.X_train_orig, self.X_test_orig, self.y_train_orig, self.y_test_orig = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y
        )
        
        # Apply SMOTE
        smote = SMOTE(sampling_strategy=0.5, random_state=self.random_state)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train_orig, self.y_train_orig)
        
        print(f"Original training set shape: {self.X_train_orig.shape}")
        print(f"SMOTE-resampled training set shape: {self.X_train_smote.shape}")
        print(f"Test set shape: {self.X_test_orig.shape}")
        
        # Print class distribution
        print(f"Original training set class distribution: {np.bincount(self.y_train_orig)}")
        print(f"SMOTE-resampled training set class distribution: {np.bincount(self.y_train_smote)}")
        print(f"Test set class distribution: {np.bincount(self.y_test_orig)}")
        
        return self
        
    def optimize_hyperparameters(self, n_trials=50):
        """Optimize hyperparameters using Optuna."""
        print(f"Optimizing hyperparameters with {n_trials} trials...")
        
        def objective(trial, X_train, y_train):
            """Objective function for Optuna optimization."""
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10, log=True),
            }
            model = XGBClassifier(**params, random_state=self.random_state, use_label_encoder=False, 
                                 eval_metric='logloss')
            model.fit(X_train, y_train)
            y_pred = model.predict(self.X_test_orig)
            return balanced_accuracy_score(self.y_test_orig, y_pred)

        # Hyperparameter optimization for original and SMOTE datasets
        self.study_orig = optuna.create_study(direction='maximize')
        self.study_orig.optimize(lambda trial: objective(trial, self.X_train_orig, self.y_train_orig), 
                                n_trials=n_trials)
        
        self.study_smote = optuna.create_study(direction='maximize')
        self.study_smote.optimize(lambda trial: objective(trial, self.X_train_smote, self.y_train_smote), 
                                 n_trials=n_trials)
        
        print(f"Best parameters (original): {self.study_orig.best_params}")
        print(f"Best parameters (SMOTE): {self.study_smote.best_params}")
        
        return self
        
    def train_models(self):
        """Train models with optimized hyperparameters."""
        print("Training final models...")
        
        # Train original model
        self.best_model_orig = XGBClassifier(**self.study_orig.best_params, 
                                           random_state=self.random_state, 
                                           use_label_encoder=False, 
                                           eval_metric='logloss')
        self.best_model_orig.fit(self.X_train_orig, self.y_train_orig)

        # Apply Calibration (Isotonic)
        self.calibrated_model_orig = CalibratedClassifierCV(self.best_model_orig, method='isotonic', cv=5)
        self.calibrated_model_orig.fit(self.X_train_orig, self.y_train_orig)
        
        # Train SMOTE model
        self.best_model_smote = XGBClassifier(**self.study_smote.best_params, 
                                            random_state=self.random_state, 
                                            use_label_encoder=False, 
                                            eval_metric='logloss')
        self.best_model_smote.fit(self.X_train_smote, self.y_train_smote)
        
        # Apply Calibration to SMOTE model
        self.calibrated_model_smote = CalibratedClassifierCV(self.best_model_smote, method='isotonic', cv=5)
        self.calibrated_model_smote.fit(self.X_train_smote, self.y_train_smote)

        print("Models trained and calibrated.")
        return self
        
    def evaluate_models(self):
        """Evaluate trained models and compute metrics."""
        print("Evaluating models...")
        
        
        # Evaluate both models
        self.metrics_orig = self.compute_metrics(self.best_model_orig, self.X_test_orig, self.y_test_orig)
        self.metrics_smote = self.compute_metrics(self.best_model_smote, self.X_test_orig, self.y_test_orig)

        # Evaluate calibrated models
        self.metrics_orig_calibrated = self.compute_metrics(self.calibrated_model_orig, self.X_test_orig, self.y_test_orig)
        self.metrics_smote_calibrated = self.compute_metrics(self.calibrated_model_smote, self.X_test_orig, self.y_test_orig)


        
        # Display results
        print(f"Original model - Balanced Accuracy: {self.metrics_orig['balanced_accuracy']:.4f}, "
              f"ROC AUC: {self.metrics_orig['roc_auc']:.4f}")
        print(f"SMOTE model - Balanced Accuracy: {self.metrics_smote['balanced_accuracy']:.4f}, "
              f"ROC AUC: {self.metrics_smote['roc_auc']:.4f}")
        print("\nSample predicted probabilities (Original Model):")
        print(self.metrics_orig['y_probs'][:10])

        print(f"Original model (Calibrated) - Balanced Accuracy: {self.metrics_orig['balanced_accuracy']:.4f}, "
          f"ROC AUC: {self.metrics_orig['roc_auc']:.4f}, "
          f"F1 Score: {self.metrics_orig['f1_score']:.4f}")
        print(f"SMOTE model (Calibrated) - Balanced Accuracy: {self.metrics_smote['balanced_accuracy']:.4f}, "
          f"ROC AUC: {self.metrics_smote['roc_auc']:.4f}, "
          f"F1 Score: {self.metrics_smote['f1_score']:.4f}")

        # Print sample predicted probabilities
        print("\nSample predicted probabilities (Original Model - Calibrated):")
        print(self.metrics_orig['y_probs'][:10])

        print("\nSample predicted probabilities (SMOTE Model - Calibrated):")
        print(self.metrics_smote['y_probs'][:10])
            
        return self
        
    def plot_roc_comparison(self):
        """Plot ROC curves for original and SMOTE models."""
        plt.figure(figsize=(10, 8))
        plt.plot(self.metrics_orig['fpr'], self.metrics_orig['tpr'], 
                label=f"Without SMOTE (AUC = {self.metrics_orig['roc_auc']:.3f})", 
                color='blue', linewidth=2)
        plt.plot(self.metrics_smote['fpr'], self.metrics_smote['tpr'], 
                label=f"With SMOTE (AUC = {self.metrics_smote['roc_auc']:.3f})", 
                color='red', linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve Comparison: With vs Without SMOTE", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save figure
        roc_fig_path = os.path.join(self.results_dir, f"roc_comparison_{self.timestamp}.png")
        plt.savefig(roc_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC comparison saved to {roc_fig_path}")
        
        return self
        
        
        
    def perform_loso_cv(self):
        """Perform Leave-One-Subject-Out cross-validation."""
        print("Performing Leave-One-Subject-Out cross-validation...")
        logo = LeaveOneGroupOut()
        self.participant_aucs = {}
        self.y_test_pooled, self.y_probs_pooled = [], []
        
        # Use the best model from the original dataset
        best_model_params = self.study_orig.best_params
        
        # For progress tracking
        total_folds = len(np.unique(self.groups))
        current_fold = 0
        
        for train_idx, test_idx in logo.split(self.X, self.y, self.groups):
            current_fold += 1
            participant = np.unique(self.groups[test_idx])[0]
            print(f"Processing fold {current_fold}/{total_folds} - Participant: {participant}")
            
            X_train_fold, X_test_fold = self.X[train_idx], self.X[test_idx]
            y_train_fold, y_test_fold = self.y[train_idx], self.y[test_idx]
            
            # Train model for this fold
            fold_model = XGBClassifier(**best_model_params, random_state=self.random_state, 
                                      use_label_encoder=False, eval_metric='logloss')
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Get predictions
            y_probs_fold = fold_model.predict_proba(X_test_fold)[:, 1]
            
            # Compute AUC for this participant
            if len(np.unique(y_test_fold)) > 1:  # Only compute AUC if both classes are present
                fpr, tpr, _ = roc_curve(y_test_fold, y_probs_fold)
                fold_auc = auc(fpr, tpr)
            else:
                fold_auc = np.nan
                print(f"Warning: Participant {participant} has only one class in test set. AUC set to NaN.")
                
            self.participant_aucs[participant] = fold_auc
            
            # Store for pooled calculation
            self.y_test_pooled.extend(y_test_fold)
            self.y_probs_pooled.extend(y_probs_fold)
        
        # Compute overall AUC using pooled predictions
        fpr_pooled, tpr_pooled, _ = roc_curve(self.y_test_pooled, self.y_probs_pooled)
        self.roc_auc_pooled = auc(fpr_pooled, tpr_pooled)
        
        print(f"Final LOSO-Based AUC using Pooled Test Set: {self.roc_auc_pooled:.4f}")
        
        return self
        
    def plot_loso_results(self):
        """Plot LOSO AUC results per participant."""
        # Sort participants by AUC
        sorted_participants = sorted(self.participant_aucs.items(), 
                                    key=lambda x: x[1] if not np.isnan(x[1]) else -1)
        participants = [p[0] for p in sorted_participants]
        aucs = [p[1] for p in sorted_participants]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(participants, aucs, color='skyblue')
        
        # Add a horizontal line for the pooled AUC
        plt.axhline(y=self.roc_auc_pooled, color='r', linestyle='-', 
                   label=f"Pooled AUC: {self.roc_auc_pooled:.3f}")
        
        # Highlight bars with NaN values
        for i, auc_val in enumerate(aucs):
            if np.isnan(auc_val):
                bars[i].set_color('lightgray')
                bars[i].set_hatch('///')
        
        plt.xlabel("Participant", fontsize=12)
        plt.ylabel("AUC Score", fontsize=12)
        plt.title("AUC per Participant (Leave-One-Subject-Out CV)", fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        # Save figure
        loso_fig_path = os.path.join(self.results_dir, f"loso_auc_per_participant_{self.timestamp}.png")
        plt.savefig(loso_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"LOSO AUC results saved to {loso_fig_path}")
        
        return self
    
    def plot_calibration_curve(self):
        """Plot calibration curves for original and SMOTE models (before and after calibration)."""
        plt.figure(figsize=(8, 6))

        # Before calibration
        plt.plot(self.metrics_orig['calib_prob_pred'], self.metrics_orig['calib_prob_true'],
                label='Original Model (Before Calibration)', marker='o', color='blue')
        plt.plot(self.metrics_smote['calib_prob_pred'], self.metrics_smote['calib_prob_true'],
                label='SMOTE Model (Before Calibration)', marker='s', color='red')

        # After calibration
        calib_metrics_orig = self.compute_metrics(self.calibrated_model_orig, self.X_test_orig, self.y_test_orig)
        calib_metrics_smote = self.compute_metrics(self.calibrated_model_smote, self.X_test_orig, self.y_test_orig)


        plt.plot(calib_metrics_orig['calib_prob_pred'], calib_metrics_orig['calib_prob_true'],
                label='Original Model (After Calibration)', marker='o', color='cyan')
        plt.plot(calib_metrics_smote['calib_prob_pred'], calib_metrics_smote['calib_prob_true'],
                label='SMOTE Model (After Calibration)', marker='s', color='orange')

        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve (Before & After)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        calib_fig_path = os.path.join(self.results_dir, f"calibration_curve_{self.timestamp}.png")
        plt.savefig(calib_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Calibration curve saved to {calib_fig_path}")

        return self


        
    def save_models(self):
        """Save trained models and preprocessing objects."""
        # Save original model
        orig_model_path = os.path.join(self.results_dir, f"original_model_{self.timestamp}.joblib")
        joblib.dump(self.best_model_orig, orig_model_path)
        
        # Save SMOTE model
        smote_model_path = os.path.join(self.results_dir, f"smote_model_{self.timestamp}.joblib")
        joblib.dump(self.best_model_smote, smote_model_path)

         # Save calibrated models
        calibrated_orig_path = os.path.join(self.results_dir, f"calibrated_model_orig_{self.timestamp}.joblib")
        joblib.dump(self.calibrated_model_orig, calibrated_orig_path)

        calibrated_smote_path = os.path.join(self.results_dir, f"calibrated_model_smote_{self.timestamp}.joblib")
        joblib.dump(self.calibrated_model_smote, calibrated_smote_path)

        print(f"Calibrated models saved to {self.results_dir}")
    
        
        # Save feature names and other metadata
        metadata = {
            'features': self.features,
            'target': self.target,
            'group_column': self.group_column,
            'original_params': self.study_orig.best_params,
            'smote_params': self.study_smote.best_params,
            'original_metrics': {
                'balanced_accuracy': self.metrics_orig['balanced_accuracy'],
                'roc_auc': self.metrics_orig['roc_auc']
            },
            'smote_metrics': {
                'balanced_accuracy': self.metrics_smote['balanced_accuracy'],
                'roc_auc': self.metrics_smote['roc_auc']
            },
            'loso_auc_pooled': self.roc_auc_pooled,
            'participant_aucs': self.participant_aucs
        }
        
        metadata_path = os.path.join(self.results_dir, f"model_metadata_{self.timestamp}.joblib")
        joblib.dump(metadata, metadata_path)
        
        print(f"Models and metadata saved to {self.results_dir}")
        return self
        
    def generate_report(self):
        """ Report of the analysis."""
        report_path = os.path.join(self.results_dir, f"analysis_report_{self.timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("==== MIND WANDERING CLASSIFICATION REPORT ====\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("== DATA SUMMARY ==\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Clean samples: {len(self.df_clean)}\n")
            f.write(f"Features used: {', '.join(self.features)}\n")
            f.write(f"Target variable: {self.target}\n")
            f.write(f"Group variable: {self.group_column}\n\n")
            
            f.write("== CLASS DISTRIBUTION ==\n")
            f.write(f"Original training set: {np.bincount(self.y_train_orig)}\n")
            f.write(f"SMOTE-resampled training set: {np.bincount(self.y_train_smote)}\n")
            f.write(f"Test set: {np.bincount(self.y_test_orig)}\n\n")

            f.write("== SAMPLE PREDICTED PROBABILITIES ==\n")
            f.write("Original Model:\n")
            f.write(", ".join([f"{prob:.4f}" for prob in self.metrics_orig['y_probs'][:10]]) + "\n")
            f.write("SMOTE Model:\n")
            f.write(", ".join([f"{prob:.4f}" for prob in self.metrics_smote['y_probs'][:10]]) + "\n\n")

            
            f.write("== MODEL PERFORMANCE ==\n")
            f.write(f"Original model - Balanced Accuracy: {self.metrics_orig['balanced_accuracy']:.4f}, "
                   f"ROC AUC: {self.metrics_orig['roc_auc']:.4f}, "f"F1 Score: {self.metrics_orig['f1_score']:.4f}\n")
            f.write(f"SMOTE model - Balanced Accuracy: {self.metrics_smote['balanced_accuracy']:.4f}, "
                   f"ROC AUC: {self.metrics_smote['roc_auc']:.4f}, "f"F1 Score: {self.metrics_smote['f1_score']:.4f}\n\n")
            
            f.write(f"Calibrated Original model - Balanced Accuracy: {self.metrics_orig_calibrated['balanced_accuracy']:.4f}, "
                f"ROC AUC: {self.metrics_orig_calibrated['roc_auc']:.4f}, "
                f"F1 Score: {self.metrics_orig_calibrated['f1_score']:.4f}\n")

            f.write(f"Calibrated SMOTE model - Balanced Accuracy: {self.metrics_smote_calibrated['balanced_accuracy']:.4f}, "
            f"ROC AUC: {self.metrics_smote_calibrated['roc_auc']:.4f}, "
            f"F1 Score: {self.metrics_smote_calibrated['f1_score']:.4f}\n\n")
            
            f.write("== CALIBRATION METRICS ==\n")
            f.write("Original Model Calibration (Mean Predicted vs. Fraction of Positives):\n")
            for pred, true in zip(self.metrics_orig['calib_prob_pred'], self.metrics_orig['calib_prob_true']):
                f.write(f"{pred:.4f} -> {true:.4f}\n")

            f.write("\nSMOTE Model Calibration (Mean Predicted vs. Fraction of Positives):\n")
            for pred, true in zip(self.metrics_smote['calib_prob_pred'], self.metrics_smote['calib_prob_true']):
                f.write(f"{pred:.4f} -> {true:.4f}\n\n")

            f.write("== LEAVE-ONE-SUBJECT-OUT RESULTS ==\n")
            f.write("AUC per Participant:\n")
            for participant, auc_score in self.participant_aucs.items():
                f.write(f"{participant}: {auc_score:.4f}\n")
            f.write(f"\nLOSO Pooled AUC: {self.roc_auc_pooled:.4f}\n\n")
            
            f.write("== BEST HYPERPARAMETERS ==\n")
            f.write("Original model:\n")
            for param, value in self.study_orig.best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\nSMOTE model:\n")
            for param, value in self.study_smote.best_params.items():
                f.write(f"  {param}: {value}\n")

            f.write("== FEATURE IMPORTANCE (Original Model) ==\n")
            feature_importances = self.best_model_orig.feature_importances_
            for feature, importance in zip(self.features, feature_importances):
                f.write(f"{feature}: {importance:.4f}\n")

            f.write("\n== FEATURE IMPORTANCE (SMOTE Model) ==\n")
            feature_importances = self.best_model_smote.feature_importances_
            for feature, importance in zip(self.features, feature_importances):
                f.write(f"{feature}: {importance:.4f}\n\n")
        
        print(f"Analysis report saved to {report_path}")
        return self

# Main execution
if __name__ == "__main__":
    file_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/all_probes4.csv"
    
    classifier = MindWanderingClassifier(random_state=42, results_dir="XGVBoostCorrectedResult")
    
    # Run pipeline
    (classifier
        .load_data(file_path)
        .select_features()
        .engineer_features()
        .clean_data()
        .prepare_datasets()
        .optimize_hyperparameters(n_trials=50)
        .train_models()
        .evaluate_models()
        .plot_roc_comparison()
        .plot_calibration_curve()
        .perform_loso_cv()
        .plot_loso_results()
        .save_models()
        .generate_report()
    )
    
    print("Analysis complete!") 