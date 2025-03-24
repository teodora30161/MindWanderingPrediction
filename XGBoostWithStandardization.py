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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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
        print("Interaction of diffeerent features...")
        
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
        print(f"Added interaction features: {self.engineered_features}")
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

    def standardize_by_participant(self):
        """
        Standardize features separately for each participant to account for individual differences.
        This method performs z-score normalization (mean=0, std=1) for each participant's data.
        """
        print("Performing participant-wise standardization...")
        
        # Create a copy of the dataframe to avoid modifying the original
        self.df_standardized = self.df_clean.copy()
        
        # Get participant IDs (assuming group_column contains participant identifiers)
        participants = self.df_standardized[self.group_column].unique()
        print(f"Found {len(participants)} unique participants")
        
        # Store the StandardScaler objects for each participant (for later use in prediction)
        self.participant_scalers = {}
        
        # Standardize features for each participant separately
        for participant in participants:
            participant_mask = self.df_standardized[self.group_column] == participant
            participant_data = self.df_standardized.loc[participant_mask, self.features]
            
            # Create and fit a StandardScaler for this participant
            scaler = StandardScaler()
            
            # Transform the data
            scaled_features = scaler.fit_transform(participant_data)
            
            # Store the scaler for this participant
            self.participant_scalers[participant] = scaler
            
            # Update the dataframe with standardized values
            self.df_standardized.loc[participant_mask, self.features] = scaled_features
        
        # Update X with standardized features
        self.X = self.df_standardized[self.features].values
        
        print("Participant-wise standardization completed")
        return self

    def prepare_datasets_with_standardization(self, train_size=0.7, calib_size=0.3):
        """
        Prepare datasets with participant-wise standardization.
        This method first standardizes features by participant and then splits the data.
        """
        print("Standardizing by participant and preparing datasets...")
        
        # First apply standardization
        self.standardize_by_participant()
        
        # Then proceed with dataset preparation as before
        self.y = self.df_standardized[self.target].values
        self.groups = self.df_standardized[self.group_column].values
        
        # First split: 70% train+validation, 30% calibration
        self.X_train_val, self.X_calib, self.y_train_val, self.y_calib = train_test_split(
            self.X, self.y, test_size=calib_size, random_state=self.random_state, stratify=self.y
        )
        
        # Second split: Within train_val set, split further for hyperparameter tuning
        test_size_adjusted = 0.2 / train_size  # Equivalent to ~14% of original data
        self.X_train_orig, self.X_test_orig, self.y_train_orig, self.y_test_orig = train_test_split(
            self.X_train_val, self.y_train_val, test_size=test_size_adjusted, 
            random_state=self.random_state, stratify=self.y_train_val
        )
        
        # Apply SMOTE only to the training data
        smote = SMOTE(sampling_strategy=0.5, random_state=self.random_state)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train_orig, self.y_train_orig)
        
        print(f"Original training set shape: {self.X_train_orig.shape}")
        print(f"SMOTE-resampled training set shape: {self.X_train_smote.shape}")
        print(f"Test set shape: {self.X_test_orig.shape}")
        print(f"Calibration set shape: {self.X_calib.shape}")
        
        # Print class distribution
        print(f"Original training set class distribution: {np.bincount(self.y_train_orig)}")
        print(f"SMOTE-resampled training set class distribution: {np.bincount(self.y_train_smote)}")
        print(f"Test set class distribution: {np.bincount(self.y_test_orig)}")
        print(f"Calibration set class distribution: {np.bincount(self.y_calib)}")
        
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
        """Train models with optimized hyperparameters and apply calibration using separate calibration set."""
        print("Training final models...")
        
        # Train original model
        self.best_model_orig = XGBClassifier(**self.study_orig.best_params, 
                                           random_state=self.random_state, 
                                           use_label_encoder=False, 
                                           eval_metric='logloss')
        self.best_model_orig.fit(self.X_train_orig, self.y_train_orig)

        # Apply Calibration (Isotonic) using the held-out calibration set
        self.calibrated_model_orig = CalibratedClassifierCV(
            self.best_model_orig, method='isotonic', cv='prefit'
        )
        self.calibrated_model_orig.fit(self.X_calib, self.y_calib)
        
        # Train SMOTE model
        self.best_model_smote = XGBClassifier(**self.study_smote.best_params, 
                                            random_state=self.random_state, 
                                            use_label_encoder=False, 
                                            eval_metric='logloss')
        self.best_model_smote.fit(self.X_train_smote, self.y_train_smote)
        
        # Apply Calibration to SMOTE model using the held-out calibration set
        self.calibrated_model_smote = CalibratedClassifierCV(
            self.best_model_smote, method='isotonic', cv='prefit'
        )
        self.calibrated_model_smote.fit(self.X_calib, self.y_calib)

        print("Models trained and calibrated with separate calibration dataset.")
        return self
    
    def evaluate_models(self):
        """Evaluate trained models and compute metrics."""
        print("Evaluating models...")
        
        # Create a holdout test set for final evaluation
        # Using original test set as our holdout for final evaluation
        
        # Evaluate both uncalibrated models
        self.metrics_orig = self.compute_metrics(self.best_model_orig, self.X_test_orig, self.y_test_orig)
        self.metrics_smote = self.compute_metrics(self.best_model_smote, self.X_test_orig, self.y_test_orig)

        # Evaluate calibrated models
        self.metrics_orig_calibrated = self.compute_metrics(self.calibrated_model_orig, self.X_test_orig, self.y_test_orig)
        self.metrics_smote_calibrated = self.compute_metrics(self.calibrated_model_smote, self.X_test_orig, self.y_test_orig)
        
        # Display results
        print(f"Original model (Uncalibrated) - Balanced Accuracy: {self.metrics_orig['balanced_accuracy']:.4f}, "
              f"ROC AUC: {self.metrics_orig['roc_auc']:.4f}, "
              f"F1 Score: {self.metrics_orig['f1_score']:.4f}")
        print(f"SMOTE model (Uncalibrated) - Balanced Accuracy: {self.metrics_smote['balanced_accuracy']:.4f}, "
              f"ROC AUC: {self.metrics_smote['roc_auc']:.4f}, "
              f"F1 Score: {self.metrics_smote['f1_score']:.4f}")

        print(f"Original model (Calibrated) - Balanced Accuracy: {self.metrics_orig_calibrated['balanced_accuracy']:.4f}, "
              f"ROC AUC: {self.metrics_orig_calibrated['roc_auc']:.4f}, "
              f"F1 Score: {self.metrics_orig_calibrated['f1_score']:.4f}")
        print(f"SMOTE model (Calibrated) - Balanced Accuracy: {self.metrics_smote_calibrated['balanced_accuracy']:.4f}, "
              f"ROC AUC: {self.metrics_smote_calibrated['roc_auc']:.4f}, "
              f"F1 Score: {self.metrics_smote_calibrated['f1_score']:.4f}")

        # Print sample predicted probabilities
        print("\nSample predicted probabilities (Original Model - Uncalibrated):")
        print(self.metrics_orig['y_probs'][:10])
        
        print("\nSample predicted probabilities (Original Model - Calibrated):")
        print(self.metrics_orig_calibrated['y_probs'][:10])

        print("\nSample predicted probabilities (SMOTE Model - Uncalibrated):")
        print(self.metrics_smote['y_probs'][:10])
        
        print("\nSample predicted probabilities (SMOTE Model - Calibrated):")
        print(self.metrics_smote_calibrated['y_probs'][:10])
            
        return self
    
    # [Other methods remain similar, with adjustments to reference the calibrated models]
    
    def perform_loso_cv(self):
        """
        Perform Leave-One-Subject-Out cross-validation with calibration.
        Each fold uses the same approach: train on n-1 subjects, calibrate on a portion of data, test on held-out subject.
        """
        print("Performing Leave-One-Subject-Out cross-validation with calibration...")
        logo = LeaveOneGroupOut()
        self.participant_aucs = {}
        self.participant_aucs_calibrated = {}
        self.y_test_pooled, self.y_probs_pooled = [], []
        self.y_test_pooled_calibrated, self.y_probs_pooled_calibrated = [], []
        
        # Use the best model from the original dataset
        best_model_params = self.study_orig.best_params
        
        # For progress tracking
        total_folds = len(np.unique(self.groups))
        current_fold = 0
        
        for train_idx, test_idx in logo.split(self.X, self.y, self.groups):
            current_fold += 1
            participant = np.unique(self.groups[test_idx])[0]
            print(f"Processing fold {current_fold}/{total_folds} - Participant: {participant}")
            
            # Get training and test data for this fold
            X_train_fold_full, X_test_fold = self.X[train_idx], self.X[test_idx]
            y_train_fold_full, y_test_fold = self.y[train_idx], self.y[test_idx]
            
            # Split training data to get a calibration set
            X_train_fold, X_calib_fold, y_train_fold, y_calib_fold = train_test_split(
                X_train_fold_full, y_train_fold_full, test_size=0.3, 
                random_state=self.random_state, stratify=y_train_fold_full
            )
            
            # Train uncalibrated model for this fold
            fold_model = XGBClassifier(**best_model_params, random_state=self.random_state, 
                                      use_label_encoder=False, eval_metric='logloss')
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Create calibrated model
            calibrated_fold_model = CalibratedClassifierCV(fold_model, method='isotonic', cv='prefit')
            calibrated_fold_model.fit(X_calib_fold, y_calib_fold)
            
            # Get predictions from both models
            y_probs_fold = fold_model.predict_proba(X_test_fold)[:, 1]
            y_probs_fold_calibrated = calibrated_fold_model.predict_proba(X_test_fold)[:, 1]
            
            # Compute AUC for this participant (uncalibrated)
            if len(np.unique(y_test_fold)) > 1:  # Only compute AUC if both classes are present
                fpr, tpr, _ = roc_curve(y_test_fold, y_probs_fold)
                fold_auc = auc(fpr, tpr)
                
                # Compute AUC for calibrated model
                fpr_calibrated, tpr_calibrated, _ = roc_curve(y_test_fold, y_probs_fold_calibrated)
                fold_auc_calibrated = auc(fpr_calibrated, tpr_calibrated)
            else:
                fold_auc = np.nan
                fold_auc_calibrated = np.nan
                print(f"Warning: Participant {participant} has only one class in test set. AUC set to NaN.")
                
            self.participant_aucs[participant] = fold_auc
            self.participant_aucs_calibrated[participant] = fold_auc_calibrated
            
            # Store for pooled calculation
            self.y_test_pooled.extend(y_test_fold)
            self.y_probs_pooled.extend(y_probs_fold)
            self.y_test_pooled_calibrated.extend(y_test_fold)
            self.y_probs_pooled_calibrated.extend(y_probs_fold_calibrated)
        
        # Compute overall AUC using pooled predictions (uncalibrated)
        fpr_pooled, tpr_pooled, _ = roc_curve(self.y_test_pooled, self.y_probs_pooled)
        self.roc_auc_pooled = auc(fpr_pooled, tpr_pooled)
        
        # Compute overall AUC using pooled predictions (calibrated)
        fpr_pooled_calibrated, tpr_pooled_calibrated, _ = roc_curve(
            self.y_test_pooled_calibrated, self.y_probs_pooled_calibrated
        )
        self.roc_auc_pooled_calibrated = auc(fpr_pooled_calibrated, tpr_pooled_calibrated)
        
        print(f"Final LOSO-Based AUC (Uncalibrated): {self.roc_auc_pooled:.4f}")
        print(f"Final LOSO-Based AUC (Calibrated): {self.roc_auc_pooled_calibrated:.4f}")
        
        return self

    def perform_loso_cv_with_standardization(self):
        """
        Perform Leave-One-Subject-Out cross-validation with participant-wise standardization.
        Each fold standardizes training participants separately from the test participant.
        """
        print("Performing Leave-One-Subject-Out cross-validation with participant-wise standardization...")
        logo = LeaveOneGroupOut()
        self.participant_aucs = {}
        self.participant_aucs_calibrated = {}
        self.y_test_pooled, self.y_probs_pooled = [], []
        self.y_test_pooled_calibrated, self.y_probs_pooled_calibrated = [], []
        
        # Use the best model from the original dataset
        best_model_params = self.study_orig.best_params
        
        # For progress tracking
        total_folds = len(np.unique(self.groups))
        current_fold = 0
        
        # Get the raw features and group info (before any standardization)
        X_raw = self.df_clean[self.features].values
        y_raw = self.df_clean[self.target].values
        groups_raw = self.df_clean[self.group_column].values
        
        for train_idx, test_idx in logo.split(X_raw, y_raw, groups_raw):
            current_fold += 1
            participant = np.unique(groups_raw[test_idx])[0]
            print(f"Processing fold {current_fold}/{total_folds} - Participant: {participant}")
            
            # Get raw data for this fold
            X_train_fold_raw = X_raw[train_idx]
            X_test_fold_raw = X_raw[test_idx]
            y_train_fold_raw = y_raw[train_idx]
            y_test_fold = y_raw[test_idx]  # We don't need to standardize the target variable
            groups_train = groups_raw[train_idx]
            
            # Standardize training data (group by training participants)
            train_participants = np.unique(groups_train)
            X_train_fold_std = X_train_fold_raw.copy()
            
            # Create a dictionary to store scalers for each training participant
            train_scalers = {}
            
            # Standardize each training participant separately
            for train_participant in train_participants:
                # Get indices for this participant
                participant_mask = groups_train == train_participant
                # Get the data for this participant
                participant_data = X_train_fold_raw[participant_mask]
                
                # Create and fit a scaler
                scaler = StandardScaler()
                participant_scaled = scaler.fit_transform(participant_data)
                
                # Store the scaled data
                X_train_fold_std[participant_mask] = participant_scaled
                
                # Store the scaler
                train_scalers[train_participant] = scaler
            
            # For the test participant, we create a separate scaler 
            # We only use their data for fitting, but we won't transform their data with this specific scaler
            test_scaler = StandardScaler()
            test_scaler.fit(X_test_fold_raw)
            
            # Now standardize the test data with the global mean and std from all training participants
            # First, combine all training data to compute global statistics
            all_train_scaler = StandardScaler()
            all_train_scaler.fit(X_train_fold_raw)
            
            # Apply the global training scaler to the test data
            X_test_fold_std = all_train_scaler.transform(X_test_fold_raw)
            
            # Split training data to get a calibration set (all with standardized features)
            X_train_fold, X_calib_fold, y_train_fold, y_calib_fold = train_test_split(
                X_train_fold_std, y_train_fold_raw, test_size=0.3, 
                random_state=self.random_state, stratify=y_train_fold_raw
            )
            
            # Train uncalibrated model for this fold
            fold_model = XGBClassifier(**best_model_params, random_state=self.random_state, 
                                    use_label_encoder=False, eval_metric='logloss')
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Create calibrated model
            calibrated_fold_model = CalibratedClassifierCV(fold_model, method='isotonic', cv='prefit')
            calibrated_fold_model.fit(X_calib_fold, y_calib_fold)
            
            # Get predictions from both models
            y_probs_fold = fold_model.predict_proba(X_test_fold_std)[:, 1]
            y_probs_fold_calibrated = calibrated_fold_model.predict_proba(X_test_fold_std)[:, 1]
            
            # Compute AUC for this participant (uncalibrated)
            if len(np.unique(y_test_fold)) > 1:  # Only compute AUC if both classes are present
                fpr, tpr, _ = roc_curve(y_test_fold, y_probs_fold)
                fold_auc = auc(fpr, tpr)
                
                # Compute AUC for calibrated model
                fpr_calibrated, tpr_calibrated, _ = roc_curve(y_test_fold, y_probs_fold_calibrated)
                fold_auc_calibrated = auc(fpr_calibrated, tpr_calibrated)
            else:
                fold_auc = np.nan
                fold_auc_calibrated = np.nan
                print(f"Warning: Participant {participant} has only one class in test set. AUC set to NaN.")
                
            self.participant_aucs[participant] = fold_auc
            self.participant_aucs_calibrated[participant] = fold_auc_calibrated
            
            # Store for pooled calculation
            self.y_test_pooled.extend(y_test_fold)
            self.y_probs_pooled.extend(y_probs_fold)
            self.y_test_pooled_calibrated.extend(y_test_fold)
            self.y_probs_pooled_calibrated.extend(y_probs_fold_calibrated)
        
        # Compute overall AUC using pooled predictions (uncalibrated)
        fpr_pooled, tpr_pooled, _ = roc_curve(self.y_test_pooled, self.y_probs_pooled)
        self.roc_auc_pooled = auc(fpr_pooled, tpr_pooled)
        
        # Compute overall AUC using pooled predictions (calibrated)
        fpr_pooled_calibrated, tpr_pooled_calibrated, _ = roc_curve(
            self.y_test_pooled_calibrated, self.y_probs_pooled_calibrated
        )
        self.roc_auc_pooled_calibrated = auc(fpr_pooled_calibrated, tpr_pooled_calibrated)
        
        print(f"Final LOSO-Based AUC with standardization (Uncalibrated): {self.roc_auc_pooled:.4f}")
        print(f"Final LOSO-Based AUC with standardization (Calibrated): {self.roc_auc_pooled_calibrated:.4f}")
        
        return self
    
    def plot_roc_comparison(self):
        """Plot ROC curves for all models (original vs SMOTE, calibrated vs uncalibrated). """
        plt.figure(figsize=(12, 8))
    
        # Plot Original Model (Uncalibrated)
        plt.plot(
            self.metrics_orig['fpr'], 
            self.metrics_orig['tpr'], 
            label=f"Original (AUC = {self.metrics_orig['roc_auc']:.3f})",
            color='blue', 
            linestyle='-'
        )
        
        # Plot Original Model (Calibrated)
        plt.plot(
            self.metrics_orig_calibrated['fpr'], 
            self.metrics_orig_calibrated['tpr'], 
            label=f"Original Calibrated (AUC = {self.metrics_orig_calibrated['roc_auc']:.3f})",
            color='blue', 
            linestyle='--'
        )
        
        # Plot SMOTE Model (Uncalibrated)
        plt.plot(
            self.metrics_smote['fpr'], 
            self.metrics_smote['tpr'], 
            label=f"SMOTE (AUC = {self.metrics_smote['roc_auc']:.3f})",
            color='red', 
            linestyle='-'
        )
        
        # Plot SMOTE Model (Calibrated)
        plt.plot(
            self.metrics_smote_calibrated['fpr'], 
            self.metrics_smote_calibrated['tpr'], 
            label=f"SMOTE Calibrated (AUC = {self.metrics_smote_calibrated['roc_auc']:.3f})",
            color='red', 
            linestyle='--'
        )
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Classifier')
        
        # Add labels and title
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Comparison: Original vs SMOTE, Calibrated vs Uncalibrated', fontsize=14)
        
        # Configure grid and legend
        plt.grid(alpha=0.3)
        plt.legend(loc='lower right')
        
        # Set axes limits
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        
        # Save figure
        roc_fig_path = os.path.join(self.results_dir, f"roc_comparison_{self.timestamp}.png")
        plt.savefig(roc_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curves comparison saved to {roc_fig_path}")
        
        return self

    def plot_loso_roc_curves(self):
        """
        Plot ROC curves for the Leave-One-Subject-Out cross-validation results.
        This plots both the pooled ROC curve and individual ROC curves for each participant.
        """
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Test participants
        participants = list(self.participant_aucs.keys())
        
        # Set colormap for participant curves
        cmap = plt.cm.get_cmap('plasma', len(participants))
        
        # Plot individual participant ROC curves (uncalibrated)
        for i, participant in enumerate(participants):
            # Find indices for this participant's test data
            participant_indices = []
            for j, group in enumerate(self.groups):
                if group == participant:
                    participant_indices.append(j)
            
            # Extract this participant's test data and predictions
            participant_indices = [j for j, p in enumerate(self.y_test_pooled) 
                                if j < len(self.y_probs_pooled) and 
                                self.groups[j % len(self.groups)] == participant]
            
            if participant_indices:
                y_test_participant = [self.y_test_pooled[j] for j in participant_indices]
                y_probs_participant = [self.y_probs_pooled[j] for j in participant_indices]
                
                # Only plot if there are enough samples with both classes
                if len(np.unique(y_test_participant)) > 1 and not np.isnan(self.participant_aucs[participant]):
                    fpr, tpr, _ = roc_curve(y_test_participant, y_probs_participant)
                    
                    # Plot with low alpha for individual participants
                    plt.plot(fpr, tpr, color=cmap(i), alpha=0.3, linestyle='-',
                            label=f"P{participant}: AUC={self.participant_aucs[participant]:.3f}" if i < 10 else "")
        
        # Plot pooled ROC curve (uncalibrated) with thicker line
        fpr_pooled, tpr_pooled, _ = roc_curve(self.y_test_pooled, self.y_probs_pooled)
        plt.plot(fpr_pooled, tpr_pooled, 'b-', linewidth=3, 
                label=f"Pooled (Uncalibrated): AUC={self.roc_auc_pooled:.3f}")
        
        # Plot pooled ROC curve (calibrated) with thicker line
        fpr_pooled_calib, tpr_pooled_calib, _ = roc_curve(
            self.y_test_pooled_calibrated, self.y_probs_pooled_calibrated
        )
        plt.plot(fpr_pooled_calib, tpr_pooled_calib, 'r-', linewidth=3, 
                label=f"Pooled (Calibrated): AUC={self.roc_auc_pooled_calibrated:.3f}")
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.8)
        
        # Set plot properties
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('LOSO Cross-Validation: ROC Curves by Participant', fontsize=16)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.grid(alpha=0.3)
        
        # Create a legend with only the first few participants 
        legend = plt.legend(loc='lower right', fontsize=10)
        
        # Save the plot
        roc_loso_path = os.path.join(self.results_dir, f"roc_loso_curves_{self.timestamp}.png")
        plt.savefig(roc_loso_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"LOSO ROC curves saved to {roc_loso_path}")
        
        # Plot accuracy vs threshold curve for the pooled results
        self.plot_threshold_analysis()
        
        return self

    def plot_threshold_analysis(self):
        """
        Plot accuracy and balanced accuracy vs threshold to help with threshold selection.
        """
        plt.figure(figsize=(12, 8))
        
        # Get pooled predictions and true labels
        y_true = np.array(self.y_test_pooled)
        y_scores = np.array(self.y_probs_pooled)
        
        # Calibrated results
        y_true_calib = np.array(self.y_test_pooled_calibrated)  # Should be the same as y_true
        y_scores_calib = np.array(self.y_probs_pooled_calibrated)
        
        # Calculate metrics at different thresholds
        thresholds = np.linspace(0, 1, 100)
        accuracy = []
        balanced_acc = []
        f1_scores = []
        
        accuracy_calib = []
        balanced_acc_calib = []
        f1_scores_calib = []
        
        for threshold in thresholds:
            # Uncalibrated
            y_pred = (y_scores >= threshold).astype(int)
            accuracy.append(accuracy_score(y_true, y_pred))
            balanced_acc.append(balanced_accuracy_score(y_true, y_pred))
            f1_scores.append(f1_score(y_true, y_pred))
            
            # Calibrated
            y_pred_calib = (y_scores_calib >= threshold).astype(int)
            accuracy_calib.append(accuracy_score(y_true_calib, y_pred_calib))
            balanced_acc_calib.append(balanced_accuracy_score(y_true_calib, y_pred_calib))
            f1_scores_calib.append(f1_score(y_true_calib, y_pred_calib))
        
        # Find optimal thresholds
        best_f1_idx = np.argmax(f1_scores)
        best_bal_acc_idx = np.argmax(balanced_acc)
        
        best_f1_idx_calib = np.argmax(f1_scores_calib)
        best_bal_acc_idx_calib = np.argmax(balanced_acc_calib)
        
        # Plot uncalibrated metrics
        plt.plot(thresholds, accuracy, 'b-', label='Accuracy (Uncalibrated)')
        plt.plot(thresholds, balanced_acc, 'g-', label='Balanced Accuracy (Uncalibrated)')
        plt.plot(thresholds, f1_scores, 'r-', label='F1 Score (Uncalibrated)')
        
        # Plot calibrated metrics
        plt.plot(thresholds, accuracy_calib, 'b--', label='Accuracy (Calibrated)')
        plt.plot(thresholds, balanced_acc_calib, 'g--', label='Balanced Accuracy (Calibrated)')
        plt.plot(thresholds, f1_scores_calib, 'r--', label='F1 Score (Calibrated)')
        
        # Mark optimal thresholds
        plt.axvline(x=thresholds[best_f1_idx], color='r', linestyle=':', alpha=0.7,
                    label=f'Best F1 Threshold (Uncalib) = {thresholds[best_f1_idx]:.2f}')
        plt.axvline(x=thresholds[best_bal_acc_idx], color='g', linestyle=':', alpha=0.7,
                    label=f'Best Bal Acc Threshold (Uncalib) = {thresholds[best_bal_acc_idx]:.2f}')
        
        plt.axvline(x=thresholds[best_f1_idx_calib], color='r', linestyle='-.', alpha=0.7,
                    label=f'Best F1 Threshold (Calib) = {thresholds[best_f1_idx_calib]:.2f}')
        plt.axvline(x=thresholds[best_bal_acc_idx_calib], color='g', linestyle='-.', alpha=0.7,
                    label=f'Best Bal Acc Threshold (Calib) = {thresholds[best_bal_acc_idx_calib]:.2f}')
        
        # Set plot properties
        plt.xlabel('Threshold', fontsize=14)
        plt.ylabel('Metric Value', fontsize=14)
        plt.title('Metrics vs Classification Threshold', fontsize=16)
        plt.grid(alpha=0.3)
        plt.legend(loc='center right')
        
        # Save the plot
        threshold_path = os.path.join(self.results_dir, f"threshold_analysis_{self.timestamp}.png")
        plt.savefig(threshold_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Threshold analysis saved to {threshold_path}")
        
        # Store the optimal thresholds
        self.optimal_thresholds = {
            'uncalibrated': {
                'f1': thresholds[best_f1_idx],
                'balanced_accuracy': thresholds[best_bal_acc_idx]
            },
            'calibrated': {
                'f1': thresholds[best_f1_idx_calib],
                'balanced_accuracy': thresholds[best_bal_acc_idx_calib]
            }
        }
        
        return self

    def plot_enhanced_roc_comparison(self):
        """
        Plot comprehensive ROC curve comparison with confidence intervals.
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Regular ROC Comparison
        ax1 = axes[0]
        
        # Plot Original Model (Uncalibrated)
        ax1.plot(
            self.metrics_orig['fpr'], 
            self.metrics_orig['tpr'], 
            label=f"Original (AUC = {self.metrics_orig['roc_auc']:.3f})",
            color='blue', 
            linestyle='-',
            linewidth=2
        )
        
        # Plot Original Model (Calibrated)
        ax1.plot(
            self.metrics_orig_calibrated['fpr'], 
            self.metrics_orig_calibrated['tpr'], 
            label=f"Original Calibrated (AUC = {self.metrics_orig_calibrated['roc_auc']:.3f})",
            color='skyblue', 
            linestyle='-',
            linewidth=2
        )
        
        # Plot SMOTE Model (Uncalibrated)
        ax1.plot(
            self.metrics_smote['fpr'], 
            self.metrics_smote['tpr'], 
            label=f"SMOTE (AUC = {self.metrics_smote['roc_auc']:.3f})",
            color='red', 
            linestyle='-',
            linewidth=2
        )
        
        # Plot SMOTE Model (Calibrated)
        ax1.plot(
            self.metrics_smote_calibrated['fpr'], 
            self.metrics_smote_calibrated['tpr'], 
            label=f"SMOTE Calibrated (AUC = {self.metrics_smote_calibrated['roc_auc']:.3f})",
            color='salmon', 
            linestyle='-',
            linewidth=2
        )
        
        # Plot diagonal line (random classifier)
        ax1.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Classifier')
        
        # Add labels and title
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curve Comparison: Regular Test Set', fontsize=14)
        
        # Configure grid and legend
        ax1.grid(alpha=0.3)
        ax1.legend(loc='lower right')
        
        # Set axes limits
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1.05])
        
        # Plot 2: LOSO ROC Comparison
        ax2 = axes[1]
        
        # Plot pooled ROC curve (uncalibrated) with thicker line
        fpr_pooled, tpr_pooled, _ = roc_curve(self.y_test_pooled, self.y_probs_pooled)
        ax2.plot(fpr_pooled, tpr_pooled, 'b-', linewidth=2.5, 
                label=f"Pooled (Uncalibrated): AUC={self.roc_auc_pooled:.3f}")
        
        # Plot pooled ROC curve (calibrated) with thicker line
        fpr_pooled_calib, tpr_pooled_calib, _ = roc_curve(
            self.y_test_pooled_calibrated, self.y_probs_pooled_calibrated
        )
        ax2.plot(fpr_pooled_calib, tpr_pooled_calib, 'r-', linewidth=2.5, 
                label=f"Pooled (Calibrated): AUC={self.roc_auc_pooled_calibrated:.3f}")
        
        # Calculate confidence bands for the ROC curves using bootstrapping (if available)
        try:
            from sklearn.utils import resample
            
            # Number of bootstrap samples
            n_bootstraps = 100
            
            # Arrays to store bootstrapped AUCs
            roc_auc_bootstraps = []
            roc_auc_bootstraps_calibrated = []
            
            # Store tpr values at specific fpr points for confidence intervals
            tpr_bootstraps = []
            tpr_bootstraps_calibrated = []
            
            # Fixed FPR points for interpolation
            fixed_fpr_points = np.linspace(0, 1, 100)
            
            # Perform bootstrapping
            for i in range(n_bootstraps):
                # Create bootstrap sample indices
                indices = resample(np.arange(len(self.y_test_pooled)), 
                                replace=True, 
                                n_samples=len(self.y_test_pooled))
                
                # Get bootstrap samples
                y_true_bs = np.array(self.y_test_pooled)[indices]
                y_score_bs = np.array(self.y_probs_pooled)[indices]
                y_score_bs_calib = np.array(self.y_probs_pooled_calibrated)[indices]
                
                # Compute ROC curve
                fpr_bs, tpr_bs, _ = roc_curve(y_true_bs, y_score_bs)
                roc_auc_bs = auc(fpr_bs, tpr_bs)
                roc_auc_bootstraps.append(roc_auc_bs)
                
                # Calibrated
                fpr_bs_calib, tpr_bs_calib, _ = roc_curve(y_true_bs, y_score_bs_calib)
                roc_auc_bs_calib = auc(fpr_bs_calib, tpr_bs_calib)
                roc_auc_bootstraps_calibrated.append(roc_auc_bs_calib)
                
                # Interpolate TPR at fixed FPR points
                tpr_interp = np.interp(fixed_fpr_points, fpr_bs, tpr_bs)
                tpr_bootstraps.append(tpr_interp)
                
                tpr_interp_calib = np.interp(fixed_fpr_points, fpr_bs_calib, tpr_bs_calib)
                tpr_bootstraps_calibrated.append(tpr_interp_calib)
            
            # Convert to numpy arrays
            tpr_bootstraps = np.array(tpr_bootstraps)
            tpr_bootstraps_calibrated = np.array(tpr_bootstraps_calibrated)
            
            # Compute confidence intervals
            tpr_lower = np.percentile(tpr_bootstraps, 2.5, axis=0)
            tpr_upper = np.percentile(tpr_bootstraps, 97.5, axis=0)
            
            tpr_lower_calib = np.percentile(tpr_bootstraps_calibrated, 2.5, axis=0)
            tpr_upper_calib = np.percentile(tpr_bootstraps_calibrated, 97.5, axis=0)
            
            # Plot confidence bands
            ax2.fill_between(fixed_fpr_points, tpr_lower, tpr_upper, color='blue', alpha=0.2,
                            label='95% CI (Uncalibrated)')
            ax2.fill_between(fixed_fpr_points, tpr_lower_calib, tpr_upper_calib, color='red', alpha=0.2,
                            label='95% CI (Calibrated)')
            
            # Calculate 95% CI for AUC
            auc_ci_lower = np.percentile(roc_auc_bootstraps, 2.5)
            auc_ci_upper = np.percentile(roc_auc_bootstraps, 97.5)
            
            auc_ci_lower_calib = np.percentile(roc_auc_bootstraps_calibrated, 2.5)
            auc_ci_upper_calib = np.percentile(roc_auc_bootstraps_calibrated, 97.5)
            
            # Add CI to legend
            ax2.text(0.6, 0.3, f"Uncalibrated AUC 95% CI: [{auc_ci_lower:.3f}, {auc_ci_upper:.3f}]", 
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.8), transform=ax2.transAxes)
            ax2.text(0.6, 0.25, f"Calibrated AUC 95% CI: [{auc_ci_lower_calib:.3f}, {auc_ci_upper_calib:.3f}]", 
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.8), transform=ax2.transAxes)
        
        except ImportError:
            print("Sklearn resample not available. Skipping confidence intervals.")
        except Exception as e:
            print(f"Error calculating confidence intervals: {e}")
        
        # Plot random classifier line
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Random Classifier')
        
        # Set plot properties
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('LOSO Cross-Validation: Pooled ROC Curves', fontsize=14)
        ax2.set_xlim([-0.01, 1.01])
        ax2.set_ylim([-0.01, 1.01])
        ax2.grid(alpha=0.3)
        ax2.legend(loc='lower right', fontsize=10)
        
        # Add a main title for the entire figure
        fig.suptitle('ROC Curve Analysis for Mind Wandering Classification', fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save the plot
        enhanced_roc_path = os.path.join(self.results_dir, f"enhanced_roc_comparison_{self.timestamp}.png")
        plt.savefig(enhanced_roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Enhanced ROC comparison saved to {enhanced_roc_path}")
        
        return self


# Main execution
if __name__ == "__main__":
    file_path = "/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/data_theodora/bhv/2_clean/all_probes4.csv"
    
    classifier = MindWanderingClassifier(random_state=42, results_dir="XGBoostWithStandardization")
    
    # Run pipeline with standardization
    (classifier
        .load_data(file_path)
        .select_features()
        .engineer_features()
        .clean_data()
        .prepare_datasets_with_standardization()
        .optimize_hyperparameters(n_trials=50)
        .train_models()
        .evaluate_models()
        .plot_roc_comparison()
        .perform_loso_cv_with_standardization()  
        .plot_enhanced_roc_comparison()          
        .plot_threshold_analysis()
        .plot_loso_roc_curves()
        
       
    )
    
    print("Analysis with participant-wise standardization complete!")