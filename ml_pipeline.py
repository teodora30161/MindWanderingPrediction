import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

class MLPipeline:
    def __init__(self, 
                 use_smote=False, 
                 model_type='rf', 
                 cv_type='grouped', 
                 n_features=10,
                 metric='auc'):

        self.use_smote = use_smote
        self.model_type = model_type
        self.cv_type = cv_type
        self.n_features = n_features
        self.metric = metric
        self.feature_importances_ = None
        self.mean_metric_ = None
        self.fold_metrics_ = []
        self.fold_feature_importances_ = []
        
    def _create_pipeline(self):
        steps = []
        
        # Add standardization
        steps.append(('scaler', StandardScaler()))
        
        # Add SMOTE if requested
        if self.use_smote:
            steps.append(('smote', SMOTE(random_state=42)))
        
        # Add feature selection
        steps.append(('feature_selection', SelectKBest(f_classif, k=self.n_features)))
        
        # Add the model
        if self.model_type == 'rf':
            steps.append(('model', RandomForestClassifier(random_state=42)))
        elif self.model_type == 'xgb':
            steps.append(('model', XGBClassifier(random_state=42)))
        else:
            raise ValueError("model_type must be 'rf' or 'xgb'")
        
        # Create the pipeline
        if self.use_smote:
            return ImbPipeline(steps)
        else:
            return Pipeline(steps)
    
    def _get_cv_splitter(self, n_splits=10):
        if self.cv_type == 'grouped':
            return GroupKFold(n_splits=n_splits)
        elif self.cv_type == 'stratified_grouped':
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif self.cv_type == 'loso':
            return LeaveOneGroupOut()
        else:
            raise ValueError("cv_type must be 'grouped', 'stratified_grouped', or 'loso'")
    
    def fit(self, X, y, groups, n_splits=10):
        """
        Fit the model using cross-validation
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples
        y : array-like of shape (n_samples,)
            The target values
        groups : array-like of shape (n_samples,)
            Group labels for the samples
        n_splits : int, default=5
            Number of splits for cross-validation (ignored if cv_type='loso')
        
        Returns:
        --------
        self : object
            Returns self
        """
        # Get CV splitter
        cv = self._get_cv_splitter(n_splits)
        
        # Initialize lists to store results
        self.fold_metrics_ = []
        self.fold_feature_importances_ = []
        feature_names = None
        
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        
        # Perform cross-validation
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            print(f"Training fold {i+1}...")
            
            # Get train and test data
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
                
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create and fit the pipeline
            pipeline = self._create_pipeline()
            pipeline.fit(X_train, y_train)
            
            if self.metric == 'auc':
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                fold_metric = roc_auc_score(y_test, y_pred_proba)
            else:  # balanced_accuracy
                y_pred = pipeline.predict(X_test)
                fold_metric = balanced_accuracy_score(y_test, y_pred)
            
            self.fold_metrics_.append(fold_metric)
            
            # Get feature importances
            if self.model_type == 'rf':
                model = pipeline.named_steps['model']
                selector = pipeline.named_steps['feature_selection']
                
                # Get indices of selected features
                selected_indices = selector.get_support(indices=True)
                
                # Initialize importances array with zeros
                all_importances = np.zeros(X.shape[1])
                
                # Assign importances only to selected features
                all_importances[selected_indices] = model.feature_importances_
                
                self.fold_feature_importances_.append(all_importances)
            elif self.model_type == 'xgb':
                model = pipeline.named_steps['model']
                selector = pipeline.named_steps['feature_selection']
                
                # Get indices of selected features
                selected_indices = selector.get_support(indices=True)
                
                # Initialize importances array with zeros
                all_importances = np.zeros(X.shape[1])
                
                # Assign importances only to selected features
                all_importances[selected_indices] = model.feature_importances_
                
                self.fold_feature_importances_.append(all_importances)
        
        # Calculate mean metric and feature importances
        self.mean_metric_ = np.mean(self.fold_metrics_)
        self.feature_importances_ = np.mean(self.fold_feature_importances_, axis=0)
        
        # Store feature names 
        self.feature_names_ = feature_names
        
        return self
    
    def plot_feature_importances(self, top_n=10):
        """
        Plot the top N feature importances
        
        Parameters:
        -----------
        top_n : int, default=10
            Number of top features to plot
        """
        if self.feature_importances_ is None:
            raise ValueError("Model has not been fitted yet")
        
        # Get feature names
        if self.feature_names_ is not None:
            features = self.feature_names_
        else:
            features = [f"Feature {i}" for i in range(len(self.feature_importances_))]
        
        # Sort features by importance
        indices = np.argsort(self.feature_importances_)[-top_n:]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.title(f"Top {top_n} Feature Importances")
        plt.barh(range(top_n), self.feature_importances_[indices])
        plt.yticks(range(top_n), [features[i] for i in indices])
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
    
    def get_results(self):
        """
        Get a summary of the results
        
        Returns:
        --------
        dict : Dictionary containing the results
        """
        if self.mean_metric_ is None:
            raise ValueError("Model has not been fitted yet")
        
        metric_name = "AUC" if self.metric == 'auc' else "Balanced Accuracy"
        
        results = {
            'mean_metric': self.mean_metric_,
            'metric_name': metric_name,
            'fold_metrics': self.fold_metrics_,
            'feature_importances': self.feature_importances_
        }
        
        if self.feature_names_ is not None:
            results['feature_names'] = self.feature_names_
        
        return results
    
    def save_results(self, output_dir='.'):
        """
        Save the results to files
        
        Parameters:
        -----------
        output_dir : str, default='.'
            Directory to save the results to
        """
        import os
        import json
        from datetime import datetime
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for the filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Config string for filenames
        config_str = f"{self.model_type}_{'smote' if self.use_smote else 'nosmote'}_{self.cv_type}_{self.metric}"
        
        # Save metrics
        metrics = {
            'mean_metric': float(self.mean_metric_),
            'metric_name': "AUC" if self.metric == 'auc' else "Balanced Accuracy",
            'fold_metrics': [float(m) for m in self.fold_metrics_],
            'config': {
                'use_smote': self.use_smote,
                'model_type': self.model_type,
                'cv_type': self.cv_type,
                'n_features': self.n_features,
                'metric': self.metric
            }
        }
        
        metrics_file = os.path.join(output_dir, f"metrics_{config_str}_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save feature importances
        feature_imp = {}
        
        if self.feature_names_ is not None:
            features = self.feature_names_
        else:
            features = [f"Feature {i}" for i in range(len(self.feature_importances_))]
        
        for i, feat in enumerate(features):
            feature_imp[feat] = float(self.feature_importances_[i])
        
        # Sort by importance
        feature_imp = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: item[1], reverse=True)}
        
        imp_file = os.path.join(output_dir, f"feature_importances_{config_str}_{timestamp}.json")
        with open(imp_file, 'w') as f:
            json.dump(feature_imp, f, indent=4)
        
        # Save feature importance plot
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances - {config_str}")
        
        # Get top features
        top_n = min(10, len(features))
        indices = np.argsort(self.feature_importances_)[-top_n:][::-1]
        
        plt.barh(range(top_n), [self.feature_importances_[i] for i in indices])
        plt.yticks(range(top_n), [features[i] for i in indices])
        plt.xlabel("Importance")
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, f"feature_importances_{config_str}_{timestamp}.png")
        plt.savefig(plot_file)
        plt.close()
        
        print(f"Results saved to {output_dir}:")
        print(f"  - Metrics: {os.path.basename(metrics_file)}")
        print(f"  - Feature importances: {os.path.basename(imp_file)}")
        print(f"  - Feature importance plot: {os.path.basename(plot_file)}")


if __name__ == "__main__":
    import os
    
    # Create output directory
    output_dir = "pipeline_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load your data
    df = pd.read_csv("/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/combined_behavior_eye_features_labeled.csv")
    print("Columns in dataset:", df.columns.tolist())

    non_features = [
    'subject_id',
    'subj_orgid',    
    'session_id',
    'block_num',
    'probe_number',
    'on_off'          # target
    ]

    X = df.drop(columns=non_features)
    print("X dtypes:", X.dtypes)  

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df['on_off'])

    from sklearn.preprocessing import LabelEncoder

    group_encoder = LabelEncoder()
    groups = group_encoder.fit_transform(df['subject_id'])

    print("Sample X row:", X.iloc[0].to_dict())
    print("Sample y value:", y[0])
    print("Sample group value (encoded):", groups[0])

    
    # Check for potential group/ID columns to exclude from features
    possible_group_columns = ['participant', 'subject', 'subject_id','session', 'participant_id', 'ID', 'id']
    group_col = None
    exclude_cols = []
    
    for col in possible_group_columns:
        if col in df.columns:
            group_col = col
            exclude_cols.append(col)
            break
    
    other_non_feature_cols = ['timestamp', 'time', 'date', 'index', 'trial', 'trial_num', 'trial_number']
    exclude_cols.extend([col for col in other_non_feature_cols if col in df.columns])
    exclude_cols.append('on_off')  # Add target to excluded columns
    
    
    # Handle grouping variable
    if group_col is None:
        print("No participant/subject ID column found. Using sample index for groups.")
        groups = np.arange(len(X))
    else:
        print(f"Using '{group_col}' as the grouping variable.")
        groups = df[group_col].values
    
    le = LabelEncoder()
    y = le.fit_transform(df['on_off'])
    # Print some information about the data
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Number of unique groups: {len(np.unique(groups))}")
    
    # Define configurations to try
    configurations = [
        # Random Forest configurations
        {'use_smote': True, 'model_type': 'rf', 'cv_type': 'grouped', 'metric': 'auc'},
        {'use_smote': True, 'model_type': 'rf', 'cv_type': 'grouped', 'metric': 'balanced_accuracy'},
        {'use_smote': False, 'model_type': 'rf', 'cv_type': 'grouped', 'metric': 'auc'},
        {'use_smote': False, 'model_type': 'rf', 'cv_type': 'grouped', 'metric': 'balanced_accuracy'},
        
        # XGBoost configurations
        {'use_smote': True, 'model_type': 'xgb', 'cv_type': 'grouped', 'metric': 'auc'},
        {'use_smote': True, 'model_type': 'xgb', 'cv_type': 'grouped', 'metric': 'balanced_accuracy'},
        {'use_smote': False, 'model_type': 'xgb', 'cv_type': 'grouped', 'metric': 'auc'},
        {'use_smote': False, 'model_type': 'xgb', 'cv_type': 'grouped', 'metric': 'balanced_accuracy'},
        
        # try stratified group k-fold
        {'use_smote': True, 'model_type': 'rf', 'cv_type': 'stratified_grouped', 'metric': 'auc'},
        {'use_smote': True, 'model_type': 'xgb', 'cv_type': 'stratified_grouped', 'metric': 'auc'},
    ]
    
    # Results dictionary to track all results
    all_results = {}
    print("Sample X row:", X.iloc[0].to_dict())
    print("Sample y value:", y[0])
    print("Sample group value:", groups[0])
    # Run each configuration
    for i, config in enumerate(configurations):
        print("\n" + "="*50)
        print(f"Running configuration {i+1}/{len(configurations)}:")
        print(f"  Model: {config['model_type'].upper()}")
        print(f"  SMOTE: {'Enabled' if config['use_smote'] else 'Disabled'}")
        print(f"  CV: {config['cv_type']}")
        print(f"  Metric: {config['metric']}")
        print("="*50)
        
        pipeline = MLPipeline(
            use_smote=config['use_smote'],
            model_type=config['model_type'],
            cv_type=config['cv_type'],
            n_features=10,
            metric=config['metric']
        )
        
        try:
            # Fit the pipeline
            pipeline.fit(X, y, groups=groups)
            
            
            # Save results
            pipeline.save_results(output_dir)
            
            # Store results in the dictionary
            config_name = (f"{config['model_type'].upper()}_"
                         f"{'SMOTE' if config['use_smote'] else 'NoSMOTE'}_"
                         f"{config['cv_type']}_"
                         f"{config['metric']}")
            
            all_results[config_name] = {
                'mean_metric': pipeline.mean_metric_,
                'fold_metrics': pipeline.fold_metrics_
            }
        except Exception as e:
            print(f"Error running configuration: {str(e)}")
            sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mean_metric'], reverse=True)
    
    # Compare all results
    print("\n" + "="*50)
    print("SUMMARY OF ALL CONFIGURATIONS")
    print("="*50)
    
    if all_results:
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mean_metric'], reverse=True)

        print("\nBest configuration: " + sorted_results[0][0])
    for i, (config_name, result) in enumerate(sorted_results):
        print(f"{i+1}. {config_name}: {result['mean_metric']:.4f}")
        print(f"   Fold metrics: {[f'{m:.4f}' for m in result['fold_metrics']]}")
    
    print(f"\nBest performance: {sorted_results[0][1]['mean_metric']:.4f}")
    print("\nResults saved to:", output_dir)
else:
    print("\nNo successful configurations.")
