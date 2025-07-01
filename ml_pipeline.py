# Updated on July 1, 2025
print("Running updated version July 1, 2025")
# Note: SMOTE configurations require the 'imbalanced-learn' package.
# Install with: pip install imbalanced-learn
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample, shuffle
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

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
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier

        steps = []
        
        # Add standardization
        steps.append(('scaler', StandardScaler()))

        # Add SMOTE if requested
        if self.use_smote:
            try:
                from imblearn.pipeline import Pipeline as ImbPipeline
                from imblearn.over_sampling import SMOTE
            except ImportError:
                raise ImportError("imblearn is not installed. Please install it using 'pip install imbalanced-learn' to use SMOTE.")
            steps.append(('smote', SMOTE(random_state=42)))
            pipeline_class = ImbPipeline
        else:
            pipeline_class = Pipeline

        # Add feature selection
        steps.append(('feature_selection', SelectKBest(f_classif, k=self.n_features)))

        # Add the model
        if self.model_type == 'rf':
            steps.append(('model', RandomForestClassifier(random_state=42)))
        elif self.model_type == 'xgb':
            steps.append(('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)))
        else:
            raise ValueError("model_type must be 'rf' or 'xgb'")

        return pipeline_class(steps)

    
    def _get_cv_splitter(self, n_splits=5):
        if self.cv_type == 'grouped':
            return GroupKFold(n_splits=n_splits)
        elif self.cv_type == 'stratified_grouped':
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif self.cv_type == 'loso':
            return LeaveOneGroupOut()
        else:
            raise ValueError("cv_type must be 'grouped', 'stratified_grouped', or 'loso'")
    
    def fit(self, X, y, groups, n_splits=5):
        """
        Fit the model using cross_validate and extract results.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        groups : array-like of shape (n_samples,)
        n_splits : int
            Number of splits for cross-validation (ignored if cv_type='loso')

        Returns:
        --------
        self : object
        """
        # Get cross-validation splitter
        cv = self._get_cv_splitter(n_splits)
        
        # Create pipeline
        pipeline = self._create_pipeline()
        
        # Define scoring method
        scoring = 'roc_auc' if self.metric == 'auc' else 'balanced_accuracy'
        
        # Apply cross-validation
        results = cross_validate(
            pipeline,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=scoring,
            return_estimator=True,
            n_jobs=-1
        )
        
        self.fold_metrics_ = results['test_score']
        self.mean_metric_ = np.mean(self.fold_metrics_)
        
        self.fold_feature_importances_ = []
        self.feature_names_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"Feature {i}" for i in range(X.shape[1])]
        
        for est in results['estimator']:
            selector = est.named_steps['feature_selection']
            model = est.named_steps['model']
            
            selected_indices = selector.get_support(indices=True)
            all_importances = np.zeros(len(self.feature_names_))
            all_importances[selected_indices] = model.feature_importances_
            self.fold_feature_importances_.append(all_importances)
        
        self.feature_importances_ = np.mean(self.fold_feature_importances_, axis=0)
        
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
        plt.title(f"Top {top_n} Feature Importances - {self.model_type.upper()} | SMOTE: {self.use_smote} | CV: {self.cv_type} | Metric: {self.metric}")
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

def save_results_by_config(config, mean_metric, fold_metrics, feature_importances, feature_names, output_dir='.'):
    # Generate config-specific folder
    config_str = (
        f"{config['model_type']}_"
        f"{'smote' if config['use_smote'] else 'nosmote'}_"
        f"{config['cv_type']}_"
        f"{config['metric']}"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dir = os.path.join(output_dir, f"{config_str}_{timestamp}")
    os.makedirs(config_dir, exist_ok=True)

    # Save metrics
    metrics = {
        'mean_metric': float(mean_metric),
        'metric_name': "AUC" if config['metric'] == 'auc' else "Balanced Accuracy",
        'fold_metrics': [float(m) for m in fold_metrics],
        'config': config
    }

    metrics_file = os.path.join(config_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Feature importances
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(feature_importances))]

    feature_imp = {
        feature_names[i]: float(feature_importances[i])
        for i in range(len(feature_importances))
    }
    feature_imp = dict(sorted(feature_imp.items(), key=lambda item: item[1], reverse=True))

    imp_file = os.path.join(config_dir, "feature_importances.json")
    with open(imp_file, 'w') as f:
        json.dump(feature_imp, f, indent=4)

    # Feature importance plot
    top_n = min(10, len(feature_imp))
    top_features = list(feature_imp.items())[:top_n]
    top_names = [name for name, _ in top_features]
    top_vals = [val for _, val in top_features]

    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances - {config_str}")
    plt.barh(range(top_n), top_vals[::-1])
    plt.yticks(range(top_n), top_names[::-1])
    plt.xlabel("Importance")
    plt.tight_layout()

    plot_file = os.path.join(config_dir, "feature_importance_plot.png")
    plt.savefig(plot_file)
    plt.close()

    return config_dir, metrics_file, imp_file, plot_file

def save_overall_confusion_matrix(config, all_y_true, all_y_pred, output_dir='.'):
    # Create confusion matrix and save to the specified output directory
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    labels = ['OFF', 'ON']

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - All Folds Combined")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "confusion_matrix_overall.png")
    plt.savefig(plot_file)
    plt.close()

    return plot_file

def run_permutation_test(pipeline, X, y, groups, n_runs=20, cv_splits=5, metric='auc'):
    """
    Run permutation and bootstrapping test on a fitted pipeline to compare AUC distributions.
    
    Parameters:
    -----------
    pipeline : MLPipeline
        An instance of the MLPipeline class with initialized parameters.
    X : pd.DataFrame
        Feature matrix.
    y : np.array or pd.Series
        Target labels.
    groups : np.array or pd.Series
        Group labels for cross-validation.
    n_runs : int
        Number of runs for both bootstrap and permutation tests.
    cv_splits : int
        Number of splits for grouped CV.
    metric : str
        Scoring metric, should be 'auc' or compatible with sklearn metrics.
    """
    from sklearn.base import clone

    real_auc_scores = []
    shuffled_auc_scores = []

    cv = pipeline._get_cv_splitter(n_splits=cv_splits)
    base_clf = pipeline._create_pipeline()

    for _ in range(n_runs):
        # Bootstrapped AUC
        idx = resample(np.arange(len(X)), replace=True)
        X_boot, y_boot, g_boot = X.iloc[idx], y[idx], groups[idx]
        aucs = []
        for train_idx, test_idx in cv.split(X_boot, y_boot, g_boot):
            clf = clone(base_clf)
            clf.fit(X_boot.iloc[train_idx], y_boot[train_idx])
            y_pred = clf.predict_proba(X_boot.iloc[test_idx])[:, 1]
            aucs.append(roc_auc_score(y_boot[test_idx], y_pred))
        real_auc_scores.append(np.mean(aucs))

        # Permuted AUC
        y_perm = shuffle(y, random_state=None)
        aucs = []
        for train_idx, test_idx in cv.split(X, y_perm, groups):
            clf = clone(base_clf)
            clf.fit(X.iloc[train_idx], y_perm[train_idx])
            y_pred = clf.predict_proba(X.iloc[test_idx])[:, 1]
            aucs.append(roc_auc_score(y_perm[test_idx], y_pred))
        shuffled_auc_scores.append(np.mean(aucs))

    # Mann-Whitney U test
    stat, p_value = mannwhitneyu(real_auc_scores, shuffled_auc_scores, alternative='greater')

    return {
        'real_auc_scores': real_auc_scores,
        'shuffled_auc_scores': shuffled_auc_scores,
        'p_value': p_value
    }

def save_permutation_results(config, perm_test_results, output_dir='.'):
    """
    Save permutation test results (AUCs and p-value) to JSON and PNG.

    Parameters:
    -----------
    config : dict
        The configuration dictionary used for the model run.
    perm_test_results : dict
        Dictionary containing 'real_auc_scores', 'shuffled_auc_scores', and 'p_value'.
    output_dir : str
        Directory where results should be saved (this should be the config-specific folder).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_data = {
        "mean_auc_real": float(np.mean(perm_test_results['real_auc_scores'])),
        "mean_auc_shuffled": float(np.mean(perm_test_results['shuffled_auc_scores'])),
        "p_value": float(perm_test_results['p_value']),
        "real_auc_scores": [float(a) for a in perm_test_results['real_auc_scores']],
        "shuffled_auc_scores": [float(a) for a in perm_test_results['shuffled_auc_scores']],
        "config": config
    }

    json_file = os.path.join(output_dir, "permutation_results.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=4)

    # Plot AUC distributions and save PNG
    plt.figure(figsize=(10, 6))
    sns.histplot(perm_test_results['real_auc_scores'], bins=10, kde=True, color='blue', label='True Labels (Bootstrapped)', stat='count')
    sns.histplot(perm_test_results['shuffled_auc_scores'], bins=10, kde=True, color='orange', label='Shuffled Labels (Permutation)', stat='count')
    plt.axvline(np.mean(perm_test_results['real_auc_scores']), color='blue', linestyle='--', label=f"Mean AUC (Real): {np.mean(perm_test_results['real_auc_scores']):.3f}")
    plt.axvline(np.mean(perm_test_results['shuffled_auc_scores']), color='orange', linestyle='--', label=f"Mean AUC (Shuffled): {np.mean(perm_test_results['shuffled_auc_scores']):.3f}")
    plt.title(f'AUC Distributions ({len(perm_test_results["real_auc_scores"])} runs) | Mann-Whitney p = {perm_test_results["p_value"]:.4f}')
    plt.xlabel("AUC")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "permutation_auc_plot.png")
    plt.savefig(plot_file)
    plt.close()

    return json_file, plot_file

if __name__ == "__main__":
    # Create output directory
    output_dir = "pipeline_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load your data
    df = pd.read_csv("/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/big_window_combined_behavior_eye_features_labeled3.csv")
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

    le = LabelEncoder()
    y = le.fit_transform(df['on_off'])
    X = X.fillna(0) 

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
            n_features=20,
            metric=config['metric']
        )
        
        try:
            # Fit the pipeline
            pipeline.fit(X, y, groups=groups)
            cv = pipeline._get_cv_splitter(n_splits=5)
            clf = pipeline._create_pipeline()

            y_true_all = []
            y_pred_all = []

            for train_idx, test_idx in cv.split(X, y, groups):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)

                if not config['use_smote'] and config['model_type'] in ['xgb', 'rf']:
                    try:
                        model = clf.named_steps['model']
                        selector = clf.named_steps['feature_selection']
                        selected_features = X.columns[selector.get_support()]
                        explainer = shap.Explainer(model, X_train[selected_features])
                        shap_values = explainer(X_test[selected_features])

                        # SHAP summary plot
                        shap.summary_plot(shap_values, X_test[selected_features], show=False)
                        shap_plot_path = os.path.join(
                            output_dir, 
                            f"shap_summary_{config['model_type']}_{config['cv_type']}_{i}.png"
                        )
                        plt.savefig(shap_plot_path)
                        plt.close()
                        print(f"Saved SHAP summary to: {shap_plot_path}")
                    except Exception as shap_err:
                        print(f"SHAP error: {shap_err}")

            # Save all results into one consistent folder
            config_dir, _, _, _ = save_results_by_config(
                config=config,
                mean_metric=pipeline.mean_metric_,
                fold_metrics=pipeline.fold_metrics_,
                feature_importances=pipeline.feature_importances_,
                feature_names=pipeline.feature_names_,
                output_dir=output_dir
            )

            # Save confusion matrix to the config folder
            save_overall_confusion_matrix(
                config=config,
                all_y_true=np.array(y_true_all),
                all_y_pred=np.array(y_pred_all),
                output_dir=config_dir
            )

            # Run permutation test
            perm_test_results = run_permutation_test(
                pipeline=pipeline,
                X=X,
                y=y,
                groups=groups,
                n_runs=20,
                cv_splits=5,
                metric=config['metric']
            )

            # Save permutation test results to same config folder
            save_permutation_results(
                config=config,
                perm_test_results=perm_test_results,
                output_dir=config_dir
            )

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
            print(f"Error running configuration {config}: {str(e)}")
            continue
    
    # Compare all results
    print("\n" + "="*50)
    print("SUMMARY OF ALL CONFIGURATIONS")
    print("="*50)
    
    if all_results:
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mean_metric'], reverse=True)
        
        print(f"\nBest configuration: {sorted_results[0][0]}")
        for i, (config_name, result) in enumerate(sorted_results):
            print(f"{i+1}. {config_name}: {result['mean_metric']:.4f}")
            print(f"   Fold metrics: {[f'{m:.4f}' for m in result['fold_metrics']]}")
        
        print(f"\nBest performance: {sorted_results[0][1]['mean_metric']:.4f}")
        print("\nResults saved to:", output_dir)
    else:
        print("\nNo successful configurations.")