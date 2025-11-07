import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.utils import resample, shuffle
from scipy.stats import mannwhitneyu
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Only Random Forest will be used.")

class WithinParticipantMLPipeline:
    def __init__(self, 
                 use_smote=False, 
                 model_type='rf', 
                 cv_type='kfold',  # Now using kfold instead of grouped
                 n_features=10,
                 metric='auc',
                 cv_splits=5):

        self.use_smote = use_smote
        self.model_type = model_type
        self.cv_type = cv_type
        self.n_features = n_features
        self.metric = metric
        self.cv_splits = cv_splits
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
            try:
                from imblearn.pipeline import Pipeline as ImbPipeline
                from imblearn.over_sampling import SMOTE
                steps.append(('smote', SMOTE(random_state=42)))
                pipeline_class = ImbPipeline
            except ImportError:
                raise ImportError("imblearn is not installed. Please install it using 'pip install imbalanced-learn' to use SMOTE.")
        else:
            pipeline_class = Pipeline

        # Add feature selection
        steps.append(('feature_selection', SelectKBest(f_classif, k=self.n_features)))

        # Add the model
        if self.model_type == 'rf':
            steps.append(('model', RandomForestClassifier(random_state=42)))
        elif self.model_type == 'xgb':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not available. Please install it or use 'rf' model type.")
            steps.append(('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)))
        else:
            raise ValueError("model_type must be 'rf' or 'xgb'")

        return pipeline_class(steps)

    def _get_cv_splitter(self):
        if self.cv_type == 'kfold':
            return KFold(n_splits=self.cv_splits, shuffle=True, random_state=42)
        elif self.cv_type == 'session_split':
            # This will be handled differently in the fit method
            return 'session_split'
        else:
            raise ValueError("cv_type must be 'kfold' or 'session_split'")
    
    def fit(self, X, y, sessions=None):
        """
        Fit the model using cross-validation within a single participant.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sessions : array-like of shape (n_samples,), optional
            Session identifiers for session-based splitting

        Returns:
        --------
        self : object
        """
        # Create pipeline
        pipeline = self._create_pipeline()
        
        # Define scoring method
        scoring = 'roc_auc' if self.metric == 'auc' else 'balanced_accuracy'
        
        # Handle different cross-validation types
        if self.cv_type == 'session_split' and sessions is not None:
            # Custom session-based splitting
            unique_sessions = np.unique(sessions)
            if len(unique_sessions) < 2:
                raise ValueError("Need at least 2 sessions for session-based splitting")
            
            self.fold_metrics_ = []
            self.fold_feature_importances_ = []
            self.feature_names_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"Feature {i}" for i in range(X.shape[1])]
            
            # Train on each session, test on others
            for test_session in unique_sessions:
                train_mask = sessions != test_session
                test_mask = sessions == test_session
                
                if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
                    continue
                
                X_train = X[train_mask] if isinstance(X, np.ndarray) else X.iloc[train_mask]
                X_test = X[test_mask] if isinstance(X, np.ndarray) else X.iloc[test_mask]
                y_train = y[train_mask]
                y_test = y[test_mask]
                
                # Check if we have both classes in train and test
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue
                
                # Fit and predict
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict_proba(X_test)[:, 1]
                
                # Calculate metric
                if self.metric == 'auc':
                    score = roc_auc_score(y_test, y_pred)
                else:
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    score = balanced_accuracy_score(y_test, y_pred_binary)
                
                self.fold_metrics_.append(score)
                
                # Extract feature importances
                selector = pipeline.named_steps['feature_selection']
                model = pipeline.named_steps['model']
                
                selected_indices = selector.get_support(indices=True)
                all_importances = np.zeros(len(self.feature_names_))
                all_importances[selected_indices] = model.feature_importances_
                self.fold_feature_importances_.append(all_importances)
        
        else:
            # Regular K-fold cross-validation
            cv = self._get_cv_splitter()
            
            # Apply cross-validation
            results = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                return_estimator=True,
                n_jobs=1  # Use single job for within-participant to avoid conflicts
            )
            
            self.fold_metrics_ = results['test_score']
            self.fold_feature_importances_ = []
            self.feature_names_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"Feature {i}" for i in range(X.shape[1])]
            
            for est in results['estimator']:
                selector = est.named_steps['feature_selection']
                model = est.named_steps['model']
                
                selected_indices = selector.get_support(indices=True)
                all_importances = np.zeros(len(self.feature_names_))
                all_importances[selected_indices] = model.feature_importances_
                self.fold_feature_importances_.append(all_importances)
        
        if len(self.fold_metrics_) > 0:
            self.mean_metric_ = np.mean(self.fold_metrics_)
            self.feature_importances_ = np.mean(self.fold_feature_importances_, axis=0)
        else:
            self.mean_metric_ = np.nan
            self.feature_importances_ = np.zeros(len(self.feature_names_))
        
        return self

def bootstrap_metric_scores(X, y, pipeline_config, n_bootstrap=100, sessions=None):
    """
    Bootstrap the metric scores by resampling the data with replacement.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    pipeline_config : dict
        Configuration for the ML pipeline
    n_bootstrap : int
        Number of bootstrap iterations
    sessions : array-like, optional
        Session identifiers
    
    Returns:
    --------
    bootstrap_scores : list
        List of bootstrap metric scores
    """
    bootstrap_scores = []
    
    for i in range(n_bootstrap):
        # Bootstrap sampling
        n_samples = len(y)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        X_bootstrap = X.iloc[bootstrap_indices] if isinstance(X, pd.DataFrame) else X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        sessions_bootstrap = sessions[bootstrap_indices] if sessions is not None else None
        
        # Check if we have both classes
        if len(np.unique(y_bootstrap)) < 2:
            continue
        
        # Create and fit pipeline
        pipeline = WithinParticipantMLPipeline(**pipeline_config)
        
        try:
            pipeline.fit(X_bootstrap, y_bootstrap, sessions=sessions_bootstrap)
            if not np.isnan(pipeline.mean_metric_):
                bootstrap_scores.append(pipeline.mean_metric_)
        except Exception:
            continue
    
    return bootstrap_scores

def permutation_test_scores(X, y, pipeline_config, n_permutations=100, sessions=None):
    """
    Perform permutation test by shuffling labels and computing metric scores.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    pipeline_config : dict
        Configuration for the ML pipeline
    n_permutations : int
        Number of permutation iterations
    sessions : array-like, optional
        Session identifiers
    
    Returns:
    --------
    permutation_scores : list
        List of permutation metric scores
    """
    permutation_scores = []
    
    for i in range(n_permutations):
        # Shuffle labels
        y_shuffled = shuffle(y, random_state=i)
        
        # Create and fit pipeline
        pipeline = WithinParticipantMLPipeline(**pipeline_config)
        
        try:
            pipeline.fit(X, y_shuffled, sessions=sessions)
            if not np.isnan(pipeline.mean_metric_):
                permutation_scores.append(pipeline.mean_metric_)
        except Exception:
            continue
    
    return permutation_scores

def plot_permutation_test(bootstrap_scores, permutation_scores, true_score, metric='AUC', 
                         n_runs=400, output_file=None, config_name=''):
    """
    Create a permutation test plot showing the distribution of true vs shuffled labels.
    
    Parameters:
    -----------
    bootstrap_scores : list
        Bootstrap scores from true labels
    permutation_scores : list
        Permutation scores from shuffled labels
    true_score : float
        The actual metric score
    metric : str
        Name of the metric (for labeling)
    n_runs : int
        Number of runs for the title
    output_file : str, optional
        File path to save the plot
    config_name : str
        Configuration name for the title
    """
    # Calculate p-value using Mann-Whitney U test
    try:
        statistic, p_value = mannwhitneyu(bootstrap_scores, permutation_scores, alternative='greater')
    except Exception:
        p_value = np.nan
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot histograms
    plt.hist(permutation_scores, bins=30, alpha=0.7, color='orange', 
             label='Shuffled Labels (Permutation)', density=False, edgecolor='black')
    plt.hist(bootstrap_scores, bins=30, alpha=0.7, color='blue', 
             label='True Labels (Bootstrapped)', density=False, edgecolor='black')
    
    # Add mean lines
    mean_bootstrap = np.mean(bootstrap_scores)
    mean_permutation = np.mean(permutation_scores)
    
    plt.axvline(mean_bootstrap, color='blue', linestyle='--', linewidth=2,
                label=f'Mean {metric} (Real): {mean_bootstrap:.3f}')
    plt.axvline(mean_permutation, color='orange', linestyle='--', linewidth=2,
                label=f'Mean {metric} (Shuffled): {mean_permutation:.3f}')
    
    # Formatting
    plt.xlabel(metric, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # Title with configuration info and p-value
    title = f'{metric} Distributions ({n_runs} runs)'
    if not np.isnan(p_value):
        title += f' | Mann-Whitney p = {p_value:.4f}'
    if config_name:
        title += f'\n{config_name}'
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Legend
    plt.legend(fontsize=12, loc='upper left')
    
    # Grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved permutation test plot: {output_file}")
    
    return p_value

def run_permutation_analysis(df, output_dir='within_participant_results', 
                           n_bootstrap=200, n_permutations=200):
    """
    Run permutation test analysis for each participant and configuration.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing all participants
    output_dir : str
        Directory to save results
    n_bootstrap : int
        Number of bootstrap iterations
    n_permutations : int
        Number of permutation iterations
    """
    # Create output directory for permutation tests
    perm_dir = os.path.join(output_dir, 'permutation_tests')
    os.makedirs(perm_dir, exist_ok=True)
    
    # Define non-feature columns
    non_features = [
        'subject_id',
        'subj_orgid',    
        'session_id',
        'block_num',
        'probe_number',
        'on_off'
    ]
    
    # Prepare data
    X_all = df.drop(columns=non_features)
    X_all = X_all.fillna(0)
    
    le = LabelEncoder()
    y_all = le.fit_transform(df['on_off'])
    
    # Get unique participants
    participants = df['subject_id'].unique()
    print(f"Found {len(participants)} unique participants for permutation testing")
    
    # Define configurations to test (simplified for permutation testing)
    configurations = [
        {'use_smote': False, 'model_type': 'rf', 'cv_type': 'kfold', 'metric': 'auc'},
        {'use_smote': False, 'model_type': 'rf', 'cv_type': 'kfold', 'metric': 'balanced_accuracy'},
    ]
    
    # Store permutation results
    permutation_results = []
    
    # Run permutation tests for each configuration
    for config_idx, config in enumerate(configurations):
        print(f"\n{'='*60}")
        print(f"Running permutation test {config_idx + 1}/{len(configurations)}")
        print(f"Model: {config['model_type'].upper()}, Metric: {config['metric']}")
        print(f"{'='*60}")
        
        config_name = f"{config['model_type']}_{config['cv_type']}_{'smote' if config['use_smote'] else 'nosmote'}_{config['metric']}"
        
        # Collect scores across all participants for this configuration
        all_bootstrap_scores = []
        all_permutation_scores = []
        participant_results = []
        
        # Run for each participant
        for participant_idx, participant in enumerate(participants):
            print(f"\nProcessing participant {participant_idx + 1}/{len(participants)}: {participant}")
            
            # Get participant data
            participant_data = df[df['subject_id'] == participant].copy()
            
            # Skip if not enough data
            if len(participant_data) < 10:
                print(f"  Skipping - insufficient data ({len(participant_data)} samples)")
                continue
            
            # Check if we have both classes
            unique_classes = participant_data['on_off'].unique()
            if len(unique_classes) < 2:
                print(f"  Skipping - only one class present")
                continue
            
            # Prepare participant-specific data
            X_participant = participant_data.drop(columns=non_features).fillna(0)
            y_participant = le.fit_transform(participant_data['on_off'])
            sessions_participant = participant_data['session_id'].values if 'session_id' in participant_data.columns else None
            
            # Adjust pipeline config
            pipeline_config = config.copy()
            pipeline_config['n_features'] = min(20, X_participant.shape[1])
            
            try:
                # Get true score first
                true_pipeline = WithinParticipantMLPipeline(**pipeline_config)
                true_pipeline.fit(X_participant, y_participant, sessions=sessions_participant)
                true_score = true_pipeline.mean_metric_
                
                if np.isnan(true_score):
                    print(f"  Skipping - invalid true score")
                    continue
                
                print(f"  True {config['metric'].upper()}: {true_score:.4f}")
                
                # Bootstrap scores (true labels)
                print(f"  Running {n_bootstrap} bootstrap iterations...")
                bootstrap_scores = bootstrap_metric_scores(
                    X_participant, y_participant, pipeline_config, 
                    n_bootstrap=n_bootstrap, sessions=sessions_participant
                )
                
                # Permutation scores (shuffled labels)
                print(f"  Running {n_permutations} permutation iterations...")
                permutation_scores = permutation_test_scores(
                    X_participant, y_participant, pipeline_config,
                    n_permutations=n_permutations, sessions=sessions_participant
                )
                
                if len(bootstrap_scores) == 0 or len(permutation_scores) == 0:
                    print(f"  Skipping - insufficient valid scores")
                    continue
                
                # Store results
                participant_result = {
                    'participant_id': participant,
                    'config': config,
                    'true_score': true_score,
                    'bootstrap_scores': bootstrap_scores,
                    'permutation_scores': permutation_scores,
                    'n_bootstrap': len(bootstrap_scores),
                    'n_permutations': len(permutation_scores)
                }
                participant_results.append(participant_result)
                
                # Add to overall collections
                all_bootstrap_scores.extend(bootstrap_scores)
                all_permutation_scores.extend(permutation_scores)
                
                # Create individual participant plot
                participant_plot_file = os.path.join(
                    perm_dir, f"permutation_{config_name}_{participant}.png"
                )
                p_value = plot_permutation_test(
                    bootstrap_scores, permutation_scores, true_score,
                    metric=config['metric'].upper(),
                    n_runs=n_bootstrap + n_permutations,
                    output_file=participant_plot_file,
                    config_name=f"Participant: {participant}"
                )
                
                print(f"  Bootstrap mean: {np.mean(bootstrap_scores):.4f}")
                print(f"  Permutation mean: {np.mean(permutation_scores):.4f}")
                print(f"  p-value: {p_value:.4f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                continue
        
        # Create overall plot for this configuration
        if all_bootstrap_scores and all_permutation_scores:
            overall_plot_file = os.path.join(
                perm_dir, f"permutation_overall_{config_name}.png"
            )
            overall_p_value = plot_permutation_test(
                all_bootstrap_scores, all_permutation_scores,
                np.mean([r['true_score'] for r in participant_results]),
                metric=config['metric'].upper(),
                n_runs=len(all_bootstrap_scores) + len(all_permutation_scores),
                output_file=overall_plot_file,
                config_name=f"All Participants ({len(participant_results)} participants)"
            )
            
            print(f"\nOverall Results for {config_name}:")
            print(f"  Participants: {len(participant_results)}")
            print(f"  Bootstrap samples: {len(all_bootstrap_scores)}")
            print(f"  Permutation samples: {len(all_permutation_scores)}")
            print(f"  Overall p-value: {overall_p_value:.4f}")
            
            # Save results
            config_result = {
                'config': config,
                'config_name': config_name,
                'n_participants': len(participant_results),
                'overall_bootstrap_scores': all_bootstrap_scores,
                'overall_permutation_scores': all_permutation_scores,
                'overall_p_value': float(overall_p_value) if not np.isnan(overall_p_value) else None,
                'participant_results': participant_results
            }
            permutation_results.append(config_result)
    
    # Save all permutation results
    results_file = os.path.join(perm_dir, "permutation_test_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for result in permutation_results:
        json_result = {
            'config': result['config'],
            'config_name': result['config_name'],
            'n_participants': result['n_participants'],
            'overall_p_value': result['overall_p_value'],
            'participant_summary': []
        }
        
        for p_result in result['participant_results']:
            p_summary = {
                'participant_id': p_result['participant_id'],
                'true_score': float(p_result['true_score']),
                'bootstrap_mean': float(np.mean(p_result['bootstrap_scores'])),
                'permutation_mean': float(np.mean(p_result['permutation_scores'])),
                'n_bootstrap': p_result['n_bootstrap'],
                'n_permutations': p_result['n_permutations']
            }
            json_result['participant_summary'].append(p_summary)
        
        json_results.append(json_result)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"\nPermutation test results saved to: {results_file}")
    return permutation_results

def save_feature_importances(all_results, output_dir='within_participant_results'):
    """
    Save feature importances for each participant and configuration.
    """
    # Create feature importances directory
    feature_dir = os.path.join(output_dir, 'feature_importances')
    os.makedirs(feature_dir, exist_ok=True)
    
    # Group results by configuration
    config_groups = {}
    for result in all_results:
        config = result['config']
        config_str = f"{config['model_type']}_{config['cv_type']}_{'smote' if config['use_smote'] else 'nosmote'}_{config['metric']}"
        
        if config_str not in config_groups:
            config_groups[config_str] = []
        config_groups[config_str].append(result)
    
    # Save for each configuration
    for config_str, config_results in config_groups.items():
        if not config_results:
            continue
        
        # Create DataFrame with feature importances for all participants
        feature_names = config_results[0]['feature_names']
        importance_data = []
        
        for result in config_results:
            if not np.isnan(result['mean_metric']):
                row = {
                    'participant_id': result['participant_id'],
                    'mean_metric': result['mean_metric'],
                    'n_samples': result['n_samples'],
                    'n_features': result['n_features']
                }
                # Add feature importances
                for i, feature_name in enumerate(feature_names):
                    row[feature_name] = result['feature_importances'][i]
                
                importance_data.append(row)
        
        if importance_data:
            # Create DataFrame
            importance_df = pd.DataFrame(importance_data)
            
            # Save to CSV
            csv_file = os.path.join(feature_dir, f"feature_importances_{config_str}.csv")
            importance_df.to_csv(csv_file, index=False)
            print(f"Saved feature importances: {csv_file}")
            
            # Save summary statistics
            feature_columns = [col for col in importance_df.columns if col not in ['participant_id', 'mean_metric', 'n_samples', 'n_features']]
            summary_stats = importance_df[feature_columns].describe()
            
            summary_file = os.path.join(feature_dir, f"feature_importance_summary_{config_str}.csv")
            summary_stats.to_csv(summary_file)
            print(f"Saved feature importance summary: {summary_file}")

def plot_feature_importance_distributions(all_results, output_dir='within_participant_results', top_n=20):
    """
    Plot boxplots of feature importance distributions across participants.
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'feature_importance_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Group results by configuration
    config_groups = {}
    for result in all_results:
        config = result['config']
        config_str = f"{config['model_type']}_{config['cv_type']}_{'smote' if config['use_smote'] else 'nosmote'}_{config['metric']}"
        
        if config_str not in config_groups:
            config_groups[config_str] = []
        config_groups[config_str].append(result)
    
    # Create plots for each configuration
    for config_str, config_results in config_groups.items():
        if not config_results:
            continue
        
        # Filter out NaN results
        valid_results = [r for r in config_results if not np.isnan(r['mean_metric'])]
        if not valid_results:
            continue
        
        # Get feature names and importances
        feature_names = valid_results[0]['feature_names']
        n_features = len(feature_names)
        
        # Create matrix of feature importances (participants x features)
        importance_matrix = np.array([r['feature_importances'] for r in valid_results])
        
        # Calculate mean importance for each feature across all participants
        mean_importances = np.mean(importance_matrix, axis=0)
        
        # Get top N features by mean importance
        top_indices = np.argsort(mean_importances)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importance_matrix[:, top_indices]
        
        # Create boxplot
        plt.figure(figsize=(15, 8))
        
        # Create boxplot data
        boxplot_data = []
        labels = []
        
        for i, feature_idx in enumerate(top_indices):
            feature_importances = importance_matrix[:, feature_idx]
            # Only include non-zero importances for better visualization
            non_zero_importances = feature_importances[feature_importances > 0]
            if len(non_zero_importances) > 0:
                boxplot_data.append(non_zero_importances)
                labels.append(feature_names[feature_idx])
        
        if boxplot_data:
            # Create the boxplot
            bp = plt.boxplot(boxplot_data, labels=labels, patch_artist=True)
            
            # Customize the plot
            plt.title(f'Feature Importance Distribution Across Participants\n{config_str}', fontsize=14, fontweight='bold')
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Feature Importance', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(boxplot_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(plots_dir, f"feature_importance_boxplot_{config_str}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved feature importance boxplot: {plot_file}")
        
        # Create heatmap of feature importances
        if len(valid_results) > 1:
            plt.figure(figsize=(20, max(8, len(valid_results) * 0.3)))
            
            # Create heatmap data with top features
            heatmap_data = importance_matrix[:, top_indices]
            participant_ids = [r['participant_id'] for r in valid_results]
            
            # Create heatmap
            sns.heatmap(heatmap_data, 
                       xticklabels=top_features,
                       yticklabels=participant_ids,
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Feature Importance'})
            
            plt.title(f'Feature Importance Heatmap Across Participants\n{config_str}', fontsize=14, fontweight='bold')
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Participants', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save heatmap
            heatmap_file = os.path.join(plots_dir, f"feature_importance_heatmap_{config_str}.png")
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved feature importance heatmap: {heatmap_file}")
        
        # Create summary statistics plot
        plt.figure(figsize=(15, 6))
        
        # Calculate statistics for top features
        feature_stats = []
        for i, feature_idx in enumerate(top_indices):
            feature_importances = importance_matrix[:, feature_idx]
            non_zero_importances = feature_importances[feature_importances > 0]
            
            if len(non_zero_importances) > 0:
                stats = {
                    'feature': feature_names[feature_idx],
                    'mean': np.mean(feature_importances),
                    'std': np.std(feature_importances),
                    'median': np.median(feature_importances),
                    'non_zero_count': len(non_zero_importances),
                    'total_count': len(feature_importances)
                }
                feature_stats.append(stats)
        
        if feature_stats:
            # Plot mean importance with error bars
            features = [s['feature'] for s in feature_stats]
            means = [s['mean'] for s in feature_stats]
            stds = [s['std'] for s in feature_stats]
            
            plt.bar(range(len(features)), means, yerr=stds, capsize=5, alpha=0.7)
            plt.title(f'Mean Feature Importance with Standard Deviation\n{config_str}', fontsize=14, fontweight='bold')
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Mean Feature Importance', fontsize=12)
            plt.xticks(range(len(features)), features, rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save summary plot
            summary_file = os.path.join(plots_dir, f"feature_importance_summary_{config_str}.png")
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved feature importance summary plot: {summary_file}")

def run_within_participant_analysis(df, output_dir='within_participant_results'):
    """
    Run the ML pipeline for each participant individually.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing all participants
    output_dir : str
        Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define non-feature columns
    non_features = [
        'subject_id',
        'subj_orgid',    
        'session_id',
        'block_num',
        'probe_number',
        'on_off'
    ]
    
    # Prepare data
    X_all = df.drop(columns=non_features)
    X_all = X_all.fillna(0)
    
    le = LabelEncoder()
    y_all = le.fit_transform(df['on_off'])
    
    # Get unique participants
    participants = df['subject_id'].unique()
    print(f"Found {len(participants)} unique participants")
    
    # Debug: Show participant data distribution
    print("\nParticipant data distribution:")
    participant_stats = df.groupby('subject_id').agg({
        'on_off': ['count', 'nunique'],
        'session_id': 'nunique' if 'session_id' in df.columns else lambda x: 'N/A'
    }).round(2)
    participant_stats.columns = ['n_samples', 'n_classes', 'n_sessions']
    print(participant_stats)
    
    # Check class distribution per participant
    print("\nClass distribution per participant:")
    class_dist = df.groupby('subject_id')['on_off'].value_counts().unstack(fill_value=0)
    print(class_dist)
    
    # Define configurations to test
    configurations = [
        # K-fold configurations
        {'use_smote': False, 'model_type': 'rf', 'cv_type': 'kfold', 'metric': 'auc'},
        {'use_smote': False, 'model_type': 'rf', 'cv_type': 'kfold', 'metric': 'balanced_accuracy'},
        {'use_smote': True, 'model_type': 'rf', 'cv_type': 'kfold', 'metric': 'auc'},
        {'use_smote': True, 'model_type': 'rf', 'cv_type': 'kfold', 'metric': 'balanced_accuracy'},
        
        # Session-based split configurations
        {'use_smote': False, 'model_type': 'rf', 'cv_type': 'session_split', 'metric': 'auc'},
        {'use_smote': False, 'model_type': 'rf', 'cv_type': 'session_split', 'metric': 'balanced_accuracy'},
    ]
    
    # Add XGBoost configurations if available
    if XGBOOST_AVAILABLE:
        xgb_configs = [
            {'use_smote': False, 'model_type': 'xgb', 'cv_type': 'kfold', 'metric': 'auc'},
            {'use_smote': False, 'model_type': 'xgb', 'cv_type': 'kfold', 'metric': 'balanced_accuracy'},
            {'use_smote': True, 'model_type': 'xgb', 'cv_type': 'kfold', 'metric': 'auc'},
            {'use_smote': True, 'model_type': 'xgb', 'cv_type': 'kfold', 'metric': 'balanced_accuracy'},
        ]
        configurations.extend(xgb_configs)
    
    # Store all results
    all_results = []
    
    # Run each configuration
    for config_idx, config in enumerate(configurations):
        print(f"\n{'='*60}")
        print(f"Running configuration {config_idx + 1}/{len(configurations)}")
        print(f"Model: {config['model_type'].upper()}, SMOTE: {config['use_smote']}")
        print(f"CV: {config['cv_type']}, Metric: {config['metric']}")
        print(f"{'='*60}")
        
        config_results = []
        
        # Run for each participant
        for participant_idx, participant in enumerate(participants):
            print(f"\nProcessing participant {participant_idx + 1}/{len(participants)}: {participant}")
            
            # Get participant data
            participant_data = df[df['subject_id'] == participant].copy()
            
            # Debug: Show participant info
            print(f"  Data shape: {participant_data.shape}")
            print(f"  Classes: {participant_data['on_off'].value_counts().to_dict()}")
            
            # Skip if not enough data (reduce threshold)
            if len(participant_data) < 5: 
                print(f"  Skipping participant {participant} - insufficient data ({len(participant_data)} samples)")
                continue
            
            # Check if we have both classes
            unique_classes = participant_data['on_off'].unique()
            print(f"  Unique classes: {unique_classes}")
            if len(unique_classes) < 2:
                print(f"  Skipping participant {participant} - only one class present: {unique_classes}")
                continue
            
            X_participant = participant_data.drop(columns=non_features).fillna(0)
            y_participant = le.fit_transform(participant_data['on_off'])
            sessions_participant = participant_data['session_id'].values if 'session_id' in participant_data.columns else None
            
            # Create and fit pipeline
            pipeline = WithinParticipantMLPipeline(
                use_smote=config['use_smote'],
                model_type=config['model_type'],
                cv_type=config['cv_type'],
                n_features=min(20, X_participant.shape[1]),  # Adjust based on available features
                metric=config['metric'],
                cv_splits=5
            )
            
            try:
                pipeline.fit(X_participant, y_participant, sessions=sessions_participant)
                
                # Store results
                result = {
                    'participant_id': participant,
                    'config': config,
                    'mean_metric': pipeline.mean_metric_,
                    'fold_metrics': pipeline.fold_metrics_,
                    'feature_importances': pipeline.feature_importances_.tolist(),
                    'feature_names': pipeline.feature_names_,
                    'n_samples': len(participant_data),
                    'n_features': X_participant.shape[1],
                    'class_distribution': participant_data['on_off'].value_counts().to_dict()
                }
                
                config_results.append(result)
                print(f"  {config['metric'].upper()}: {pipeline.mean_metric_:.4f}")
                
            except Exception as e:
                print(f"  Error processing participant {participant}: {str(e)}")
                continue
        
        # Save results for this configuration
        all_results.extend(config_results)
        
        # Create summary for this configuration
        if config_results:
            config_name = f"{config['model_type']}_{config['cv_type']}_{'smote' if config['use_smote'] else 'nosmote'}_{config['metric']}"
            
            metrics = [r['mean_metric'] for r in config_results if not np.isnan(r['mean_metric'])]
            
            if metrics:
                summary = {
                    'config': config,
                    'n_participants': len(metrics),
                    'mean_metric': np.mean(metrics),
                    'std_metric': np.std(metrics),
                    'min_metric': np.min(metrics),
                    'max_metric': np.max(metrics),
                    'median_metric': np.median(metrics),
                    'individual_metrics': metrics
                }
                
                # Save summary
                summary_file = os.path.join(output_dir, f"summary_{config_name}.json")
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=4)
                
                print(f"\nConfiguration Summary:")
                print(f"  Participants: {len(metrics)}")
                print(f"  Mean {config['metric'].upper()}: {np.mean(metrics):.4f} ± {np.std(metrics):.4f}")
                print(f"  Range: {np.min(metrics):.4f} - {np.max(metrics):.4f}")
    
    # Save all results
    results_file = os.path.join(output_dir, "all_participant_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for result in all_results:
        json_result = result.copy()
        json_result['fold_metrics'] = [float(x) for x in result['fold_metrics']]
        json_result['mean_metric'] = float(result['mean_metric']) if not np.isnan(result['mean_metric']) else None
        json_results.append(json_result)
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"\nAll results saved to: {results_file}")
    
    # Save feature importances
    print("\nSaving feature importances...")
    save_feature_importances(all_results, output_dir)
    
    return all_results

def plot_results_distributions(all_results, output_dir='within_participant_results'):
    """
    Plot distributions of AUC and Balanced Accuracy across participants.
    """
    # Create plots for each configuration
    configs = set()
    for result in all_results:
        config = result['config']
        config_str = f"{config['model_type']}_{config['cv_type']}_{'smote' if config['use_smote'] else 'nosmote'}_{config['metric']}"
        configs.add(config_str)
    
    for config_str in configs:
        # Filter results for this configuration
        config_results = [r for r in all_results if f"{r['config']['model_type']}_{r['config']['cv_type']}_{'smote' if r['config']['use_smote'] else 'nosmote'}_{r['config']['metric']}" == config_str]
        
        if not config_results:
            continue
        
        metrics = [r['mean_metric'] for r in config_results if not np.isnan(r['mean_metric'])]
        
        if not metrics:
            continue
        
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(metrics, bins=min(10, len(metrics)), alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(metrics), color='red', linestyle='--', label=f'Mean: {np.mean(metrics):.3f}')
        plt.axvline(np.median(metrics), color='green', linestyle='--', label=f'Median: {np.median(metrics):.3f}')
        plt.xlabel(config_results[0]['config']['metric'].upper())
        plt.ylabel('Number of Participants')
        plt.title(f'Distribution of {config_results[0]["config"]["metric"].upper()}\n{config_str}')
        plt.legend()
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(metrics)
        plt.ylabel(config_results[0]['config']['metric'].upper())
        plt.title(f'Box Plot\n{config_str}')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"distribution_{config_str}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved distribution plot: {plot_file}")
def extract_feature_importances_to_csv(df, output_file='participant_feature_importances.csv'):
    """
    Extract feature importances for every participant and save to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing all participants
    output_file : str
        Output CSV filename
    """
    
    # Define non-feature columns (same as in original code)
    non_features = [
        'subject_id',
        'subj_orgid',    
        'session_id',
        'block_num',
        'probe_number',
        'on_off'
    ]
    
    # Prepare data
    X_all = df.drop(columns=non_features)
    X_all = X_all.fillna(0)
    
    le = LabelEncoder()
    y_all = le.fit_transform(df['on_off'])
    
    # Get unique participants using subj_orgid
    participants = df['subj_orgid'].unique()
    participants = participants[~pd.isna(participants)]  # Remove NaN values
    print(f"Found {len(participants)} unique participants")
    
    # Store results for CSV
    csv_rows = []
    
    # Simple configuration - using Random Forest with default settings
    config = {
        'use_smote': False, 
        'model_type': 'rf', 
        'cv_type': 'kfold',
        'n_features': 20,
        'metric': 'auc',
        'cv_splits': 5
    }
    
    print("Extracting feature importances for each participant...")
    
    # Process each participant
    for participant_idx, participant in enumerate(participants):
        print(f"Processing participant {participant_idx + 1}/{len(participants)}: {participant}")
        
        # Get participant data (all sessions for this participant)
        participant_data = df[df['subj_orgid'] == participant].copy()
        
        if len(participant_data) < 10:
            print(f"  Skipping - insufficient data ({len(participant_data)} samples)")
            continue
        
        # Check if we have both classes
        unique_classes = participant_data['on_off'].unique()
        if len(unique_classes) < 2:
            print(f"  Skipping - only one class present")
            continue
        
        # Prepare participant-specific data
        X_participant = participant_data.drop(columns=non_features).fillna(0)
        y_participant = le.fit_transform(participant_data['on_off'])
        
        # Adjust number of features based on available data
        n_features = min(20, X_participant.shape[1])
        
        try:
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectKBest(f_classif, k=n_features)),
                ('model', RandomForestClassifier(random_state=42))
            ])
            
            # Use cross-validation to get feature importances across folds
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Perform cross-validation and collect estimators
            cv_results = cross_validate(
                pipeline, X_participant, y_participant,
                cv=cv, return_estimator=True, scoring='roc_auc'
            )
            
            # Extract feature importances from each fold
            fold_importances = []
            feature_names = X_participant.columns.tolist()
            
            for estimator in cv_results['estimator']:
                selector = estimator.named_steps['feature_selection']
                model = estimator.named_steps['model']
                
                # Get selected feature indices
                selected_indices = selector.get_support(indices=True)
                
                # Initialize importance array with zeros
                all_importances = np.zeros(len(feature_names))
                
                # Fill in importances for selected features
                all_importances[selected_indices] = model.feature_importances_
                
                fold_importances.append(all_importances)
            
            # Average feature importances across folds
            mean_importances = np.mean(fold_importances, axis=0)
            mean_cv_score = np.mean(cv_results['test_score'])
            
            # Create row for CSV
            row = {
                'participant_id': participant,
                'n_samples': len(participant_data),
                'n_features_total': X_participant.shape[1],
                'n_features_selected': n_features,
                'cv_auc_mean': mean_cv_score,
                'cv_auc_std': np.std(cv_results['test_score'])
            }
            
            # Add feature importances
            for i, feature_name in enumerate(feature_names):
                row[f'importance_{feature_name}'] = mean_importances[i]
            
            csv_rows.append(row)
            
            print(f"  Success - AUC: {mean_cv_score:.4f} ± {np.std(cv_results['test_score']):.4f}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            continue
    
    # Create DataFrame and save to CSV
    if csv_rows:
        results_df = pd.DataFrame(csv_rows)
        results_df.to_csv(output_file, index=False)
        print(f"\nFeature importances saved to: {output_file}")
        print(f"Shape: {results_df.shape}")
        
        # Display summary
        print(f"\nSummary:")
        print(f"Participants processed: {len(csv_rows)}")
        print(f"Mean AUC across participants: {results_df['cv_auc_mean'].mean():.4f} ± {results_df['cv_auc_mean'].std():.4f}")
        
        # Show top features overall
        importance_cols = [col for col in results_df.columns if col.startswith('importance_')]
        mean_importances_overall = results_df[importance_cols].mean()
        top_features = mean_importances_overall.nlargest(10)
        
        print(f"\nTop 10 features (mean importance across all participants):")
        for i, (feature, importance) in enumerate(top_features.items(), 1):
            feature_name = feature.replace('importance_', '')
            print(f"{i:2d}. {feature_name}: {importance:.4f}")
        
        return results_df
    else:
        print("No participants could be processed successfully.")
        return None

def create_feature_importance_summary(results_df, summary_file='feature_importance_summary.csv'):
    """
    Create a summary of feature importances across all participants.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with feature importances for each participant
    summary_file : str
        Output CSV filename for summary
    """
    if results_df is None:
        return
    
    # Get importance columns
    importance_cols = [col for col in results_df.columns if col.startswith('importance_')]
    
    summary_stats = results_df[importance_cols].describe()
    
    summary_stats.loc['non_zero_count'] = (results_df[importance_cols] > 0).sum()
    summary_stats.loc['non_zero_percentage'] = (results_df[importance_cols] > 0).mean() * 100
    
    summary_stats.columns = [col.replace('importance_', '') for col in summary_stats.columns]
    
    summary_stats = summary_stats.T
    
    summary_stats = summary_stats.sort_values('mean', ascending=False)
    
    # Save summary
    summary_stats.to_csv(summary_file)
    print(f"Feature importance summary saved to: {summary_file}")
    
    return summary_stats
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("/Users/teostei/Desktop/MINDWANDERINGDEEPLEARNING/big_window_combined_behavior_eye_features_labeled3.csv")
    df['subject_id'] = df['subj_orgid']  
    
    # Run within-participant analysis
    results = run_within_participant_analysis(df)
    
    # Plot distributions
    plot_results_distributions(results)
    
    # Plot feature importance distributions
    print("\nPlotting feature importance distributions...")
    plot_feature_importance_distributions(results)
    
    # Run permutation tests
    print("\nRunning permutation tests...")
    permutation_results = run_permutation_analysis(df, n_bootstrap=200, n_permutations=200)
    
    print("Within-participant analysis complete with permutation testing!")