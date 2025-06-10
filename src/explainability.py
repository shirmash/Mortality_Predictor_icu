from joblib import load
import pickle
import numpy as np
import pandas as pd
import shap
import xgboost
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from itertools import combinations

def global_logistic_regression(model, column_info):
    # For LogisticRegression, get coefficients and odds ratios
    coefficients = model.coef_[0]  # shape: (n_features,)
    odds_ratios = np.exp(coefficients)

    # Create DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'Feature': column_info,
        'Coefficient': coefficients,
        'Odds Ratio': odds_ratios
    })

    # Sort by absolute effect size
    feature_importance['Abs Coef'] = np.abs(coefficients)
    feature_importance = feature_importance.sort_values(by='Abs Coef', ascending=False)

    # Drop helper column if you want
    feature_importance = feature_importance.drop(columns='Abs Coef')

    # Print result
    print(feature_importance)


def pdp_plot_with_inverse(model, features):
    """
    Plot PDP using scaled input (for model) but unscaled x-axis values (for readability).

    Parameters:
    - model: Trained on scaled features
    - features: List of feature names (e.g., ['age', 'glucose'])
    """
    # Load and prepare data
    x_train = pd.read_csv('train_preprocessed_data.csv')
    x_train = x_train.drop(columns=['hospital_death'])
    scaler = load('scaler.joblib')

    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 5 * n_features))

    # Handle single feature case
    if n_features == 1:
        axes = [axes]

    display = PartialDependenceDisplay.from_estimator(
        model,
        x_train,
        features=features,  # Use indices
        kind='average',
        grid_resolution=100,
        ax=axes
    )
    # Generate PDP for each feature
    for i, feature in enumerate(features):
        feature_index = list(scaler.feature_names_in_).index(feature)

        # Get the current x-ticks (these are in scaled space)
        scaled_ticks = axes[i].get_xticks()

        # Only keep ticks that are within the data range
        scaled_min, scaled_max = x_train[feature].min(), x_train[feature].max()
        valid_ticks = [tick for tick in scaled_ticks if scaled_min <= tick <= scaled_max]
        if not valid_ticks:  # If no valid ticks, create some
            valid_ticks = np.linspace(scaled_min, scaled_max, 10)

        # Convert these ticks to original scale - use manual formula instead of inverse_transform
        orig_ticks = []
        for scaled_val in valid_ticks:
            mean = scaler.mean_[feature_index]
            std = np.sqrt(scaler.var_[feature_index])
            # Use the manual formula: original = scaled * std + mean
            original_val = scaled_val * std + mean
            orig_ticks.append(original_val)

        # Update the x-axis with original values
        axes[i].set_xticks(valid_ticks)
        axes[i].set_xticklabels([f"{val:.2f}" for val in orig_ticks])

        # Update the axis label
        feature_display_name = x_train.columns[feature] if isinstance(feature, int) else feature
        axes[i].set_xlabel(f"{feature_display_name} (original scale)")

        # Add grid lines
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].set_title(f'Partial Dependence Plot for {feature_display_name}')

    plt.tight_layout()
    plt.show()

def knn_explainability(model):
    # Load and sample data
    x_train = pd.read_csv('train_preprocessed_data.csv')
    x_train_samples = pd.concat([
        x_train[x_train['hospital_death'] == 0].sample(n=200, random_state=0),
        x_train[x_train['hospital_death'] == 1].sample(n=200, random_state=0)
    ])

    y_train = x_train_samples['hospital_death']
    x_train_samples = x_train_samples.drop(columns=['hospital_death'])

    # Unscale age
    scaler = load('scaler.joblib')
    feature_index = list(scaler.feature_names_in_).index('age')
    mean = scaler.mean_[feature_index]
    std = np.sqrt(scaler.var_[feature_index])
    x_train_samples['age_unscaled'] = x_train_samples['age'] * std + mean

    # Features for explanation plot
    feature_x = 'd1_lactate_min'
    feature_y = 'd1_lactate_max'

    # Fit KNN on just these 2 features
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_samples[[feature_x, feature_y]], y_train)

    # Create grid for background
    x_min, x_max = x_train_samples[feature_x].min(), x_train_samples[feature_x].max()
    y_min, y_max = x_train_samples[feature_y].min(), x_train_samples[feature_y].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = knn.predict(grid)
    Z = preds.reshape(xx.shape)

    # Get training point prediction confidences
    train_probs = knn.predict_proba(x_train_samples[[feature_x, feature_y]])
    confidences = train_probs.max(axis=1)

    # Select a random test sample
    test_index = 3000  # or use random: np.random.randint(0, len(x_train))
    test_sample = x_train.iloc[test_index][[feature_x, feature_y]].values.reshape(1, -1)

    # Find neighbors
    distances, indices = knn.kneighbors(test_sample)

    # Plot background + contour
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.contour(xx, yy, Z, levels=np.unique(y_train), colors='black', linewidths=0.5)

    # Separate training samples by class for custom coloring
    class_0_mask = y_train == 0
    class_1_mask = y_train == 1

    plt.scatter(
        x_train_samples.loc[class_0_mask, feature_x],
        x_train_samples.loc[class_0_mask, feature_y],
        color='blue', label='Survived (0)',
        edgecolor='k',
        s=50 + 100 * confidences[class_0_mask],
        alpha=0.8
    )

    plt.scatter(
        x_train_samples.loc[class_1_mask, feature_x],
        x_train_samples.loc[class_1_mask, feature_y],
        color='red', label='Died (1)',
        edgecolor='k',
        s=50 + 100 * confidences[class_1_mask],
        alpha=0.8
    )

    # Plot test sample
    plt.scatter(
        test_sample[0, 0], test_sample[0, 1],
        color='black', s=150, marker='X', label='Test Sample'
    )

    # Plot lines from test sample to its neighbors
    # Plot lines to neighbors
    for idx in indices[0]:
        neighbor = x_train_samples.iloc[idx][[feature_x, feature_y]]
        plt.plot(
            [test_sample[0, 0], neighbor[0]],
            [test_sample[0, 1], neighbor[1]],
            color='gray', linestyle='--', linewidth=1
        )

    # Final touches
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def shap_model(model):
    x_train = pd.read_csv('train_preprocessed_data.csv')
    x_train = x_train.drop(columns=['hospital_death'])

    background = shap.sample(x_train, 100, random_state=42)
    # Define prediction function for class 1 probability
    predict_fn = lambda x: model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_fn, background)
    sample_to_explain = x_train.iloc[:100]

    shap_values = explainer.shap_values(sample_to_explain)
    # Confirm matching shape
    assert len(shap_values) == 100
    assert shap_values.shape[1] == x_train.shape[1]

    # SHAP summary plot
    shap.summary_plot(shap_values, sample_to_explain, show=True, max_display=20)

def shap_model_tree(model, train_path):
    x_train = pd.read_csv(train_path)
    x_train = x_train.drop(columns=['hospital_death'])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_train)

    # Create summary bar plot
    fig = plt.figure(figsize=(12, 8))  # Wider figure
    shap.summary_plot(
        shap_values,
        x_train,
        show=False,  # Don't display yet
        max_display=20  # Control how many features to show
    )

    # Fix layout and display properly
    plt.tight_layout()
    plt.show()

def xgb_builtin_feature_importance(model, importance_type='gain'):
    """
    Plot built-in XGBoost feature importance.

    Parameters:
    - model: Trained XGBClassifier or XGBRegressor
    - importance_type: 'weight', 'gain', or 'cover'
    """
    booster = model.get_booster()
    scores = booster.get_score(importance_type=importance_type)
    # Convert to DataFrame
    importance_df = pd.DataFrame(list(scores.items()), columns=['Feature', 'Importance'])

    # Normalize to 0–1 scale
    importance_df['Importance'] = importance_df['Importance'].astype(float)
    importance_df['Normalized'] = importance_df['Importance'] / importance_df['Importance'].max()

    # Sort and keep top N
    importance_df = importance_df.sort_values(by='Normalized', ascending=False).head(20)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Normalized'], color='steelblue')
    plt.xlabel("Importance")
    plt.title(f"Top {20} Features Importance - XGBoost Model")
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load a model
    model_xgb = load('xgboost_best_model.joblib')
    model_logistic = load('logistic_regression_best_model.joblib')
    model_knn = load('knn_best_model.joblib')
    model_xgb_apa = load('xgboost.joblib')

    with open('column_names.pkl', 'rb') as f:
        column_info = pickle.load(f)

    x_train_path = 'train_preprocessed_data.csv'
    x_train_apa_path = 'train_preprocessed_data_apa.csv'
    # shap_model(model_logistic)
    # shap_model(model_knn)
    # global_logistic_regression(model, column_info)
    # shap_model_tree(model_xgb, x_train_path)
    shap_model_tree(model_xgb_apa, x_train_apa_path)
    # scaler = load('scaler.joblib')

    # pdp_plot_with_inverse(model_xgb, features=['age', 'd1_bun_max', 'bmi'])
    # pdp_plot_with_inverse(model_xgb, features=['age'])
    # knn_explainability(model_knn)

    # xgb_builtin_feature_importance(model_xgb)
