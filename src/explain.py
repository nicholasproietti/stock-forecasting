"""
explain.py: Module for model explainability using SHAP.
"""
import shap

def explain_model(model, X_train, X_test):
    """
    Compute SHAP values for a trained tree-based model (e.g. XGBoost).
    
    Returns the SHAP explanation object for X_test.
    """
    explainer = shap.TreeExplainer(model, data=X_train)
    shap_values = explainer.shap_values(X_test)
    return shap_values