"""
Funções auxiliares para interpretabilidade do modelo final via SHAP.

Isola as chamadas de SHAP usadas no notebook de interpretabilidade, deixando
o notebook focado na leitura dos resultados em vez do código repetitivo.
"""

import shap


def calcular_shap_values(modelo_treinado, X_transformado):
    """Calcula SHAP values para um modelo baseado em árvore (ex: XGBoost, RF)."""
    explainer = shap.TreeExplainer(modelo_treinado)
    shap_values = explainer.shap_values(X_transformado)
    return explainer, shap_values


def plot_resumo_shap(shap_values, X_transformado, nomes_features, tipo="bar"):
    """
    Plota o summary_plot do SHAP.

    tipo="bar" -> importância agregada por feature
    tipo=None  -> gráfico padrão (distribuição do impacto por feature)
    """
    shap.summary_plot(shap_values, X_transformado, feature_names=nomes_features, plot_type=tipo)


def plot_dependencia_shap(feature, shap_values, X_transformado, nomes_features):
    """Wrapper para shap.dependence_plot — mostra como uma feature afeta a previsão."""
    shap.dependence_plot(feature, shap_values, X_transformado, feature_names=nomes_features)


def explicar_caso(explainer, shap_values, X_transformado, posicao, nomes_features):
    """Plota o force_plot do SHAP para um caso individual (ex: um falso negativo)."""
    shap.force_plot(
        explainer.expected_value,
        shap_values[posicao],
        X_transformado[posicao],
        feature_names=nomes_features,
        matplotlib=True,
        text_rotation=45,
    )
