"""
Funções auxiliares de treino, busca de hiperparâmetros, escolha de threshold,
avaliação de modelo e montagem da tabela comparativa entre modelos.
"""


import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from pathlib import Path


from sklearn.model_selection import GridSearchCV, cross_val_predict

from src.config import (
    COLUNAS_FATURAS,
    COLUNAS_HISTORICO_PAGAMENTO,
    COLUNAS_PAGAMENTOS_ANTERIORES,
    DIR_MODELOS,
)



def realizar_busca_hiperparametros(
    modelo, hiperparametros, X_train, y_train, scoring="roc_auc", cv=5
):
    """Executa GridSearchCV e retorna o melhor estimador, parâmetros e score."""
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=hiperparametros,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def melhor_threshold_via_cv(modelo, X_train, y_train, cv, beta=2, faixa=(0.15, 0.55, 0.01)):
    """
    Busca o threshold que maximiza o F-beta score usando previsões
    out-of-fold via cross_val_predict — evita otimizar o threshold
    olhando para o próprio conjunto de teste.
    """
    y_prob_cv = cross_val_predict(modelo, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    thresholds = np.arange(*faixa)
    resultados = []
    for t in thresholds:
        y_pred_t = (y_prob_cv >= t).astype(int)
        resultados.append({"threshold": t, "fbeta": fbeta_score(y_train, y_pred_t, beta=beta)})

    return pd.DataFrame(resultados)


def avaliar_modelo(y_true, y_pred, imprimir: bool = True) -> dict:
    """Calcula métricas de classificação e, opcionalmente, imprime um resumo."""
    metricas = {
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f2": fbeta_score(y_true, y_pred, beta=2),
    }

    if imprimir:
        print("Matriz de confusão:")
        print(metricas["confusion_matrix"])
        print(f"\nPrecision: {metricas['precision']:.3f}")
        print(f"Recall:    {metricas['recall']:.3f}")
        print(f"F1:        {metricas['f1']:.3f}")
        print(f"F2:        {metricas['f2']:.3f}")

    return metricas


def identificar_grupo(feature: str) -> str:
    """Mapeia o nome de uma feature (pós ColumnTransformer) para um grupo de negócio."""
    feature = feature.split("__")[-1]

    if feature == "X1":
        return "Limite de crédito"
    if feature.startswith("X2_") or feature == "X3" or feature.startswith("X4_") or feature == "X5":
        return "Demográficas"
    if feature in COLUNAS_HISTORICO_PAGAMENTO:
        return "Histórico de pagamento"
    if feature in COLUNAS_FATURAS:
        return "Valores das faturas"
    if feature in COLUNAS_PAGAMENTOS_ANTERIORES:
        return "Pagamentos anteriores"
    return "Outros"


def montar_tabela_comparacao(resultados: dict) -> pd.DataFrame:
    """
    Monta a tabela de comparação entre modelos automaticamente a partir de um
    dicionário {nome_modelo: metricas}, em vez de digitar os números à mão
    no notebook/README (o que fica desatualizado se algo mudar).

    Exemplo
    -------
    resultados = {
        "Regressão Logística": avaliar_modelo(y_test, y_pred_lr, imprimir=False),
        "Random Forest": avaliar_modelo(y_test, y_pred_rf, imprimir=False),
        "XGBoost": avaliar_modelo(y_test, y_pred_xgb, imprimir=False),
    }
    montar_tabela_comparacao(resultados)
    """
    linhas = []
    for nome, metricas in resultados.items():
        linhas.append(
            {
                "Modelo": nome,
                "Precision": round(metricas["precision"], 3),
                "Recall": round(metricas["recall"], 3),
                "F1": round(metricas["f1"], 3),
                "F2": round(metricas["f2"], 3),
            }
        )
    return pd.DataFrame(linhas)

def salvar_modelo_final(modelo, threshold, metricas, nome_arquivo="modelo_final_xgb"):
    """
    Salva o pipeline final (preprocessador + modelo), o threshold otimizado
    e as métricas de teste em um único artefato .joblib.
    """
    DIR_MODELOS.mkdir(exist_ok=True, parents=True)

    if not nome_arquivo.endswith(".joblib"):
        nome_arquivo += ".joblib"

    artefato = {
        "pipeline": modelo,             # Pipeline completo (preprocessador + modelo)
        "threshold": threshold,         # threshold otimizado via F2-score
        "metricas_teste": metricas,     # métricas no conjunto de teste, para referência
    }

    caminho = DIR_MODELOS / nome_arquivo
    joblib.dump(artefato, caminho)

    return f"Modelo salvo em {caminho}"


def carregar_modelo_final(nome_arquivo="modelo_final_xgb"):
    """Carrega o artefato salvo por salvar_modelo_final()."""
    if not nome_arquivo.endswith(".joblib"):
        nome_arquivo += ".joblib"

    return joblib.load(DIR_MODELOS / nome_arquivo)
