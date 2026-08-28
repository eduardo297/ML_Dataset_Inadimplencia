"""
Funções de limpeza dos dados e construção do pipeline de pré-processamento.

Extraído do notebook original (pre_processamento.ipynb) para permitir reuso
e testes, sem precisar copiar/colar código entre notebooks.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import FEATURES_NOMINAIS, FEATURES_NUMERICAS, FEATURES_ORDINAIS


def remover_duplicatas_completas(X: pd.DataFrame, y: pd.Series):
    """
    Remove registros onde X e y juntos são idênticos (duplicata exata).

    Mantém os casos em que X é idêntico mas Y difere — como identificado na
    EDA, não é possível afirmar que sejam erros de rótulo.
    """
    dados = X.copy()
    dados["Y"] = y.values

    dados = dados.drop_duplicates()

    X_limpo = dados.drop(columns=["Y"]).reset_index(drop=True)
    y_limpo = dados["Y"].reset_index(drop=True)

    return X_limpo, y_limpo


def tratar_categorias_invalidas(X: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa códigos não documentados de X3 (educação) e X4 (estado civil)
    na categoria "outros" mais próxima, conforme decidido na EDA.
    """
    X = X.copy()

    # X3 - educação: códigos 0, 5, 6 não documentados -> agrupa em 4 ("outros")
    X["X3"] = X["X3"].replace({0: 4, 5: 4, 6: 4})

    # X4 - estado civil: código 0 não documentado -> agrupa em 3 ("outros")
    X["X4"] = X["X4"].replace({0: 3})

    return X


def construir_preprocessador() -> ColumnTransformer:
    """Cria o ColumnTransformer usado nos pipelines de modelagem."""
    return ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), FEATURES_NUMERICAS),
            (
                "nominais",
                OneHotEncoder(handle_unknown="ignore"),
                FEATURES_NOMINAIS,
            ),
            (
                "ordinais",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                FEATURES_ORDINAIS,
            ),
        ]
    )
