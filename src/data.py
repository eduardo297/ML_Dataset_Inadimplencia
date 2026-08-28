"""
Carregamento do dataset "Default of Credit Card Clients" (UCI id=350).

Na primeira chamada, os dados são baixados via `ucimlrepo` e salvos
localmente em data/raw/*.csv. Nas chamadas seguintes, os dados são lidos
direto do disco — o projeto deixa de depender da API do UCI (e da conexão
com a internet) toda vez que alguém abre um notebook.
"""

import pandas as pd

from src.config import (
    CAMINHO_FEATURES,
    CAMINHO_TARGET,
    DIR_DADOS_RAW,
    UCI_DATASET_ID,
)


def _baixar_e_salvar_dataset():
    """Baixa o dataset via ucimlrepo e salva localmente em CSV."""
    from ucimlrepo import fetch_ucirepo

    print("Dataset local não encontrado. Baixando da UCI ML Repository...")
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)

    X = dataset.data.features
    y = dataset.data.targets

    DIR_DADOS_RAW.mkdir(parents=True, exist_ok=True)
    X.to_csv(CAMINHO_FEATURES, index=False)
    y.to_csv(CAMINHO_TARGET, index=False)

    print(f"Dados salvos em: {DIR_DADOS_RAW}")
    return X, y


def carregar_dataset(forcar_download: bool = False):
    """
    Carrega features (X) e target (y) do dataset de inadimplência.

    Parameters
    ----------
    forcar_download : bool
        Se True, ignora o cache local em data/raw/ e baixa novamente da UCI.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    tem_cache = CAMINHO_FEATURES.exists() and CAMINHO_TARGET.exists()

    if forcar_download or not tem_cache:
        X, y = _baixar_e_salvar_dataset()
    else:
        X = pd.read_csv(CAMINHO_FEATURES)
        y = pd.read_csv(CAMINHO_TARGET)

    # y vem como DataFrame de 1 coluna na primeira vez; padroniza para Series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    return X, y


if __name__ == "__main__":
    # Rodar com: python -m src.data
    # Popula o cache local em data/raw/ na primeira execução.
    X, y = carregar_dataset(forcar_download=True)
    print("Features:", X.shape)
    print("Target:", y.shape)
