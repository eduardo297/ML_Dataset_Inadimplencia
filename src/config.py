"""
Configurações centrais do projeto: caminhos e definição de grupos de features.

Manter esses valores aqui (em vez de espalhados pelos notebooks) evita que
uma mudança precise ser repetida em vários lugares.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
RAIZ_PROJETO = Path(__file__).resolve().parents[1]

DIR_DADOS_RAW = RAIZ_PROJETO / "data" / "raw"
DIR_MODELOS = RAIZ_PROJETO / "models"

CAMINHO_FEATURES = DIR_DADOS_RAW / "features.csv"
CAMINHO_TARGET = DIR_DADOS_RAW / "target.csv"

# ID do dataset "Default of Credit Card Clients" no UCI ML Repository
UCI_DATASET_ID = 350

RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Grupos de features (usados no pré-processamento e na análise de importância)
# --------------------------------------------------------------------------
FEATURES_NUMERICAS = [
    "X1", "X5",
    "X6", "X7", "X8", "X9", "X10", "X11",
    "X12", "X13", "X14", "X15", "X16", "X17",
    "X18", "X19", "X20", "X21", "X22", "X23",
]

FEATURES_NOMINAIS = ["X2", "X4"]
FEATURES_ORDINAIS = ["X3"]

COLUNAS_HISTORICO_PAGAMENTO = ["X6", "X7", "X8", "X9", "X10", "X11"]
COLUNAS_FATURAS = ["X12", "X13", "X14", "X15", "X16", "X17"]
COLUNAS_PAGAMENTOS_ANTERIORES = ["X18", "X19", "X20", "X21", "X22", "X23"]
