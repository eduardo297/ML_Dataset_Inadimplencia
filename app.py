import streamlit as st
import pandas as pd

from src.modeling import carregar_modelo_final

# Configuração da página
st.set_page_config(
    page_title="Previsão de Inadimplência",
    page_icon="💳",
    layout="centered",
)

# --------------------------------------------------------------------------
# Configuração dos campos de entrada
# --------------------------------------------------------------------------
# Em vez de escrever um st.sidebar.xxx() para cada uma das 23 variáveis,
# descrevemos cada campo como um dicionário e geramos os widgets em um loop.
# Para adicionar/remover uma variável, basta editar esta lista.
CAMPOS = [
    # --- Dados gerais -----------------------------------------------------
    {
        "coluna": "X1", "label": "Limite de crédito (X1)", "tipo": "number",
        "min": 1_000, "max": 1_000_000, "default": 50_000, "step": 5_000,
    },
    {
        "coluna": "X2", "label": "Sexo (X2)", "tipo": "select",
        "opcoes": {"Masculino": 1, "Feminino": 2},
    },
    {
        "coluna": "X3", "label": "Educação (X3)", "tipo": "select",
        "opcoes": {"Pós-graduação": 1, "Graduação": 2, "Ensino médio": 3, "Outros": 4},
    },
    {
        "coluna": "X4", "label": "Estado civil (X4)", "tipo": "select",
        "opcoes": {"Casado(a)": 1, "Solteiro(a)": 2, "Outros": 3},
    },
    {
        "coluna": "X5", "label": "Idade (X5)", "tipo": "number",
        "min": 18, "max": 100, "default": 35, "step": 1,
    },
    # --- Histórico de pagamento (X6-X11) -----------------------------------
    *[
        {
            "coluna": f"X{i}",
            "label": f"Histórico de pagamento — {rotulo} (X{i})",
            "tipo": "slider",
            "min": -2, "max": 8, "default": 0,
            "help": "-2/-1: em dia | 0: uso revolving | 1+: meses de atraso",
        }
        for i, rotulo in zip(
            range(6, 12),
            ["mês mais recente", "2 meses atrás", "3 meses atrás",
             "4 meses atrás", "5 meses atrás", "6 meses atrás"],
        )
    ],
    # --- Valores das faturas (X12-X17) -------------------------------------
    *[
        {
            "coluna": f"X{i}", "label": f"Valor da fatura — {j} meses atrás (X{i})",
            "tipo": "number", "min": -100_000, "max": 1_000_000, "default": 0, "step": 1_000,
        }
        for i, j in zip(range(12, 18), range(1, 7))
    ],
    # --- Pagamentos anteriores (X18-X23) -----------------------------------
    *[
        {
            "coluna": f"X{i}", "label": f"Pagamento realizado — {j} meses atrás (X{i})",
            "tipo": "number", "min": 0, "max": 1_000_000, "default": 0, "step": 1_000,
        }
        for i, j in zip(range(18, 24), range(1, 7))
    ],
]


def renderizar_campo(campo: dict):
    """Cria o widget correto no sidebar a partir da configuração do campo e devolve o valor."""
    if campo["tipo"] == "number":
        return st.sidebar.number_input(
            campo["label"],
            min_value=campo["min"],
            max_value=campo["max"],
            value=campo["default"],
            step=campo.get("step", 1),
            help=campo.get("help"),
        )
    if campo["tipo"] == "slider":
        return st.sidebar.slider(
            campo["label"],
            min_value=campo["min"],
            max_value=campo["max"],
            value=campo["default"],
            help=campo.get("help"),
        )
    if campo["tipo"] == "select":
        rotulo_escolhido = st.sidebar.selectbox(campo["label"], list(campo["opcoes"].keys()))
        return campo["opcoes"][rotulo_escolhido]

    raise ValueError(f"Tipo de campo não suportado: {campo['tipo']}")


# --------------------------------------------------------------------------
# Carregamento do modelo
# --------------------------------------------------------------------------
# carregar_modelo_final espera o NOME do arquivo (não o Path completo) e
# retorna um dicionário com pipeline + threshold + métricas salvos juntos.
NOME_ARQUIVO_MODELO = "modelo_final_xgb"  # sem .joblib, a função completa sozinha

try:
    from src.config import DIR_MODELOS
    #st.sidebar.caption(f"🔍 Debug — procurando modelo em: `{DIR_MODELOS.resolve()}`")
    #st.sidebar.caption(f"🔍 Debug — pasta existe: {DIR_MODELOS.exists()}")
    #if DIR_MODELOS.exists():
    #    st.sidebar.caption(f"🔍 Debug — arquivos na pasta: {[p.name for p in DIR_MODELOS.iterdir()]}")

    artefato = carregar_modelo_final(NOME_ARQUIVO_MODELO)
    pipeline = artefato["pipeline"]
    threshold = artefato["threshold"]
    modelo_carregado = True
except Exception as e:
    #st.sidebar.error(f"🔍 Debug — erro real ao carregar: {type(e).__name__}: {e}")
    modelo_carregado = False


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
st.title("💳 Previsão de Inadimplência de Cartão de Crédito")
st.markdown(
    "Esta aplicação utiliza um modelo de **Machine Learning (XGBoost)** treinado "
    "para prever a probabilidade de inadimplência de um cliente no mês seguinte, "
    "com base em dados históricos e demográficos."
)

if not modelo_carregado:
    st.warning(
        "⚠️ O modelo serializado não foi encontrado em `models/`. "
        "Rode o notebook `pre_processamento.ipynb` até a etapa de "
        "`salvar_modelo_final()` antes de usar este app."
    )
else:
    st.sidebar.header("Parâmetros do cliente")

    # Gera todos os inputs a partir da lista CAMPOS, em vez de repetir o código
    valores = {campo["coluna"]: renderizar_campo(campo) for campo in CAMPOS}

    if st.sidebar.button("Calcular risco"):
        # DataFrame de uma linha, na ordem que o pipeline espera (o
        # ColumnTransformer seleciona colunas pelo nome, então a ordem do
        # dict não precisa ser idêntica à do treino — só os nomes precisam bater)
        input_data = pd.DataFrame([valores])

        proba = pipeline.predict_proba(input_data)[0][1]

        st.subheader("Resultado da análise")

        if proba >= threshold:
            st.error(f"⚠️ Alto risco de inadimplência (probabilidade: {proba:.2%})")
            st.markdown("O modelo sinaliza este cliente como **inadimplente** para o próximo mês.")
        else:
            st.success(f"✅ Baixo risco de inadimplência (probabilidade: {proba:.2%})")
            st.markdown("O modelo sinaliza este cliente como **adimplente** para o próximo mês.")

        st.caption(f"Threshold de decisão usado: {threshold:.2f} (otimizado via F2-score)")