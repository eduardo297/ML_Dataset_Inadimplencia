# Previsão de Inadimplência — Cartão de Crédito

Projeto de Machine Learning para prever a inadimplência de clientes de cartão de
crédito no mês seguinte, utilizando o dataset **Default of Credit Card Clients**
(UCI Machine Learning Repository, id 350).

O problema é tratado como uma tarefa de **classificação binária**, priorizando a
identificação correta dos clientes inadimplentes (recall/F2-score), já que o
custo de não identificar um inadimplente tende a ser maior do que o custo de
sinalizar um cliente adimplente por engano.

## Estrutura do projeto

```
.
├── data/
│   └── raw/                    # dataset em cache local (gerado, não versionado)
├── models/                     # modelo final serializado (gerado, não versionado)
├── notebooks/
│   ├── analise_exploratoria.ipynb   # EDA: qualidade dos dados, distribuições, correlações
│   └── pre_processamento.ipynb      # limpeza, modelagem, tuning, comparação e interpretabilidade
├── src/
│   ├── config.py               # caminhos e definição de grupos de features
│   ├── data.py                 # carregamento do dataset com cache local
│   ├── preprocessing.py        # limpeza dos dados e ColumnTransformer
│   ├── modeling.py              # tuning, threshold, avaliação, comparação e persistência do modelo
│   └── interpretability.py     # funções auxiliares de SHAP
├── app.py                      # demo interativa (Streamlit) do modelo final
├── .gitignore
├── requirements.txt
└── README.md
```

## Como rodar

**1. Criar o ambiente e instalar as dependências**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Baixar o dataset e popular o cache local**

```bash
python -m src.data
```

Isso baixa o dataset uma única vez via `ucimlrepo` e salva em `data/raw/`. Nas
execuções seguintes, os notebooks leem direto do disco — sem depender da API
da UCI.

**3. Rodar os notebooks, na ordem**

1. `notebooks/analise_exploratoria.ipynb`
2. `notebooks/pre_processamento.ipynb` — ao final, salva o modelo treinado em
   `models/modelo_final_xgb.joblib` via `salvar_modelo_final()`.

**4. Rodar a demo interativa**

Com o modelo já salvo (passo 3), a partir da raiz do projeto:

```bash
streamlit run app.py
```

A aplicação abre no navegador, recebe os dados de um cliente através de
campos na barra lateral e retorna a probabilidade de inadimplência e a
decisão do modelo (usando o mesmo threshold otimizado no notebook).

## Sobre os dados

- **30.000 registros**, 23 variáveis preditoras (`X1`–`X23`) e a variável alvo
  `Y` (1 = inadimplente no mês seguinte, 0 = adimplente).
- Sem valores ausentes.
- Detalhes completos da investigação de qualidade dos dados estão no notebook
  de EDA.

### Decisões de pré-processamento

| Problema encontrado | Decisão |
| :--- | :--- |
| Valores ausentes | Nenhum tratamento necessário |
| 35 duplicatas completas (X e Y idênticos) | Remover |
| 21 grupos com X idêntico e Y diferente | Manter (não é possível confirmar erro de rótulo) |
| X3 (educação) com códigos não documentados | Agrupar como "Outros" |
| X4 (estado civil) com código não documentado | Agrupar como "Outros" |
| X6–X11 com códigos de -2 a 8 | Preservar inicialmente |
| Valores negativos em X12–X17 (faturas) | Preservar inicialmente |
| Escalas diferentes entre variáveis | Padronização (StandardScaler) |
| Valores extremos (outliers) | Investigados, sem remoção automática |

## Modelagem

Três modelos foram treinados, ajustados via `GridSearchCV` (otimizando
ROC-AUC) e tiveram o threshold de decisão ajustado via validação cruzada,
maximizando o **F2-score** — métrica que dá peso maior ao recall, alinhada
com o objetivo do projeto de priorizar a identificação de inadimplentes.

### Comparação dos modelos otimizados

| Métrica | Regressão Logística | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0,716 | 0,770 | **0,775** |
| **Threshold** | 0,31 | 0,15 | 0,34 |
| **Precision** | 0,249 | **0,349** | 0,325 |
| **Recall** | **0,881** | 0,763 | 0,825 |
| **F1-Score** | 0,388 | **0,479** | 0,466 |
| **F2-Score** | 0,584 | 0,617 | **0,631** |

### Matrizes de confusão

| Modelo | TN | FP | FN | TP |
| :--- | ---: | ---: | ---: | ---: |
| **Regressão Logística** | 1.145 | 3.522 | 158 | 1.168 |
| **Random Forest** | 2.776 | 1.891 | 314 | 1.012 |
| **XGBoost** | 2.394 | 2.273 | 232 | 1.094 |

**Modelo final escolhido: XGBoost.** Apesar de a Regressão Logística ter o
maior recall isolado, o XGBoost apresenta o melhor equilíbrio entre
identificar inadimplentes e não gerar excesso de falsos positivos (maior
F2-score e ROC-AUC).

> As métricas acima são geradas automaticamente pela função
> `montar_tabela_comparacao()` (`src/modeling.py`) a partir dos resultados de
> cada modelo, evitando que a tabela fique desatualizada caso os notebooks
> sejam executados novamente com alguma alteração.

## Interpretabilidade (XGBoost)

- **SHAP** foi utilizado para entender a contribuição de cada variável nas
  previsões do modelo final, com gráficos de resumo (`summary_plot`) e de
  dependência (`dependence_plot`) para as variáveis mais relevantes (X6, X12,
  X1).
- O histórico de pagamento mais recente (**X6**) é o fator mais associado à
  inadimplência: valores baixos (atraso) têm forte impacto positivo na
  probabilidade prevista.
- Casos individuais de acerto e erro do modelo (TP, FP, FN, TN) foram
  analisados com `force_plot`, incluindo uma comparação detalhada entre um
  verdadeiro positivo e um falso negativo, para entender em que situações o
  modelo tende a errar.

## Análise de fairness

O modelo final (XGBoost) foi avaliado quanto a disparidade de desempenho
entre subgrupos de duas variáveis sensíveis: **X2 (sexo)** e **X4 (estado
civil)**.

### X2 — Sexo

| Grupo | Accuracy | Precision | Selection Rate | Contagem |
| :---: | :---: | :---: | :---: | ---: |
| 1 | 0,557 | 0,335 | 0,604 | 2.332 |
| 2 | 0,598 | 0,318 | 0,535 | 3.661 |

- **Accuracy Difference:** 0,0416
- **Selection Rate Difference:** 0,0694

Disparidade pequena. Os dois grupos têm amostra grande (>2.300 cada), então
a diferença de ~4 pontos percentuais de acurácia é uma estimativa
relativamente confiável e não indica um problema relevante de equidade.

### X4 — Estado civil

| Grupo | Accuracy | Precision | Selection Rate | Contagem |
| :---: | :---: | :---: | :---: | ---: |
| 1 | 0,581 | 0,346 | 0,578 | 2.756 |
| 2 | 0,585 | 0,307 | 0,545 | 3.172 |
| 3 | 0,462 | 0,273 | 0,677 | 65 |

- **Accuracy Difference:** 0,1236

A maior disparidade encontrada em todo o projeto está aqui: o grupo 3
("outros", códigos não documentados de X4 agrupados no pré-processamento)
apresenta accuracy ~12 pontos percentuais abaixo dos demais.

**Ressalva importante:** esse grupo tem apenas **65 observações**, contra
mais de 2.700 nos outros dois grupos. Com uma amostra tão pequena, o
erro-padrão da estimativa é grande — não é possível distinguir, com os dados
atuais, se essa queda de desempenho reflete:

1. **Ruído estatístico** (amostra pequena demais para confiar no número), ou
2. Uma limitação real do modelo nesse subgrupo, já que "outros" provavelmente
   mistura perfis heterogêneos de estado civil, e o modelo teve poucos
   exemplos para aprender esse padrão.

**Conclusão:** não há evidência de discriminação sistemática do modelo por
sexo. Para estado civil, a maior disparidade está concentrada exatamente no
subgrupo com menor representação amostral, o que limita a confiabilidade da
conclusão — não é possível afirmar nem descartar viés nesse grupo com os
dados disponíveis.

**Recomendação:** antes de qualquer uso em produção, coletar mais exemplos
do grupo "outros" de X4 (ou tratar esse subgrupo separadamente) para permitir
uma avaliação de equidade mais confiável.

## Limitações e próximos passos

- O dataset é de 2005 (Taiwan); padrões de comportamento de crédito podem não
  refletir contextos ou períodos diferentes.
- O subgrupo "outros" de estado civil (X4) tem representação amostral
  insuficiente (65 casos) para uma avaliação de equidade robusta — ver seção
  de fairness.
- O threshold ótimo foi escolhido para maximizar F2 no conjunto de
  treino/validação; validar periodicamente se esse valor continua adequado
  caso o modelo seja reutilizado com novos dados.
- Não há testes automatizados para as funções de `src/` ainda.

## Conclusão

Este projeto teve como objetivo prever a inadimplência de clientes de cartão
de crédito no mês seguinte, a partir de dados históricos de pagamento,
faturas e informações demográficas de 30.000 clientes.

A análise exploratória revelou um conjunto de dados de boa qualidade, sem
valores ausentes, mas com alguns pontos que exigiram decisões cuidadosas: 35
registros duplicados foram removidos, códigos não documentados nas variáveis
de educação e estado civil foram agrupados em categorias "outros", e valores
negativos nas faturas foram preservados por falta de evidência de que fossem
erros — todas essas decisões foram documentadas e centralizadas em
`src/preprocessing.py` para garantir reprodutibilidade.

Três modelos foram treinados e comparados — Regressão Logística, Random
Forest e XGBoost — todos com hiperparâmetros ajustados via `GridSearchCV` e
threshold de decisão otimizado para maximizar o F2-score, priorizando a
identificação de inadimplentes sobre a precisão geral. O XGBoost apresentou o
melhor equilíbrio entre recall e precisão (F2-score de 0,631 e ROC-AUC de
0,775), sendo selecionado como modelo final.

A análise de interpretabilidade via SHAP mostrou que o histórico de
pagamento recente (X6) é o fator mais determinante nas previsões do modelo,
seguido pelos valores de fatura e limite de crédito — um resultado alinhado
com a intuição de negócio de que o comportamento de pagamento recente é o
melhor preditor de inadimplência futura. A análise de casos individuais
(falsos negativos vs. verdadeiros positivos) reforçou esse padrão, mostrando
que o modelo tende a errar justamente quando o histórico recente não é
claramente negativo, mesmo que outros indicadores financeiros já sinalizem
risco.

A avaliação de fairness não identificou disparidade relevante entre grupos
de sexo. Para estado civil, a maior disparidade encontrada está concentrada
no subgrupo "outros", que representa menos de 2% da amostra — o que impede
conclusões definitivas sobre viés nesse grupo específico.

O modelo final foi serializado junto com o threshold otimizado e disponibilizado
através de uma demo interativa (`app.py`), permitindo simular a previsão para
um cliente novo diretamente pela interface, sem precisar rodar os notebooks.

**Limitações principais:** o dataset é de 2005 e de um contexto econômico
específico (Taiwan), o que limita a generalização para outros mercados ou
períodos; o subgrupo "outros" de estado civil tem representação amostral
insuficiente para uma avaliação de equidade robusta; e o modelo ainda não foi
testado em produção ou com dados fora da distribuição de treino.

**Próximos passos:** coletar mais dados do subgrupo sub-representado antes de
qualquer decisão sobre equidade; adicionar testes automatizados para as
funções de `src/`; e revisitar periodicamente o threshold de decisão caso o
modelo seja aplicado a uma nova base de clientes.

## Dependências principais

Ver `requirements.txt`. Principais bibliotecas: `pandas`, `numpy`,
`scikit-learn`, `xgboost`, `shap`, `matplotlib`, `ucimlrepo`, `streamlit`.