# SysManut_2026 - Manutenção Preditiva — Indústria 4.0

Projeto de Ciência de Dados para prever falhas mecânicas em máquinas industriais a partir de sensores e características operacionais.

## Problema resolvido

Um parque fabril precisa antecipar possíveis quebras mecânicas para reduzir paradas inesperadas na linha de produção. O projeto constrói um pipeline de classificação para prever a variável `falha_maquina` e compara três algoritmos:

- K-Nearest Neighbors (KNN);
- Árvore de Decisão;
- LightGBM.

## Estrutura principal do projeto

```text
projeto_manutencao_preditiva/
├── data/
│   ├── manutencao_preditiva.csv
│   └── manutencao_preditiva_limpa.csv
├── dashboard
│    └── dashboard_monitiramento.py
├── notebooks
├── outputs
├── tools
├── projeto_manutencao_preditiva.ipynb
├── notebook_complementar_modelos_dashboard_lightgbm.ipynb
├── requirements.txt
└── README.md
```

✅ - O arquivo `projeto_manutencao_preditiva.ipynb` é o notebook principal atualizado. As pastas `data/`,`notebooks/`, `outputs/` e `tools/` são auxiliares.

## Etapas do pipeline

1. Carregamento da base local.
2. Análise Exploratória de Dados (EDA).
3. Limpeza e tratamento de valores ausentes.
4. Feature engineering e seleção das variáveis.
5. Separação estratificada entre treino e teste.
6. Balanceamento da base de treino com SMOTE.
7. Treinamento e comparação de KNN, Árvore de Decisão e LightGBM.
8. Avaliação das configurações e análise de overfitting.
9. Recomendação automática do modelo com maior acurácia no teste.

## Técnicas utilizadas

- Análise Exploratória de Dados com pandas, Seaborn e Matplotlib.
- Limpeza de duplicados e tratamento de valores nulos.
- Imputação por mediana nas variáveis numéricas.
- Análise de outliers com boxplots.
- Feature engineering com a variável `potencia_w`.
- Separação treino/teste 80/20 com `stratify=y` e `random_state=42`.
- Balanceamento com SMOTE aplicado somente ao conjunto de treino.
- Escalonamento com `StandardScaler` somente para o KNN.
- Treinamento de modelos baseados em árvores sem escalonamento.
- Ajuste de hiperparâmetros para analisar desempenho e overfitting.
- Avaliação por acurácia, gap treino-teste, matriz de confusão e relatório de classificação.

## Prevenção de vazamento de dados

As seguintes colunas não são usadas como preditoras:

- `falha_maquina`: variável alvo, usada exclusivamente em `y`;
- `falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf` e `falha_rnf`: motivos técnicos históricos de falha;
- `udi` e `id_produto`: identificadores sem valor físico direto para a previsão.

Isso evita que os modelos aprendam a resposta a partir de informações que não deveriam estar disponíveis no momento real da previsão.

## Modelos avaliados

### K-Nearest Neighbors

O KNN é avaliado com diferentes valores de `n_neighbors`. Como calcula distâncias entre observações, utiliza as variáveis padronizadas com `StandardScaler`.

### Árvore de Decisão

A Árvore de Decisão é avaliada com diferentes valores de `max_depth`, permitindo observar a relação entre complexidade, desempenho e risco de overfitting.

### LightGBM

O LightGBM utiliza *gradient boosting* baseado em árvores. O notebook compara configurações com diferentes valores de:

- `n_estimators`;
- `learning_rate`;
- `num_leaves`.

O modelo é treinado com os mesmos dados balanceados por SMOTE usados pelos demais algoritmos. Como modelos baseados em árvores não dependem da escala das variáveis, o LightGBM utiliza os dados imputados sem `StandardScaler`.

## Métricas de avaliação

Para cada configuração são calculadas:

- acurácia no conjunto de treino;
- acurácia no conjunto de teste;
- gap entre treino e teste, usado como indicação de overfitting.

✅ - Os melhores modelos também são avaliados por matriz de confusão, precisão, recall e F1-score. Em manutenção preditiva, o recall da classe de falha merece atenção especial, pois um falso negativo representa uma falha que o sistema deixou de detectar.

## Como executar

1. Clone o repositório e acesse a pasta do projeto:

```bash
cd projeto_Mod-1
```

2. Crie e ative um ambiente virtual, se necessário.


3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Abra o notebook principal:

```bash
jupyter notebook projeto_manutencao_preditiva.ipynb
```

5. Execute todas as células em ordem.

A base deve estar em:

```text
data/manutencao_preditiva.csv
```

Não há busca na Kaggle nem uso de caminho absoluto no notebook.

## Dependências principais

- pandas;
- NumPy;
- Matplotlib;
- Seaborn;
- scikit-learn;
- imbalanced-learn;
- LightGBM;
- Jupyter.

## Observação sobre os resultados

O notebook atualizado foi entregue sem resultados antigos armazenados nas células. Isso evita apresentar métricas calculadas antes da inclusão do LightGBM como se ainda fossem a comparação final.

Após executar todas as células, o notebook gera:

- tabelas comparativas dos hiperparâmetros;
- acurácias de treino e teste;
- gaps de overfitting;
- matrizes de confusão;
- relatórios de classificação;
- recomendação do modelo com maior acurácia no teste.

## Organização sugerida no GitHub

Fluxo recomendado:

```bash
git checkout -b develop
git checkout -b feature/eda
git checkout -b feature/data-prep
git checkout -b feature/modelagem
git checkout -b feature/lightgbm
git checkout -b feature/readme
```

Mensagens de commit sugeridas:

```text
implementa análise exploratória
implementa limpeza e imputação de dados
implementa feature engineering
implementa treino e avaliação dos modelos
adiciona treinamento e avaliação do LightGBM
atualiza documentação e dependências
```

Ao final, faça merge para `main`.

## Complemento: modelos avançados e dashboard

### 🎯 - O arquivo `notebook_complementar_modelos_dashboard_lightgbm.ipynb` adiciona funcionalidades além do escopo do notebook principal:

- avaliação com foco em recall e F1-score da classe de falha;
- teste de Random Forest, Gradient Boosting, três configurações de LightGBM e XGBoost opcional;
- pipeline com `ColumnTransformer`, `StandardScaler`, `OneHotEncoder` e SMOTE;
- salvamento do modelo final com `joblib`;
- exportação de uma comparação complementar de modelos;
- dashboard para monitoramento das previsões.

### Como executar o notebook complementar

```bash
jupyter notebook notebook_complementar_modelos_dashboard_lightgbm.ipynb
```

### Como executar o dashboard

```bash
streamlit run dashboard/dashboard_monitoramento.py
```

O dashboard carrega o melhor pipeline salvo em `models/modelo_final_manutencao_preditiva_lightgbm.joblib`, informa qual algoritmo venceu, calcula a probabilidade de falha por máquina, permite ajustar o threshold e exporta as previsões em CSV.

## Melhorias futuras

- Selecionar hiperparâmetros com validação cruzada e métricas voltadas à classe minoritária.
- Avaliar PR-AUC e ROC-AUC além da acurácia.
- Ajustar o threshold de decisão conforme o custo de falsos negativos e falsos positivos.
- Integrar o melhor modelo ao pipeline operacional e ao dashboard.
- Monitorar degradação de desempenho e mudanças na distribuição dos sensores.

## Observação para o vídeo

No vídeo, explique:

- o objetivo do sistema;
- como instalar as dependências e executar o notebook;
- como o projeto foi organizado em branches;
- por que as colunas de vazamento foram removidas;
- por que o SMOTE foi aplicado somente ao treino;
- por que apenas o KNN foi escalonado;
- como KNN, Árvore de Decisão e LightGBM foram comparados;
- quais hiperparâmetros do LightGBM foram testados;
- qual modelo apresentou o melhor resultado após a execução atualizada;
- por que recall e F1-score também são importantes para falhas raras.
