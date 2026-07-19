from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    plt = None
    sns = None


ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCAL = ROOT / "data" / "raw" / "manutencao_preditiva.csv"
SOURCE_EXTERNAL = Path(r"C:\Users\User\Documents\Cursos\SCTEC\Projeto_Final\manutencao_preditiva.csv")


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "graficos"
NOTEBOOK_DIR = ROOT / "notebooks"


SENSOR_COLS = [
    "temperatura_ar_k",
    "temperatura_processo_k",
    "velocidade_rotacao_rpm",
    "torque_nm",
]

TARGET = "falha_maquina"
LEAKAGE_COLS = ["falha_twf", "falha_hdf", "falha_pwf", "falha_osf", "falha_rnf"]
DROP_COLS = ["udi", "id_produto", *LEAKAGE_COLS]
PREDICTOR_COLS = [
    "tipo",
    "temperatura_ar_k",
    "temperatura_processo_k",
    "velocidade_rotacao_rpm",
    "torque_nm",
    "desgaste_ferramenta_min",
    "potencia",
]


def setup_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, CHART_DIR, NOTEBOOK_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    if not SOURCE_LOCAL.exists():
        shutil.copy2(SOURCE_EXTERNAL, SOURCE_LOCAL)


def load_and_clean() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(SOURCE_LOCAL)
    missing_before = df.isna().sum().to_dict()
    medians = {}

    for col in SENSOR_COLS:
        medians[col] = float(df[col].median())
        df[col] = df[col].fillna(medians[col])

    df["potencia"] = df["velocidade_rotacao_rpm"] * df["torque_nm"]
    df["classe_manutencao"] = np.where(df[TARGET] == 1, "Falha", "Normal")

    report = {
        "linhas": int(df.shape[0]),
        "colunas_original": 14,
        "colunas_tratado": int(df.shape[1]),
        "missing_before": missing_before,
        "missing_after": df.isna().sum().to_dict(),
        "medianas_imputacao": medians,
        "target_distribution": df[TARGET].value_counts().sort_index().to_dict(),
    }

    df.to_csv(PROCESSED_DIR / "manutencao_preditiva_limpa.csv", index=False)
    return df, report


def iqr_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = SENSOR_COLS + ["desgaste_ferramenta_min", "potencia"]
    rows = []
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        rows.append(
            {
                "variavel": col,
                "q1": q1,
                "q3": q3,
                "limite_inferior": lower,
                "limite_superior": upper,
                "outliers_iqr": count,
            }
        )
    outliers = pd.DataFrame(rows)
    outliers.to_csv(OUTPUT_DIR / "outliers_iqr.csv", index=False)
    return outliers


def make_charts(df: pd.DataFrame) -> None:
    if plt is None or sns is None:
        (CHART_DIR / "AVISO_GRAFICOS.txt").write_text(
            "Os graficos nao foram gerados porque matplotlib/seaborn nao estao "
            "instalados no ambiente atual. Instale requirements.txt e execute "
            "python tools/gerar_projeto.py para gerar as imagens.\n",
            encoding="utf-8",
        )
        return

    sns.set_theme(style="whitegrid", palette="Set2")

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x=TARGET)
    ax.set_title("Distribuicao da variavel alvo")
    ax.set_xlabel("Falha da maquina")
    ax.set_ylabel("Quantidade")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Falha (1)"])
    plt.tight_layout()
    plt.savefig(CHART_DIR / "distribuicao_alvo.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[SENSOR_COLS + ["desgaste_ferramenta_min"]], orient="h")
    plt.title("Boxplots das variaveis explicativas numericas")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "boxplots_variaveis.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="potencia", hue="classe_manutencao", bins=40, kde=True)
    plt.title("Distribuicao da potencia estimada por classe")
    plt.xlabel("potencia = velocidade_rotacao_rpm * torque_nm")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "potencia_por_classe.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    corr_cols = SENSOR_COLS + ["desgaste_ferramenta_min", "potencia", TARGET]
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlacao entre variaveis numericas")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "correlacao.png", dpi=160)
    plt.close()

    aviso = CHART_DIR / "AVISO_GRAFICOS.txt"
    if aviso.exists():
        aviso.unlink()


def balance_train(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    train = X_train.copy()
    temp_target = "_alvo_balanceamento"
    train[temp_target] = y_train.values
    majority = train[train[temp_target] == 0]
    minority = train[train[temp_target] == 1]

    minority_up = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42,
    )
    balanced = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)
    return balanced.drop(columns=[temp_target]), balanced[temp_target]


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    forbidden_cols = {TARGET, *LEAKAGE_COLS}
    X = df[PREDICTOR_COLS].copy()
    X = pd.get_dummies(X, columns=["tipo"], drop_first=True)
    y = df[TARGET]
    leaked_cols = forbidden_cols.intersection(X.columns)
    if leaked_cols:
        raise ValueError(f"Colunas proibidas encontradas em X: {sorted(leaked_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train_bal, y_train_bal = balance_train(X_train, y_train)

    scaler = StandardScaler()
    X_train_knn = scaler.fit_transform(X_train_bal)
    X_test_knn = scaler.transform(X_test)

    rows = []
    best_models = {}

    for k in [3, 5, 7]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_knn, y_train_bal)
        train_acc = accuracy_score(y_train_bal, model.predict(X_train_knn))
        test_pred = model.predict(X_test_knn)
        test_acc = accuracy_score(y_test, test_pred)
        rows.append({"modelo": "KNN", "parametro": f"n_neighbors={k}", "treino": train_acc, "teste": test_acc})
        best_models[f"KNN k={k}"] = (test_acc, model, test_pred)

    for depth in [3, 5, None]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train_bal, y_train_bal)
        train_acc = accuracy_score(y_train_bal, model.predict(X_train_bal))
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        label = "None" if depth is None else str(depth)
        rows.append({"modelo": "Arvore de Decisao", "parametro": f"max_depth={label}", "treino": train_acc, "teste": test_acc})
        best_models[f"Arvore depth={label}"] = (test_acc, model, test_pred)

    metrics = pd.DataFrame(rows)
    metrics["gap_overfitting"] = metrics["treino"] - metrics["teste"]
    metrics.to_csv(OUTPUT_DIR / "metricas_modelos.csv", index=False)

    best_name, (best_acc, _, best_pred) = max(best_models.items(), key=lambda item: item[1][0])
    cm = confusion_matrix(y_test, best_pred)
    report = classification_report(y_test, best_pred, output_dict=True, zero_division=0)

    model_report = {
        "features_usadas": list(X.columns),
        "features_removidas_por_vazamento_ou_identificador": [TARGET, *DROP_COLS],
        "treino_original": int(len(X_train)),
        "teste": int(len(X_test)),
        "treino_balanceado": int(len(X_train_bal)),
        "distribuicao_treino_original": y_train.value_counts().sort_index().to_dict(),
        "distribuicao_treino_balanceado": y_train_bal.value_counts().sort_index().to_dict(),
        "melhor_modelo": best_name,
        "melhor_acuracia_teste": float(best_acc),
        "matriz_confusao_melhor": cm.tolist(),
        "classification_report_melhor": report,
    }
    return metrics, model_report


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def make_readme(data_report: dict, model_report: dict, metrics: pd.DataFrame) -> None:
    best_model = model_report["melhor_modelo"]
    best_acc = model_report["melhor_acuracia_teste"]
    metric_table = metrics.copy()
    metric_table["treino"] = metric_table["treino"].map(lambda v: f"{v:.4f}")
    metric_table["teste"] = metric_table["teste"].map(lambda v: f"{v:.4f}")
    metric_table["gap_overfitting"] = metric_table["gap_overfitting"].map(lambda v: f"{v:.4f}")

    readme = f"""# PredManutencao - Pipeline de Manutencao Preditiva

Projeto avaliativo de IA aplicado a Industria 4.0. O objetivo e prever `falha_maquina` a partir de leituras de sensores e caracteristicas operacionais, evitando vazamento de dados das colunas historicas de causa da falha.

## Estrutura

- `notebooks/projeto_manutencao_preditiva.ipynb`: notebook principal segmentado nas fases do enunciado.
- `data/raw/manutencao_preditiva.csv`: base original.
- `data/processed/manutencao_preditiva_limpa.csv`: base tratada com imputacao e feature `potencia`.
- `outputs/graficos/`: graficos da EDA.
- `outputs/metricas_modelos.csv`: acuracias de treino e teste por hiperparametro.
- `outputs/resumo_execucao.json`: resumo auditavel da limpeza e modelagem.

## Tecnicas utilizadas

- Analise exploratoria com pandas, seaborn e matplotlib.
- Tratamento de nulos com mediana nas variaveis numericas continuas.
- Feature engineering: `potencia = velocidade_rotacao_rpm * torque_nm`.
- Remocao de identificadores e colunas de causa de falha para evitar data leakage.
- Uso exclusivo da base local `data/raw/manutencao_preditiva.csv`.
- Montagem de `X` por lista positiva de colunas permitidas, com validacao para bloquear `falha_maquina` nas preditoras.
- Divisao treino/teste 80/20 com `random_state=42` e estratificacao do alvo.
- Balanceamento por oversampling aplicado somente ao treino.
- Comparacao entre KNN e Arvore de Decisao com ajuste de hiperparametros.

## Resultado principal

Melhor configuracao encontrada: **{best_model}**, com acuracia de teste de **{best_acc:.4f}**.

| Modelo | Parametro | Acuracia treino | Acuracia teste | Gap treino-teste |
|---|---:|---:|---:|---:|
"""
    for _, row in metric_table.iterrows():
        readme += f"| {row['modelo']} | {row['parametro']} | {row['treino']} | {row['teste']} | {row['gap_overfitting']} |\n"

    readme += f"""
## Como executar

1. Crie um ambiente virtual Python.
2. Instale as dependencias:

```bash
pip install -r requirements.txt
```

3. Abra e execute o notebook:

```bash
jupyter notebook notebooks/projeto_manutencao_preditiva.ipynb
```

Tambem e possivel regenerar os artefatos pelo script:

```bash
python tools/gerar_projeto.py
```

## Decisoes tecnicas

As colunas `falha_twf`, `falha_hdf`, `falha_pwf`, `falha_osf` e `falha_rnf` foram mantidas apenas para consulta historica, pois representam causas ja conhecidas da quebra. Usa-las como preditoras inflaria artificialmente a acuracia. A imputacao por mediana foi escolhida por ser robusta a outliers, observados nos boxplots.

A coluna `falha_maquina` e usada somente para criar `y`. Ela nao entra em `X`, e o notebook possui uma validacao explicita para interromper a execucao se essa coluna aparecer nas features.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


def make_requirements() -> None:
    (ROOT / "requirements.txt").write_text(
        "\n".join(
            [
                "pandas",
                "numpy",
                "matplotlib",
                "seaborn",
                "scikit-learn",
                "jupyter",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def nb_cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def make_notebook() -> None:
    code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample

sns.set_theme(style="whitegrid", palette="Set2")
RANDOM_STATE = 42
"""
    cells = [
        nb_cell("markdown", "# PredManutencao - Analise Preditiva de Falhas\n\nNotebook principal do projeto avaliativo, estruturado conforme as fases do enunciado."),
        nb_cell("markdown", "## Fase 1 - Analise Exploratoria (EDA)\n\nCarregamento da base local do projeto, dimensoes, tipos, nulos, distribuicao do alvo e estatisticas iniciais."),
        nb_cell("code", code_imports),
        nb_cell("code", """local_data_paths = [
    Path("data/raw/manutencao_preditiva.csv"),
    Path("../data/raw/manutencao_preditiva.csv"),
]
DATA_PATH = next((path for path in local_data_paths if path.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Base local nao encontrada. Coloque manutencao_preditiva.csv em data/raw/ "
        "e execute o notebook pela raiz do projeto ou pela pasta notebooks/."
    )
df = pd.read_csv(DATA_PATH)
display(df.head())
print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
display(df.dtypes)
display(df.isna().sum())
display(df["falha_maquina"].value_counts(normalize=True).rename("proporcao"))"""),
        nb_cell("code", """numeric_cols = [
    "temperatura_ar_k",
    "temperatura_processo_k",
    "velocidade_rotacao_rpm",
    "torque_nm",
    "desgaste_ferramenta_min",
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x="falha_maquina", ax=axes[0])
axes[0].set_title("Distribuicao do alvo")
axes[0].set_xlabel("falha_maquina")
axes[0].set_ylabel("Quantidade")
sns.heatmap(df[numeric_cols + ["falha_maquina"]].corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=axes[1])
axes[1].set_title("Correlacao numerica")
plt.tight_layout()
plt.show()"""),
        nb_cell("markdown", "## Fase 2 - Limpeza e Tratamento de Dados\n\nOs nulos das variaveis continuas sao tratados com mediana, pois a mediana e menos sensivel a outliers que a media. As colunas de motivo da falha nao entram no modelo para evitar vazamento de dados."),
        nb_cell("code", """sensor_cols = ["temperatura_ar_k", "temperatura_processo_k", "velocidade_rotacao_rpm", "torque_nm"]
leakage_cols = ["falha_twf", "falha_hdf", "falha_pwf", "falha_osf", "falha_rnf"]

medianas = df[sensor_cols].median()
df[sensor_cols] = df[sensor_cols].fillna(medianas)
display(medianas.rename("mediana_usada"))
display(df.isna().sum())"""),
        nb_cell("code", """plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_cols], orient="h")
plt.title("Boxplots para identificacao de outliers")
plt.tight_layout()
plt.show()"""),
        nb_cell("markdown", "## Fase 3 - Feature Engineering\n\nCriacao da variavel numerica `potencia`, usando a sugestao de manutencao do enunciado."),
        nb_cell("code", """df["potencia"] = df["velocidade_rotacao_rpm"] * df["torque_nm"]
df["classe_manutencao"] = np.where(df["falha_maquina"] == 1, "Falha", "Normal")
display(df[["velocidade_rotacao_rpm", "torque_nm", "potencia", "falha_maquina"]].head())

plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="potencia", hue="classe_manutencao", bins=40, kde=True)
plt.title("Potencia estimada por classe")
plt.tight_layout()
plt.show()"""),
        nb_cell("markdown", "## Fase 4 - Divisao e Balanceamento dos Dados\n\nA divisao usa 80% treino e 20% teste com `random_state=42`. Para eliminar a chance de vazamento, `X` e criado somente com colunas preditoras permitidas; a coluna `falha_maquina` fica exclusivamente em `y`. O oversampling e aplicado somente no treino."),
        nb_cell("code", """predictor_cols = [
    "tipo",
    "temperatura_ar_k",
    "temperatura_processo_k",
    "velocidade_rotacao_rpm",
    "torque_nm",
    "desgaste_ferramenta_min",
    "potencia",
]
forbidden_cols = {"falha_maquina", *leakage_cols}

X = df[predictor_cols].copy()
X = pd.get_dummies(X, columns=["tipo"], drop_first=True)
y = df["falha_maquina"]

leaked_cols = forbidden_cols.intersection(X.columns)
assert not leaked_cols, f"Colunas proibidas encontradas em X: {sorted(leaked_cols)}"
print("Colunas usadas no modelo:")
display(pd.Series(X.columns, name="features"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

train = X_train.copy()
temp_target = "_alvo_balanceamento"
train[temp_target] = y_train.values
majority = train[train[temp_target] == 0]
minority = train[train[temp_target] == 1]
minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=RANDOM_STATE)
train_bal = pd.concat([majority, minority_up]).sample(frac=1, random_state=RANDOM_STATE)

X_train_bal = train_bal.drop(columns=[temp_target])
y_train_bal = train_bal[temp_target]

assert "falha_maquina" not in X_train_bal.columns
assert not forbidden_cols.intersection(X_train_bal.columns)

print("Treino original:")
display(y_train.value_counts())
print("Treino balanceado:")
display(y_train_bal.value_counts())"""),
        nb_cell("markdown", "## Fase 5 - Treinamento dos Modelos\n\nSao treinados KNN e Arvore de Decisao. O KNN recebe padronizacao porque depende de distancia entre pontos."),
        nb_cell("code", """scaler = StandardScaler()
X_train_knn = scaler.fit_transform(X_train_bal)
X_test_knn = scaler.transform(X_test)

resultados = []
predicoes = {}

for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_knn, y_train_bal)
    train_acc = accuracy_score(y_train_bal, knn.predict(X_train_knn))
    test_pred = knn.predict(X_test_knn)
    test_acc = accuracy_score(y_test, test_pred)
    resultados.append(["KNN", f"n_neighbors={k}", train_acc, test_acc])
    predicoes[f"KNN k={k}"] = test_pred

for depth in [3, 5, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    tree.fit(X_train_bal, y_train_bal)
    train_acc = accuracy_score(y_train_bal, tree.predict(X_train_bal))
    test_pred = tree.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    label = "None" if depth is None else str(depth)
    resultados.append(["Arvore de Decisao", f"max_depth={label}", train_acc, test_acc])
    predicoes[f"Arvore depth={label}"] = test_pred

metricas = pd.DataFrame(resultados, columns=["modelo", "parametro", "acuracia_treino", "acuracia_teste"])
metricas["gap_overfitting"] = metricas["acuracia_treino"] - metricas["acuracia_teste"]
display(metricas.sort_values("acuracia_teste", ascending=False))"""),
        nb_cell("markdown", "## Fase 6 - Ajuste de Parametros e Combate ao Overfitting\n\nO gap entre treino e teste indica configuracoes com maior risco de overfitting. Configuracoes com alta acuracia de treino e queda relevante no teste devem ser evitadas."),
        nb_cell("code", """plt.figure(figsize=(10, 5))
sns.barplot(data=metricas, x="parametro", y="acuracia_teste", hue="modelo")
plt.ylim(0, 1)
plt.title("Acuracia de teste por configuracao")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

display(metricas[["modelo", "parametro", "acuracia_treino", "acuracia_teste", "gap_overfitting"]])"""),
        nb_cell("markdown", "## Fase 7 - Avaliacao da Acuracia e Veredito Final\n\nA escolha final considera a maior acuracia no conjunto de teste, pois ele representa dados nao vistos durante o treino."),
        nb_cell("code", """melhor_linha = metricas.sort_values("acuracia_teste", ascending=False).iloc[0]
nome_melhor = "KNN k=" + melhor_linha["parametro"].split("=")[1] if melhor_linha["modelo"] == "KNN" else "Arvore depth=" + melhor_linha["parametro"].split("=")[1]
pred_melhor = predicoes[nome_melhor]

print("Melhor modelo:", nome_melhor)
print("Acuracia teste:", round(melhor_linha["acuracia_teste"], 4))
print("\\nMatriz de confusao:")
display(pd.DataFrame(confusion_matrix(y_test, pred_melhor), index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]))
print("\\nRelatorio de classificacao:")
print(classification_report(y_test, pred_melhor, zero_division=0))"""),
        nb_cell("markdown", "## Conclusao\n\nO modelo recomendado e aquele com maior acuracia de teste. As colunas de causas especificas de falha foram removidas das variaveis preditoras para garantir uma avaliacao honesta, sem vazamento de informacao historica."),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOK_DIR / "projeto_manutencao_preditiva.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    setup_dirs()
    df, data_report = load_and_clean()
    outliers = iqr_outlier_report(df)
    make_charts(df)
    metrics, model_report = train_models(df)
    write_json(OUTPUT_DIR / "resumo_limpeza.json", data_report)
    write_json(OUTPUT_DIR / "resumo_modelagem.json", model_report)
    write_json(
        OUTPUT_DIR / "resumo_execucao.json",
        {"limpeza": data_report, "modelagem": model_report, "outliers": outliers.to_dict(orient="records")},
    )
    make_readme(data_report, model_report, metrics)
    make_requirements()
    make_notebook()


if __name__ == "__main__":
    main()
