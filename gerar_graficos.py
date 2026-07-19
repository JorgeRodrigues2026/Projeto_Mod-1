from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "manutencao_preditiva_limpa.csv"
CHART_DIR = ROOT / "outputs" / "graficos"

SENSOR_COLS = [
    "temperatura_ar_k",
    "temperatura_processo_k",
    "velocidade_rotacao_rpm",
    "torque_nm",
]


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    sns.set_theme(style="whitegrid", palette="Set2")

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="falha_maquina")
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
    corr_cols = SENSOR_COLS + ["desgaste_ferramenta_min", "potencia", "falha_maquina"]
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Correlacao entre variaveis numericas")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "correlacao.png", dpi=160)
    plt.close()

    aviso = CHART_DIR / "AVISO_GRAFICOS.txt"
    if aviso.exists():
        aviso.unlink()


if __name__ == "__main__":
    main()
