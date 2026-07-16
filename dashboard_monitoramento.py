from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "modelo_final_manutencao_preditiva_lightgbm.joblib"
DEFAULT_DATA_PATH = ROOT / "data" / "manutencao_preditiva.csv"

COLUNAS_VAZAMENTO = [
    "falha_maquina",
    "falha_twf",
    "falha_hdf",
    "falha_pwf",
    "falha_osf",
    "falha_rnf",
]
COLUNAS_IDENTIFICACAO = ["udi", "id_produto"]
COLUNAS_NECESSARIAS_ENGENHARIA = [
    "velocidade_rotacao_rpm",
    "torque_nm",
    "temperatura_processo_k",
    "temperatura_ar_k",
]


@st.cache_resource(show_spinner="Carregando o modelo preditivo...")
def carregar_modelo(caminho: Path) -> dict:
    """Carrega o artefato gerado pelo notebook complementar."""
    return joblib.load(caminho)


@st.cache_data(show_spinner=False)
def carregar_csv_padrao(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho)


def validar_artefato(artefato: dict) -> None:
    campos_obrigatorios = {"pipeline", "threshold", "features", "metrics"}
    campos_ausentes = campos_obrigatorios.difference(artefato)
    if campos_ausentes:
        raise ValueError(
            "O artefato do modelo está incompleto. Campos ausentes: "
            + ", ".join(sorted(campos_ausentes))
        )


def preparar_dados(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Replica a preparação usada no notebook complementar."""
    dados = df.drop_duplicates().copy()

    colunas_base_ausentes = [
        coluna for coluna in COLUNAS_NECESSARIAS_ENGENHARIA
        if coluna not in dados.columns
    ]
    if colunas_base_ausentes:
        raise ValueError(
            "O CSV não contém todas as leituras necessárias: "
            + ", ".join(colunas_base_ausentes)
        )

    for coluna in dados.select_dtypes(include="number").columns:
        mediana = dados[coluna].median()
        if pd.isna(mediana):
            raise ValueError(
                f"A coluna numérica '{coluna}' não possui nenhum valor válido."
            )
        dados[coluna] = dados[coluna].fillna(mediana)

    for coluna in dados.select_dtypes(exclude="number").columns:
        moda = dados[coluna].mode()
        preenchimento = moda.iloc[0] if not moda.empty else "desconhecido"
        dados[coluna] = dados[coluna].fillna(preenchimento)

    dados["potencia_operacional"] = (
        dados["velocidade_rotacao_rpm"] * dados["torque_nm"]
    )
    dados["delta_temperatura"] = (
        dados["temperatura_processo_k"] - dados["temperatura_ar_k"]
    )

    dados = dados.drop(
        columns=[
            coluna
            for coluna in COLUNAS_VAZAMENTO + COLUNAS_IDENTIFICACAO
            if coluna in dados.columns
        ],
        errors="ignore",
    )

    features_ausentes = [feature for feature in features if feature not in dados.columns]
    if features_ausentes:
        raise ValueError(
            "O CSV não permite reconstruir todas as variáveis do modelo: "
            + ", ".join(features_ausentes)
        )

    return dados[features]


st.title("Dashboard de Monitoramento — Manutenção Preditiva")
st.caption(
    "Monitoramento baseado no melhor pipeline selecionado entre ensembles e "
    "as configurações avaliadas do LightGBM."
)

if not MODEL_PATH.exists():
    st.error(
        "Modelo não encontrado. Execute primeiro o notebook "
        "`notebook_complementar_modelos_dashboard_lightgbm.ipynb` para gerar "
        f"`{MODEL_PATH.relative_to(ROOT)}`."
    )
    st.stop()

try:
    artefato = carregar_modelo(MODEL_PATH)
    validar_artefato(artefato)
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo: {erro}")
    st.stop()

pipeline = artefato["pipeline"]
threshold_modelo = float(artefato["threshold"])
features = list(artefato["features"])
metricas = artefato["metrics"]
modelo_nome = artefato.get("modelo_nome", metricas.get("modelo", "Não informado"))
configuracoes_lightgbm = artefato.get("configuracoes_lightgbm", [])

st.sidebar.header("Configurações")
st.sidebar.success(f"Modelo carregado: {modelo_nome}")

threshold_inicial = min(max(threshold_modelo, 0.05), 0.95)
threshold_dashboard = st.sidebar.slider(
    "Threshold de alerta de falha",
    min_value=0.05,
    max_value=0.95,
    value=threshold_inicial,
    step=0.05,
    help=(
        "Valores menores tendem a elevar o recall, mas podem aumentar "
        "a quantidade de falsos positivos."
    ),
)

uploaded_file = st.sidebar.file_uploader(
    "Enviar novo CSV para monitoramento",
    type=["csv"],
)

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        origem_dados = uploaded_file.name
    else:
        if not DEFAULT_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Base padrão não encontrada em {DEFAULT_DATA_PATH}."
            )
        df = carregar_csv_padrao(DEFAULT_DATA_PATH)
        origem_dados = str(DEFAULT_DATA_PATH.relative_to(ROOT))

    if df.empty:
        raise ValueError("O CSV selecionado não contém registros.")

    X_monitoramento = preparar_dados(df, features)
except Exception as erro:
    st.error(f"Não foi possível preparar os dados: {erro}")
    st.stop()

if not hasattr(pipeline, "predict_proba"):
    st.error("O pipeline salvo não oferece probabilidades com `predict_proba`.")
    st.stop()

try:
    probabilidades = pipeline.predict_proba(X_monitoramento)[:, 1]
except Exception as erro:
    st.error(f"Falha durante a geração das previsões: {erro}")
    st.stop()

predicoes = (probabilidades >= threshold_dashboard).astype(int)

resultado = df.loc[X_monitoramento.index].copy()
resultado["probabilidade_falha"] = probabilidades
resultado["alerta_falha"] = predicoes

st.caption(f"Fonte monitorada: `{origem_dados}`")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Modelo ativo", modelo_nome)
col2.metric("Registros", len(resultado))
col3.metric("Alertas", int(resultado["alerta_falha"].sum()))
col4.metric("Taxa de alerta", f"{resultado['alerta_falha'].mean() * 100:.2f}%")
col5.metric("Threshold", f"{threshold_dashboard:.2f}")

aba_monitoramento, aba_modelo, aba_dados = st.tabs(
    ["Monitoramento", "Modelo e métricas", "Base completa"]
)

with aba_monitoramento:
    st.subheader("Distribuição dos alertas")
    distribuicao = (
        resultado["alerta_falha"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(index={0: "Normal", 1: "Alerta"})
    )
    st.bar_chart(distribuicao)

    st.subheader("Máquinas com maior risco previsto")
    colunas_exibir = [
        coluna
        for coluna in [
            "udi",
            "id_produto",
            "tipo",
            "probabilidade_falha",
            "alerta_falha",
        ]
        if coluna in resultado.columns
    ]
    top_risco = resultado.sort_values(
        "probabilidade_falha",
        ascending=False,
    ).head(20)
    st.dataframe(top_risco[colunas_exibir], use_container_width=True)

with aba_modelo:
    st.subheader("Métricas do pipeline selecionado")
    st.dataframe(pd.DataFrame([metricas]), use_container_width=True)

    st.write(f"**Threshold escolhido no treinamento:** {threshold_modelo:.2f}")
    st.write(f"**Quantidade de features esperadas:** {len(features)}")

    if configuracoes_lightgbm:
        st.subheader("Configurações de LightGBM avaliadas")
        st.dataframe(
            pd.DataFrame(configuracoes_lightgbm),
            use_container_width=True,
            hide_index=True,
        )

with aba_dados:
    st.subheader("Base monitorada com previsões")
    st.dataframe(resultado, use_container_width=True)

csv = resultado.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Baixar previsões em CSV",
    data=csv,
    file_name="previsoes_manutencao_preditiva_lightgbm.csv",
    mime="text/csv",
)
