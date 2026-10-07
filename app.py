import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

st.set_page_config(
    page_title="Previsão de Churn",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def carregar_modelo_e_referencia():
    df = pd.read_csv(DATA_URL)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

    mediana_total = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(mediana_total)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Mantém a mesma preparação usada no notebook entregue.
    df = df.drop_duplicates().reset_index(drop=True)

    referencia_raw = df.drop(columns=["Churn"]).copy()

    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    modelo = XGBClassifier(
        learning_rate=0.03,
        max_depth=3,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1
    )

    modelo.fit(X_train, y_train)

    return modelo, referencia_raw, list(X.columns)


def preparar_cliente(dados_cliente, referencia_raw, colunas_modelo):
    novo_cliente = pd.DataFrame([dados_cliente])

    # A concatenação com a base de referência garante que o One-Hot Encoding
    # produza exatamente as mesmas categorias/colunas do treinamento.
    combinado = pd.concat(
        [referencia_raw, novo_cliente],
        ignore_index=True
    )

    combinado_encoded = pd.get_dummies(combinado, drop_first=True)

    cliente_encoded = combinado_encoded.iloc[[-1]].copy()
    cliente_encoded = cliente_encoded.reindex(
        columns=colunas_modelo,
        fill_value=0
    )

    return cliente_encoded


modelo, referencia_raw, colunas_modelo = carregar_modelo_e_referencia()

st.title("📊 Previsão de Churn de Clientes")
st.write(
    "Preencha os dados abaixo para estimar a probabilidade de um cliente "
    "cancelar o serviço."
)

with st.form("form_churn"):
    st.subheader("Dados do cliente")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gênero", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Idoso (65+)", [0, 1])
        Partner = st.selectbox("Possui parceiro(a)", ["No", "Yes"])
        Dependents = st.selectbox("Possui dependentes", ["No", "Yes"])
        tenure = st.number_input(
            "Tempo como cliente (meses)",
            min_value=0,
            max_value=100,
            value=12,
            step=1
        )
        PhoneService = st.selectbox("Serviço de telefone", ["No", "Yes"])
        MultipleLines = st.selectbox(
            "Múltiplas linhas",
            ["No phone service", "No", "Yes"]
        )
        InternetService = st.selectbox(
            "Serviço de internet",
            ["DSL", "Fiber optic", "No"]
        )
        OnlineSecurity = st.selectbox(
            "Segurança online",
            ["No internet service", "No", "Yes"]
        )
        OnlineBackup = st.selectbox(
            "Backup online",
            ["No internet service", "No", "Yes"]
        )

    with col2:
        DeviceProtection = st.selectbox(
            "Proteção do dispositivo",
            ["No internet service", "No", "Yes"]
        )
        TechSupport = st.selectbox(
            "Suporte técnico",
            ["No internet service", "No", "Yes"]
        )
        StreamingTV = st.selectbox(
            "Streaming de TV",
            ["No internet service", "No", "Yes"]
        )
        StreamingMovies = st.selectbox(
            "Streaming de filmes",
            ["No internet service", "No", "Yes"]
        )
        Contract = st.selectbox(
            "Tipo de contrato",
            ["Month-to-month", "One year", "Two year"]
        )
        PaperlessBilling = st.selectbox(
            "Fatura digital",
            ["No", "Yes"]
        )
        PaymentMethod = st.selectbox(
            "Forma de pagamento",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        MonthlyCharges = st.number_input(
            "Cobrança mensal",
            min_value=0.0,
            max_value=500.0,
            value=70.0,
            step=1.0
        )
        TotalCharges = st.number_input(
            "Cobrança total",
            min_value=0.0,
            max_value=20000.0,
            value=840.0,
            step=10.0
        )

    prever = st.form_submit_button("Calcular risco de churn")

if prever:
    dados_cliente = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    cliente_modelo = preparar_cliente(
        dados_cliente,
        referencia_raw,
        colunas_modelo
    )

    probabilidade = float(
        modelo.predict_proba(cliente_modelo)[0, 1]
    )

    previsao = int(probabilidade >= 0.50)

    st.divider()
    st.subheader("Resultado")

    st.metric(
        "Probabilidade de churn",
        f"{probabilidade * 100:.2f}%"
    )

    if previsao == 1:
        st.error("⚠️ Previsão: cliente com risco de churn.")
    else:
        st.success("✅ Previsão: cliente tende a permanecer.")

    st.caption(
        "Classificação realizada com limiar de 50%. "
        "Modelo: XGBoost ajustado por Grid Search."
    )
