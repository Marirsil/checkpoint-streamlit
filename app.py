import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

st.set_page_config(
    page_title="Previsão de Churn de Clientes",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def carregar_dados():
    df = pd.read_csv(DATA_URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df = df.drop_duplicates().reset_index(drop=True)
    return df

@st.cache_resource
def treinar_modelo(df):
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
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
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]
    return modelo, X, X_test, y_test, pred, proba

def preparar_cliente(dados_cliente, referencia_raw, colunas_modelo):
    novo_cliente = pd.DataFrame([dados_cliente])
    combinado = pd.concat([referencia_raw, novo_cliente], ignore_index=True)
    combinado_encoded = pd.get_dummies(combinado, drop_first=True)
    cliente_encoded = combinado_encoded.iloc[[-1]].copy()
    cliente_encoded = cliente_encoded.reindex(columns=colunas_modelo, fill_value=0)
    return cliente_encoded

df = carregar_dados()
modelo, X, X_test, y_test, pred_test, proba_test = treinar_modelo(df)
referencia_raw = df.drop(columns=["Churn"]).copy()
colunas_modelo = list(X.columns)

roc_auc = roc_auc_score(y_test, proba_test)
acuracia = accuracy_score(y_test, pred_test)
precisao = precision_score(y_test, pred_test)
recall = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
matriz_confusao = confusion_matrix(y_test, pred_test)

col_img, col_titulo = st.columns([1, 7])
with col_img:
    st.markdown("<div style='font-size:76px;text-align:center;'>📊</div>", unsafe_allow_html=True)
with col_titulo:
    st.title("Previsão de Churn de Clientes")
    st.caption("Modelo de classificação para estimativa do risco de cancelamento")

st.write("""
Este projeto utiliza **XGBoost** para prever a probabilidade de churn de clientes de uma empresa de telecomunicações.
O modelo considera informações demográficas, serviços contratados, tipo de contrato, forma de pagamento, tempo de permanência e valores cobrados ao cliente.
""")

st.markdown("""
**Fonte dos dados:** Telco Customer Churn Dataset, disponibilizado publicamente pela IBM.

**Variável resposta (y):** `Churn` — indica se o cliente cancelou ou não o serviço.

**Variáveis explicativas (X):** características demográficas, serviços contratados, contrato, cobrança e histórico do cliente.
""")

st.divider()
st.header("1. Exploração dos dados")
st.subheader("Amostra da base tratada")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Estatísticas descritivas")
st.dataframe(df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].describe().round(2), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribuição de churn")
    contagem = df["Churn"].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.bar(["Não", "Sim"], [contagem.get(0, 0), contagem.get(1, 0)])
    ax1.set_title("Distribuição da variável Churn")
    ax1.set_ylabel("Quantidade de clientes")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.subheader("Cobrança mensal × churn")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(df.loc[df["Churn"] == 0, "MonthlyCharges"], bins=30, alpha=0.60, label="Não")
    ax2.hist(df.loc[df["Churn"] == 1, "MonthlyCharges"], bins=30, alpha=0.60, label="Sim")
    ax2.set_title("MonthlyCharges por classe de churn")
    ax2.set_xlabel("Cobrança mensal")
    ax2.set_ylabel("Frequência")
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()
st.header("2. Avaliação do modelo final")
st.write("""
O modelo final selecionado foi o **XGBoost ajustado por Grid Search**.
As métricas abaixo são calculadas no conjunto de teste com `test_size=0.20`, `random_state=42` e estratificação pela variável resposta.
""")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("ROC-AUC", f"{roc_auc:.4f}")
m2.metric("Acurácia", f"{acuracia:.4f}")
m3.metric("Precisão", f"{precisao:.4f}")
m4.metric("Recall", f"{recall:.4f}")
m5.metric("F1-score", f"{f1:.4f}")

col3, col4 = st.columns(2)
with col3:
    st.subheader("Matriz de confusão")
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.imshow(matriz_confusao)
    for i in range(matriz_confusao.shape[0]):
        for j in range(matriz_confusao.shape[1]):
            ax3.text(j, i, matriz_confusao[i, j], ha="center", va="center")
    ax3.set_title("Matriz de confusão")
    ax3.set_xlabel("Previsto")
    ax3.set_ylabel("Real")
    ax3.set_xticks([0, 1], labels=["Não", "Sim"])
    ax3.set_yticks([0, 1], labels=["Não", "Sim"])
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

with col4:
    st.subheader("Distribuição das probabilidades")
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    y_arr = y_test.to_numpy()
    ax4.hist(proba_test[y_arr == 0], bins=30, alpha=0.60, label="Não churn")
    ax4.hist(proba_test[y_arr == 1], bins=30, alpha=0.60, label="Churn")
    ax4.axvline(0.50, linestyle="--", label="Limiar 0,50")
    ax4.set_title("Probabilidade prevista de churn")
    ax4.set_xlabel("Probabilidade")
    ax4.set_ylabel("Frequência")
    ax4.legend()
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

st.divider()
st.header("3. Faça uma nova previsão")
st.write("Preencha as características do cliente abaixo. Os dados serão preparados com o mesmo conjunto de variáveis utilizado no treinamento.")

with st.form("form_previsao"):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gênero", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Cliente idoso", [0, 1])
        Partner = st.selectbox("Possui parceiro(a)", ["No", "Yes"])
        Dependents = st.selectbox("Possui dependentes", ["No", "Yes"])
        tenure = st.number_input("Tempo como cliente (meses)", min_value=0, max_value=100, value=12, step=1)
        PhoneService = st.selectbox("Serviço de telefone", ["No", "Yes"])
        MultipleLines = st.selectbox("Múltiplas linhas", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Serviço de internet", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Segurança online", ["No internet service", "No", "Yes"])
        OnlineBackup = st.selectbox("Backup online", ["No internet service", "No", "Yes"])
    with c2:
        DeviceProtection = st.selectbox("Proteção do dispositivo", ["No internet service", "No", "Yes"])
        TechSupport = st.selectbox("Suporte técnico", ["No internet service", "No", "Yes"])
        StreamingTV = st.selectbox("Streaming de TV", ["No internet service", "No", "Yes"])
        StreamingMovies = st.selectbox("Streaming de filmes", ["No internet service", "No", "Yes"])
        Contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Fatura digital", ["No", "Yes"])
        PaymentMethod = st.selectbox("Forma de pagamento", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        MonthlyCharges = st.number_input("Cobrança mensal", min_value=0.0, max_value=500.0, value=70.0, step=1.0)
        TotalCharges = st.number_input("Cobrança total", min_value=0.0, max_value=20000.0, value=840.0, step=10.0)
    enviar = st.form_submit_button("Calcular risco de churn")

if enviar:
    dados_cliente = {
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents,
        "tenure": tenure, "PhoneService": PhoneService, "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies, "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges
    }
    cliente_modelo = preparar_cliente(dados_cliente, referencia_raw, colunas_modelo)
    probabilidade = float(modelo.predict_proba(cliente_modelo)[0, 1])
    previsao = int(probabilidade >= 0.50)
    st.divider()
    st.subheader("Resultado da previsão")
    r1, r2 = st.columns(2)
    r1.metric("Probabilidade de churn", f"{probabilidade * 100:.2f}%")
    r2.metric("Classificação", "Churn" if previsao == 1 else "Não churn")
    if previsao == 1:
        st.error("⚠️ O cliente apresenta risco de cancelamento.")
    else:
        st.success("✅ O cliente tende a permanecer.")
    st.caption("A classificação utiliza limiar de 50%. A previsão representa uma estimativa estatística e não deve ser interpretada como certeza de cancelamento.")
    with st.expander("Ver dados enviados ao modelo"):
        st.dataframe(pd.DataFrame([dados_cliente]), use_container_width=True)
