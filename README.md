# Checkpoint — Previsão de Churn com Streamlit

Aplicação desenvolvida para o Checkpoint de Machine Learning.

## Objetivo

Prever a probabilidade de **churn** de clientes de telecomunicações a partir do conjunto de dados Telco Customer Churn.

## Modelo

O modelo selecionado no notebook foi:

- **XGBoost**
- Estratégia de ajuste: **Grid Search**
- `learning_rate = 0.03`
- `max_depth = 3`
- `n_estimators = 300`
- Métrica principal: **ROC-AUC**

A aplicação reproduz o mesmo pré-processamento e a mesma divisão treino/teste utilizada no notebook, com `random_state=42`.

## Executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Arquivos principais

- `app.py`: aplicação Streamlit.
- `requirements.txt`: dependências.
- Notebook do Checkpoint: contém EDA, comparação dos modelos, Grid Search, Optuna e seleção do modelo final.

## Base de dados

Telco Customer Churn, disponibilizada publicamente pela IBM.

## Uso

Preencha os dados de um cliente e clique em **Calcular risco de churn**. O app exibirá a probabilidade estimada de cancelamento e a classificação com limiar de 50%.
