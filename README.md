# Previsão de Churn de Clientes

Projeto desenvolvido para um **Checkpoint de Machine Learning - FIAP**.

O objetivo é utilizar técnicas de **classificação supervisionada** para analisar características de clientes de telecomunicações e desenvolver um modelo capaz de estimar a probabilidade de **churn**, ou seja, cancelamento do serviço.

---

# Links

**Repositório:**  
[GitHub](https://github.com/Marirsil/checkpoint-streamlit)

**Aplicação Streamlit:**  
Adicione aqui o link após a publicação no Streamlit Community Cloud.

---

# Tecnologias Utilizadas

<div align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40px" height="40px" alt="Python" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="40px" height="40px" alt="Pandas" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="40px" height="40px" alt="NumPy" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="40px" height="40px" alt="Scikit-learn" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" width="40px" height="40px" alt="Google Colab / Jupyter" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" width="40px" height="40px" alt="Git" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="40px" height="40px" alt="GitHub" />
</div>

---

# Base de Dados

Foi utilizada a base **Telco Customer Churn**, disponibilizada publicamente pela IBM.

A variável resposta utilizada no projeto é:

- `Churn` — indica se o cliente cancelou ou não o serviço.

Entre as principais variáveis utilizadas para previsão estão:

- Gênero
- Indicador de cliente idoso
- Parceiro(a)
- Dependentes
- Tempo como cliente (`tenure`)
- Serviço de telefone
- Serviço de internet
- Segurança online
- Backup online
- Proteção do dispositivo
- Suporte técnico
- Streaming de TV
- Streaming de filmes
- Tipo de contrato
- Fatura digital
- Forma de pagamento
- Cobrança mensal
- Cobrança total

---

# Modelo

Foram comparados diferentes modelos de classificação no notebook.

O modelo final selecionado foi o **XGBoost ajustado por Grid Search**.

Principais hiperparâmetros:

| Parâmetro | Valor |
|---|---:|
| learning_rate | 0.03 |
| max_depth | 3 |
| n_estimators | 300 |

A métrica principal utilizada para comparação dos modelos foi o **ROC-AUC**.

---

# Aplicação Streamlit

Foi desenvolvida uma aplicação em **Streamlit** que permite:

- Visualizar informações da base tratada
- Visualizar estatísticas descritivas
- Visualizar gráficos exploratórios
- Consultar as métricas do modelo final
- Visualizar a matriz de confusão
- Informar características de um cliente
- Gerar uma previsão de churn
- Consultar a probabilidade estimada de cancelamento

---

# Estrutura do Projeto

```text
checkpoint-streamlit/
│
├── app.py
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

# Como Executar

Clone o repositório:

```bash
git clone https://github.com/Marirsil/checkpoint-streamlit.git
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
streamlit run app.py
```

O notebook também pode ser aberto e executado pelo **Google Colab** ou Jupyter Notebook.

---

# Limitações

- O modelo foi treinado a partir de uma base histórica de clientes.
- A previsão representa uma probabilidade estatística e não uma certeza de cancelamento.
- Mudanças no comportamento dos clientes ao longo do tempo podem afetar o desempenho do modelo.
- Clientes com combinações de características pouco representadas na base podem apresentar previsões menos confiáveis.

---


- Augusto Valerio - RM:562185  
  GitHub: https://github.com/Augusto-Valerio

- Jonas Esteves França - RM:564143  
  GitHub: https://github.com/Jonas-Franca

- Mariana Silva Oliveira - RM:564241  
  GitHub: https://github.com/Marirsil

- Pedro Marchese - RM:563339  
  GitHub: https://github.com/PedroMarchese01

- Vitor Rodrigues Tigre - RM:561746  
  GitHub: https://github.com/VitorTigre
