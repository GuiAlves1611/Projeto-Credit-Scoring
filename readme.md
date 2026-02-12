# 📊 Credit Scoring — Modelagem de Risco de Crédito End-to-End

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-black?style=for-the-badge&logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## 🚀 Visão Geral

Este projeto simula o desenvolvimento de um modelo de **Credit Scoring** completo, desde o tratamento de dados brutos em ambiente SQL até o deploy de uma aplicação preditiva funcional. 

O diferencial desta solução é o **Pipeline Híbrido**: o **PostgreSQL** foi utilizado para o processamento massivo e extração de regras de negócio (Feature Engineering), enquanto o **Ambiente Python** foi aplicado para a modelagem estatística e criação da interface de decisão.

---

## 🏗️ Arquitetura do Projeto

### 1️⃣ Engenharia de Dados (PostgreSQL)
A inteligência do dado começa no banco de dados. Antes da modelagem, o **PostgreSQL** foi utilizado para transformar dados transacionais brutos em uma **ABT (Analytical Base Table)** consolidada.
* **Feature Engineering via SQL:** Uso de *Window Functions* e *CTEs* para calcular variáveis históricas e status recente.
* **Construção do Target:** Definição lógica da inadimplência processada diretamente no banco.
* **Exportação Otimizada:** Preparação do dataset final para garantir performance e integridade durante o treinamento.

### 2️⃣ Inteligência Preditiva (Python & XGBoost)
No ambiente de desenvolvimento Python, o trabalho seguiu focado em:
* **Modelagem:** Implementação do algoritmo **XGBoost**, otimizando a capacidade de separação entre bons e maus pagadores.
* **Métricas de Performance:** O modelo apresentou **excelente capacidade discriminatória**, com métricas de **AUC 91%** e **RECALL 80%** entre treino e teste, garantindo robustez e baixa variância.
* **Validação de Estabilidade:** Testes rigorosos para garantir que o modelo seja generalizável e livre de *data leakage*.

### 3️⃣ Metodologia de Score Bancário (PDO)
Para traduzir a probabilidade estatística em uma métrica de negócio, aplicamos a metodologia de **Points to Double the Odds (PDO)**:
$$Score = Offset + Factor \cdot \ln(Odds)$$
* **Configuração:** PDO 60 / Base Score 400.
* Esta abordagem garante **explicabilidade**, permitindo que o negócio compreenda o risco de forma clara e padronizada.

---

## 📊 Impacto Simulado e Resultados de Negócio

Este projeto não entrega apenas um modelo, mas uma **base para política de crédito escalável**. O impacto esperado inclui:

* **Redução Estimada de Inadimplência:** Melhor identificação de perfis de alto risco (*default*), permitindo barrar propostas nocivas à carteira.
* **Melhor Separação de Risco:** Diferenciação precisa entre clientes "VIP", "Regulares" e "Risco", otimizando a oferta de produtos financeiros.
* **Política Escalável:** Automação de regras que reduz o tempo de análise manual e permite o crescimento da base de clientes com segurança.
* **Pronto para Integração:** Arquitetura modular que facilita a exposição do modelo via API para sistemas de originação.

---

## 📈 Estratégia de Crédito e Análise "What-If"

O projeto utiliza réguas de corte (*cut-offs*) estratégicas para definir o apetite de risco da instituição:

* **Aprovação Automática:** Baixíssimo risco e alta probabilidade de adimplência.
* **Aprovação com Restrição:** Clientes intermediários, sugerindo limites reduzidos ou garantias.
* **Reprovação:** Perfis de alto risco identificados preventivamente para mitigação de perdas.

---

## 🖥️ Aplicação Streamlit
Interface interativa que permite simular o score de novos proponentes em tempo real e visualizar o impacto das variáveis na decisão final de crédito.

---

## 📂 Estrutura do Repositório
```bash
├── app/                # Aplicação interativa (Streamlit)
├── business_notes/     # Documentação de regras de decisão e negócio
├── data/               # Camada de dados (Raw, Clean e Features)
│   ├── credit/         # Datasets analíticos (CSV/Parquet)
│   └── fraud/          # Dados complementares transacionais
├── models/             # Artefatos do modelo treinado (Pipelines e Encoders .pkl)
├── notebooks/          # Experimentos de EDA, Cleaning e Treinamento
├── src/                # Código fonte modular (Dataset Builder, Scoring e Pipelines)
├── .gitignore          # Arquivos ignorados pelo Git
├── readme.md           # Documentação principal
└── requirements.txt    # Dependências do projeto
