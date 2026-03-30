# 📊 Tech Challenge: Monitor de Risco de Cliente (NPS & Atraso)

**Grupo:** Reinaldo Fernandes, Leonardo, Winny

**Status:** Tech Challenge Fase 1 (FIAP - AI Scientist) - Finalizado 🚀

---

## 🎯 1. Visão de Negócio e Objetivo

O **Tech Challenge: NPS Preditivo** transforma dados operacionais em inteligência antecipatória. O objetivo é classificar potenciais **Detratores** antes mesmo que eles respondam à pesquisa oficial, permitindo que a empresa atue de forma proativa na retenção de clientes.

---

## 🔬 2. Metodologia Científica e Ciclo Analítico

Estruturamos o projeto seguindo rigorosamente os pilares fundamentais do curso:

### 📂 Aula 1: Metodologia CRISP-DM

Adotamos o framework **CRISP-DM** como guia para garantir que o modelo de IA esteja alinhado aos objetivos de negócio.

* **Entendimento do Negócio:** O NPS é o KPI central.
* **Entendimento dos Dados:** Identificação de variáveis críticas de logística.
* **Preparação:** Limpeza e normalização dos dados brutos.

### 📂 Aula 2: Análise Exploratória de Dados (EDA)

Nesta fase, realizamos o  **Storytelling com Dados** :

* **Outliers e Boxplots:** Identificamos como os extremos de atraso impactam drasticamente a percepção de valor.
* **Heatmaps de Correlação:** Provamos visualmente que `delivery_delay_days` é a variável com maior peso negativo na satisfação.
* **Ponto de Ruptura:** Identificamos que após o  **3º dia de atraso** , a chance de o cliente se tornar detrator é superior a 50%.

### 📂 Aula 3: Estatística Essencial para Cientistas de Dados

Aplicamos rigor matemático para validar nossas hipóteses:

* **Teste de Hipótese (Mann-Whitney U):** Provamos estatisticamente (**$p$**-valor < 0.05) que a diferença de notas entre pedidos no prazo e atrasados não é obra do acaso.
* **Probabilidade e Distribuição:** Utilizamos o modelo para estimar a probabilidade de detração, permitindo a definição de *thresholds* de decisão baseados em risco.

### 📂 Aula 4: Ambientes de Trabalho e ML Engineering

Garantimos a reprodutibilidade e escalabilidade do projeto:

* **Configuração de Ambiente:** Uso de Ambientes Virtuais (`venv`) e isolamento de dependências.
* **Arquitetura de Pastas:** Organização profissional separando dados, modelos (`models/`), lógica de predição (`src/`) e testes (`tests/`).

---

## 🤖 3. Modelagem e Engenharia de IA

### Algoritmo e Estratégia

* **Modelo:** `Random Forest Classifier`. Escolhido por fornecer **Feature Importance** (IA Explicável).
* **Balanceamento de Classe:** Implementamos o **SMOTE** para que a IA aprenda corretamente o perfil dos detratores (classe minoritária).

### Métricas de Performance

* **Recall (Sensibilidade):** Otimizado para capturar o máximo de detratores.
* **AUC-ROC (0.92):** Demonstrando alta capacidade de separação entre clientes satisfeitos e insatisfeitos.

---

## 🛡️ 4. Governança, QA e Deploy (MLOps)

* **Modularização:** O motor de predição está isolado para fácil integração em sistemas de terceiros.
* **QA Automatizado (Pytest):** Testes unitários validam se o motor de predição é resiliente e se os alertas **CRÍTICO** disparam no momento certo.
* **Stress Test:** Simulações de crises logísticas para validar a robustez da IA.

---

## 🚀 5. Como Executar


### 1. Instalação (Requisitos da Aula 4)

**Bash**

```
pip install -r requirements.txt
```

### 2. Execução da Predição (Produção)

**Bash**

```
python src/predict_tech_challenge.py
```

### 3. Execução dos Testes (Qualidade)

**Bash**

```
pytest tests/test_tech_challenge.py -v
```

---

*Este projeto é o resultado da integração prática dos conhecimentos adquiridos nos quatro primeiros blocos da Pós-Graduação em AI Scientist.*
