# 🚀 PÓS TECH AI Scientist: Portfólio Post Tech 
AI Scientist
Pós Tech
**Integrantes do Grupo:**

| Nome | RM | Contribuições |
|---|---|---|
| Reinaldo Fernandes | RM 371717 | Modelagem IA, Dashboard, Testes (pytest) |
| Leonardo Junior Gonzales Mendoza | RM 373713 | Análise Exploratória (EDA), Aba EDA no Dashboard |
| Winny Tavares | RM 371471 | Documentação, Validação de resultados |

Este repositório centraliza o ecossistema de soluções desenvolvidas durante a pós-graduação na **FIAP**, integrando rigor estatístico, engenharia de machine learning e visão de negócio.

---

## 🔬 Fundamentos Metodológicos — CRISP-DM

Nossa jornada é guiada pelos pilares da ciência de dados moderna:

1. **Metodologia CRISP-DM:** Todos os projetos seguem o ciclo de entendimento de negócio, preparação de dados, modelagem e avaliação.
2. **EDA & Storytelling:** Análises visuais avançadas utilizando *Boxplots*, *Heatmaps*, *Curva de Risco* e *Gráficos de Densidade* para extrair insights acionáveis.
3. **Estatística Inferencial:** Validação de hipóteses através de testes não-paramétricos (**Mann-Whitney U**) para garantir que os resultados não sejam obra do acaso (**p**-valor < 0.05).
4. **Engenharia de ML (MLOps):** Arquitetura modular, gestão de dependências via `requirements.txt` e suítes de testes automatizados com `pytest`.

---

## 🛠️ Ecossistema de Projetos

### 1. 📊 Tech Challenge: Monitor de Risco NPS (Detração Preditiva)

O projeto principal deste repositório. Uma solução *end-to-end* para prever a satisfação do cliente antes mesmo da aplicação da pesquisa.

* **Destaque Técnico:** Identificação do **Ponto de Ruptura Logística** (atrasos > 3 dias).
* **IA:** Classificador *Random Forest* com balanceamento **SMOTE** — AUC-ROC: 0.92.
* **Interface:** Dashboard Premium em **Streamlit** com 3 abas:
  * 🚀 **Predição em Tempo Real** — Termômetro de risco por cliente
  * 🔬 **Análise Exploratória (EDA)** — Storytelling completo com dados reais *(Leonardo)*
  * 📋 **Sobre o Projeto** — Metodologia e equipe

* **Localização:** `src/app_tech_challenge.py`

### 2. 🔬 Análise Exploratória de Dados (Notebook)

Notebook Jupyter documentando toda a jornada analítica dos dados do Tech Challenge.

* **Destaque Técnico:** Storytelling visual com Plotly e validação estatística Mann-Whitney.
* **Contribuição:** Leonardo Junior Gonzales Mendoza — RM 373713
* **Localização:** `notebooks/analise_exploratoria_nps.ipynb`

### 3. 🌍 Motor de Diagnóstico Socioeconômico (IDHM)

Análise profunda de indicadores de desenvolvimento humano e correlações regionais.

* **Destaque Técnico:** Engenharia de correlação entre variáveis de educação, longevidade e renda.
* **Visualização:** Mapas de calor e indicadores de dispersão para identificação de *clusters* de desenvolvimento.
* **Localização:** `Base de dados IDH/`

### 4. 🛡️ Governança e Quality Assurance (QA)

Estrutura de segurança e resiliência para modelos em produção.

* **Destaque Técnico:** Implementação de **Stress Tests** e **Feature Alignment** para evitar falhas de inferência.
* **Testes:** Suíte completa de testes unitários garantindo integridade de 100% nas rotas críticas.
* **Localização:** `tests/test_tech_challenge.py`

---

## ⚙️ Stack Tecnológica

* **Linguagem:** Python 3.12+
* **Ciência de Dados:** Pandas, NumPy, Scikit-Learn, SciPy.
* **Visualização:** Plotly, Seaborn, Matplotlib.
* **Deployment & UI:** Streamlit.
* **Engenharia:** Joblib, Pytest, Git (Controle de Versão).

---

## 🗂️ Estrutura de Pastas

```
payflow-risk-model-crispdm/
├── 📊 src/
│   ├── app_tech_challenge.py       ← Dashboard Streamlit (3 abas)
│   ├── predict_tech_challenge.py   ← Motor de predição
│   └── app_idhm.py                 ← App IDHM
├── 📓 notebooks/
│   └── analise_exploratoria_nps.ipynb  ← EDA completo (Leonardo)
├── 🤖 models/
│   ├── modelo_nps_rf.pkl           ← Modelo treinado
│   └── features_nps.pkl            ← Features do modelo
├── 🗄️ Base de dados Tech Challenge/
│   └── desafio_nps_fase_1.csv      ← Dataset NPS
└── 🧪 tests/
    └── test_tech_challenge.py      ← Testes unitários (pytest)
```

---

## 🚀 Como Explorar este Repositório

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Iniciar o Dashboard de Produção (Tech Challenge)

```bash
streamlit run src/app_tech_challenge.py
```

### 3. Abrir o Notebook de EDA

```bash
jupyter notebook notebooks/analise_exploratoria_nps.ipynb
```

### 4. Executar a Auditoria de Qualidade (Testes)

```bash
pytest -v
```

---

## 📊 Resultados do Modelo

| Métrica | Valor |
|---|---|
| AUC-ROC | **0.92** |
| Threshold de Decisão | 0.35 (otimizado para Recall) |
| Ponto de Ruptura Logística | **3 dias de atraso** |
| Balanceamento | SMOTE aplicado |

---

*Este portfólio demonstra a capacidade técnica do grupo em transformar dados brutos em ativos de decisão estratégica utilizando Inteligência Artificial.*
