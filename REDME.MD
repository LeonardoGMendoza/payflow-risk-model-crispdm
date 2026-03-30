# 🚀 PÓS TECH AI Scientist: Portfólio de Engenharia e Ciência de Dados

**Integrantes do Grupo:**

* Reinaldo Fernandes - RM 371717
* Leonardo Junior Gonzales Mendoza - RM 373713
* Winny Tavares - RM 371471

Este repositório centraliza o ecossistema de soluções desenvolvidas durante a pós-graduação na  **FIAP** , integrando rigor estatístico, engenharia de machine learning e visão de negócio.

---

## 🔬 Fundamentos Metodológicos (Aulas 01 a 04)

Nossa jornada é guiada pelos pilares da ciência de dados moderna:

1. **Metodologia CRISP-DM:** Todos os projetos seguem o ciclo de entendimento de negócio, preparação de dados, modelagem e avaliação.
2. **EDA & Storytelling:** Análises visuais avançadas utilizando  *Boxplots* , *Heatmaps* e *Gráficos de Densidade* para extrair insights acionáveis.
3. **Estatística Inferencial:** Validação de hipóteses através de testes não-paramétricos ( **Mann-Whitney U** ) para garantir que os resultados não sejam obra do acaso (**$p$**-valor < 0.05).
4. **Engenharia de ML (MLOps):** Arquitetura modular, gestão de dependências via `requirements.txt` e suítes de testes automatizados com `pytest`.

---

## 🛠️ Ecossistema de Projetos

### 1. 📊 Tech Challenge: Monitor de Risco NPS (Detração Preditiva)

O projeto principal deste repositório. Uma solução *end-to-end* para prever a satisfação do cliente antes mesmo da aplicação da pesquisa.

* **Destaque Técnico:** Identificação do **Ponto de Ruptura Logística** (atrasos > 3 dias).
* **IA:** Classificador *Random Forest* com balanceamento  **SMOTE** .
* **Interface:** Dashboard Premium em **Streamlit** com termômetro de risco em tempo real.
* **Localização:** `src/app_tech_challenge.py`

### 2. 🌍 Motor de Diagnóstico Socioeconômico (IDHM)

Análise profunda de indicadores de desenvolvimento humano e correlações regionais.

* **Destaque Técnico:** Engenharia de correlação entre variáveis de educação, longevidade e renda.
* **Visualização:** Mapas de calor e indicadores de dispersão para identificação de *clusters* de desenvolvimento.
* **Localização:** `Base de dados IDH/`

### 3. 🛡️ Governança e Quality Assurance (QA)

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

## 🚀 Como Explorar este Repositório

### Instalar Dependências

**Bash**

```
pip install -r requirements.txt
```

### Iniciar o Dashboard de Produção (Tech Challenge)

**Bash**

```
streamlit run src/app_tech_challenge.py
```

### Executar a Auditoria de Qualidade (Testes)

**Bash**

```
pytest -v
```

---

*Este portfólio demonstra a capacidade técnica do grupo em transformar dados brutos em ativos de decisão estratégica utilizando Inteligência Artificial.*
