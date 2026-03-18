# Tech Challenge - Fase 1: NPS Preditivo 🚀

## 🎯 Objetivo do Projeto
Este projeto desenvolve uma solução de **Machine Learning** para antecipar a satisfação do cliente no e-commerce, identificando potenciais **Detratores** através de dados operacionais antes da pesquisa oficial de NPS.

---

## 🎓 Conformidade Acadêmica (Fase 1 - FIAP)
Este projeto consolida 100% dos pilares ensinados na Fase 1, demonstrando a aplicação prática da ementa:

### 1. Metodologia CRISP-DM (Aulas 01 a 04)
O projeto seguiu o framework completo, desde o **Business Understanding** (identificação da dor de churn) até o **Deployment** (serialização do modelo para produção).

### 2. Análise Exploratória & Storytelling (Aulas 01 a 04)
Utilizamos **Seaborn** e **Matplotlib** para validar hipóteses operacionais. A narrativa de dados prova, através de **Boxplots**, que o atraso na entrega é o principal driver de detração, conectando a análise técnica ao valor de negócio.

### 3. Estatística para Cientistas de Dados (Aulas 01 a 05)
- **Correlação de Pearson:** Identificamos a força da relação entre logística e NPS.
- **Avaliação de Performance:** Implementamos métricas de classificação (**AUC-ROC de 0.92** e **Recall de 100%**) para garantir a captura de detratores críticos.
- **Stress Testing:** Validação de hipóteses operacionais em cenários de borda (atrasos > 15 dias).

### 4. Modelagem e Engenharia (Aulas de IA Supervisionada)
Implementamos o **Random Forest Classifier** por sua robustez a outliers e capacidade de gerar **Feature Importance**, garantindo transparência (Explainable AI) sobre os fatores de insatisfação.

---

## 🛠️ Estrutura do Projeto
- `data/`: Bases de dados originais.
- `models/`: Artefatos do modelo treinado (`.pkl`) e lista de features.
- `src/`: Scripts de produção (`predict_nps.py`, `predict_payflow.py`).
- `Fase 1/`: Notebooks com a análise exploratória e científica completa.
- `requirements.txt`: Dependências do projeto.

---

## 🚀 Como Executar
```bash
# Instalação das dependências
pip install -r requirements.txt

# Inferência em Produção (Tech Challenge)
python src/predict_nps.py
```

---

💻 Hardware & Benchmark (Estação de Trabalho)
- CPU: AMD Ryzen 7 9800X3D (Zen 5)

- RAM: 32GB DDR5

- GPU: GeForce RTX 5070 Ti

- Benchmark: Processamento de matrizes de alta densidade em 0.33s.

---

⚖️ Governança e QA
- Data Leakage: Removidas variáveis com dados futuros para evitar vício do modelo.

- Stress Test: Scripts de validação garantem que o modelo responda corretamente a atrasos críticos > 15 dias.
