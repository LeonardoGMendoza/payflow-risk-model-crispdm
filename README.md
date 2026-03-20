# 📊 PayFlow AI: Monitor de Risco de Cliente (NPS & Atraso)

> **Grupo:** Reinaldo Fernandes, Leonardo, Winny
> **Status:** Tech Challenge Fase 1 (FIAP - AI Scientist) - Finalizado 🚀

## 🎯 1. Visão de Negócio e Objetivo

O **PayFlow AI** é uma solução de inteligência preditiva que antecipa o comportamento do cliente no e-commerce. O objetivo técnico é classificar potenciais **Detratores** (NPS 0-6) utilizando variáveis operacionais e logísticas, permitindo ações preventivas de Customer Success.

---

## 🔬 2. Metodologia Científica e Ciclo Analítico (Aulas 01 a 04)

Seguimos o framework **CRISP-DM**, integrando os conceitos fundamentais de Ciência de Dados:

### A. Análise Exploratória de Dados (EDA) & Storytelling

* **Distribuição e Outliers:** Utilizamos **Histogramas** para analisar a frequência de atrasos e **Boxplots** para identificar valores discrepantes que distorcem a média de entrega.
* **Correlação de Pearson:** Geramos um **Heatmap (Matriz de Correlação)** para identificar quais variáveis (ex: Dias de Atraso, Número de Reclamações) possuem relação linear forte com a nota de NPS.
* **Insight Chave:** Validamos que o atraso na entrega é o principal *driver* de detração, com comportamento de queda de satisfação não-linear após o 5º dia.

### B. Estatística para Cientistas de Dados (Aula 05)

* **Probabilidade:** O modelo estima a probabilidade de um evento (Detração), permitindo definir *thresholds* de decisão.
* **Amostragem:** Garantimos a integridade dos dados originais, tratando valores nulos e inconsistentes para evitar o **Bias (Vício)** do modelo.

---

## 🤖 3. Modelagem e Engenharia de IA

* **Algoritmo:** `Random Forest Classifier`. Escolhido por lidar bem com grandes volumes de dados (20 colunas) e fornecer a **Feature Importance** (Explainable AI).
* **Métricas de Performance (Avaliação Técnica):**
  * **Recall (Sensibilidade):** Otimizado para **100%**. Em churn/detração, é vital não deixar nenhum detrator passar despercebido.
  * **AUC-ROC (0.92):** Demonstra a robustez do modelo em distinguir entre Promotores e Detratores em diferentes limiares.
  * **Matriz de Confusão:** Implementada para validar o equilíbrio entre Falsos Positivos e Falsos Negativos.

---

## 🛡️ 4. Governança, QA e Deploy (Aulas de MLOps)

Como especialistas em **Quality Assurance**, aplicamos rigor técnico no produto final:

* **Feature Alignment:** Garantia de que a entrada de dados no Streamlit respeita a tipagem e ordem das 20 features do treino.
* **Robustez:** Tratamento de erros para inputs inválidos no Dashboard.
* **Modularização:** Organização em pasta `src/` para escalabilidade e manutenção do código.

---

## 🚀 5. Como Executar

1. **Instalação:** `pip install -r requirements.txt`
2. **Execução:**

```bash
streamlit run src/app.py
```
