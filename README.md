# PayFlow Credit Risk AI 🚀

Desenvolvimento de uma Inteligência Artificial robusta para prever a inadimplência de clientes (`default_90d`) da **PayFlow**, utilizando aprendizado supervisionado focado em segurança de crédito e mitigação de riscos financeiros através de rigor estatístico.

---

## 🔄 Metodologia: CRISP-DM
O projeto foi estruturado seguindo o framework **CRISP-DM** para garantir um ciclo de vida de dados escalável e auditável:

* **Business Understanding:** Alinhamento estratégico para reduzir o Default e prejuízos operacionais por inadimplência.
* **Data Understanding:** Análise de 5.000 registros com 12% de classe positiva e identificação proativa de *Data Leakage*.
* **Data Preparation:** Imputação de nulos via mediana, *Feature Engineering* (`comprometimento_renda`) e remoção de variáveis viciadas.
* **Modeling:** Implementação de **Random Forest Classifier** com otimização de hiperparâmetros.
* **Evaluation:** Validação focada na métrica **Recall (0.34)** para priorizar a captura de clientes em risco real de default.

---

## 📊 Dicionário de Dados & Governança

| Nome da Coluna | Descrição | Papel no Modelo |
| :--- | :--- | :--- |
| `default_90d` | Inadimplência superior a 90 dias (0 ou 1). | **Target** |
| `score_credito` | Pontuação de risco do bureau de crédito. | Feature |
| `utilizacao_credito` | Percentual de uso do limite disponível. | Feature |
| `dias_atraso_max_12m` | Maior atraso observado nos últimos 12 meses. | Feature |
| `parcelas_pagas_ate_3m` | Dados gerados após a concessão do crédito. | **LEAKAGE (Removido)** |
| `status_apos_90d` | Status futuro do cliente (vazamento). | **LEAKAGE (Removido)** |

---

## 💻 Hardware & Benchmark (Estação de Trabalho)
Validação de performance em hardware de última geração para garantir a escalabilidade do treinamento:

* **CPU:** AMD Ryzen 7 9800X3D (Zen 5 Architecture).
* **RAM:** 32GB DDR5.
* **Benchmark:** Multiplicação de matrizes $5000 \times 5000$ em **0.3298 segundos** (PyTorch/LibTorch).

---

## 📂 Estrutura do Projeto

```text
├── data/                 # Base de dados (CSV)
├── models/               # Artefatos do modelo treinado (.pkl)
├── tests/                # Testes unitários de integridade (Pytest)
├── analise_desafio.ipynb  # Notebook com EDA e documentação completa
├── predict.py            # Script de produção/inferência
├── benchmark.py          # Script de validação de performance de hardware
├── requirements.txt      # Dependências e versões do projeto
└── .gitignore            # Filtro de arquivos para o Git

```

---

## 🚀 Como executar?

### 1. Configurar Ambiente

```bash
# Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate  # No Linux/Mac
# No Windows: venv\Scripts\activate

# Instalar dependências estáveis
pip install -r requirements.txt

```

### 2. Validar Integridade (Testes & Hardware)

```bash
# Executar suíte de testes unitários
python -m pytest

# Validar performance computacional
python benchmark.py

```

### 3. Realizar Predição (Inferência)

```bash
# Executar o pipeline de predição em produção
python predict.py

```

---

## 🕵️‍♂️ Investigação Final

> "A IA não vai substituir o humano. A IA vai substituir o humano que não sabe usar IA."
> — **Prof. Alexandre Santos (FIAP)**
