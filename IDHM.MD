# 🌍 Inteligência Territorial: Processamento e Auditoria IDHM

Este módulo faz parte do ecossistema **PayFlow** e tem como objetivo transformar dados brutos do Índice de Desenvolvimento Humano Municipal (IDHM 2010) em inteligência acionável para a expansão de produtos de crédito e análise de mercado.

## 🎯 Objetivo

Implementar um pipeline robusto de **ETL (Extract, Load, Transform)** e **Auditoria de Dados** que garanta a integridade das informações utilizadas em modelos de Machine Learning e dashboards executivos.

## 🏗️ Arquitetura do Módulo

O projeto segue uma estrutura modular para garantir escalabilidade e facilitar a manutenção:

* `src/idhm.py`: O "motor" do módulo. Contém a lógica de localização de arquivos, limpeza e auditoria.
* `tests/test_idhm.py`: Suite de testes unitários para garantir o rigor técnico.
* `src/app_idhm.py`: Interface interativa (Streamlit) para visualização de indicadores e tomadas de decisão.

## ⚙️ Funcionalidades Técnicas

### 1. Pipeline de ETL (idhm.py)

O motor de processamento realiza as seguintes operações:

* **Normalização de Schema** : Converte cabeçalhos complexos para `snake_case`.
* **Seleção Estratégica** : Filtra as 6 colunas críticas (IDHM Geral, Educação, Longevidade e Renda).
* **Engine de Compatibilidade** : Utiliza o motor `xlrd` para garantir a leitura correta de arquivos `.xls` legados.

### 2. Auditoria de Qualidade (Data Quality)

Implementamos a função `run_data_audit`, que atua como um *gatekeeper* de MLOps, validando:

* **Schema** : Presença obrigatória de colunas mapeadas.
* **Integridade** : Zero tolerância para valores nulos no indicador principal.
* **Domínio** : Validação matemática da escala do IDH (entre 0 e 1).
* **Geografia** : Verificação de consistência das 27 Unidades Federativas do Brasil.

## 🧪 Validação e Testes (Pytest)

A robustez do código é validada através de testes automatizados. Para rodar a suite de testes, execute:


```
PYTHONPATH="src" python -m pytest tests/test_idhm.py -v
```


**Resultados esperados:**

* `test_file_discovery`: Sucesso na localização dinâmica do arquivo.
* `test_data_cleaning_logic`: Validação da transformação e tipagem.
* `test_audit_integrity`: Aprovação dos critérios de auditoria técnica.

## 📊 Dashboard Interativo (Streamlit)

A interface permite uma análise exploratória (EDA) em tempo real com os seguintes recursos:

* **Filtros Dinâmicos** : Seleção por UF e piso de IDH.
* **Gráficos de Alta Performance** :
* Ranking de Municípios Premium.
* Correlação entre Renda e Educação (Scatter Plot).
* Distribuição Regional (Boxplot) para identificação de outliers.
* **Auditoria em Tempo Real** : O dashboard executa o `run_data_audit` automaticamente sobre os dados filtrados.


```
streamlit run src/app_idhm.py
```


## 🛠️ Tecnologias Utilizadas

* **Python 3.12+**
* **Pandas** : Manipulação de DataFrames de alta performance.
* **Plotly** : Gráficos interativos e dinâmicos.
* **Streamlit** : Deployment rápido de data apps.
* **Pytest** : Garantia de qualidade e testes unitários.

**Rigor Científico:** Este módulo foi desenvolvido seguindo os princípios de MLOps, garantindo que a base de dados seja auditável e reprodutível para futuras etapas de modelagem preditiva.
