# Já que você está na pasta, execute:
cat > README.md << 'EOF'
# 🚀 PayFlow Credit Risk AI - CRISP-DM

Desenvolvimento de uma Inteligência Artificial robusta para prever a **inadimplência de clientes (`default_90d`)** da **PayFlow**, utilizando aprendizado supervisionado focado em **segurança de crédito e mitigação de riscos financeiros**.

---

## 📋 Sobre o Projeto
**Modelo Preditivo para Análise de Risco de Inadimplência**

Este projeto foi desenvolvido para a **Fase 1 da Pós Tech FIAP - AI Scientist**, seguindo a metodologia CRISP-DM para estruturar um modelo de machine learning que prevê o risco de inadimplência de clientes da PayFlow.

---

## 🔄 Metodologia: CRISP-DM
O projeto foi estruturado seguindo o framework **CRISP-DM** para garantir um ciclo de vida de dados escalável e auditável:

| Fase | Descrição | Status |
|------|-----------|--------|
| **Business Understanding** | Alinhamento estratégico para reduzir o risco de crédito e prejuízos operacionais por inadimplência. | ✅ Concluído |
| **Data Understanding** | Análise de 5.000 registros com 12% de classe positiva e identificação proativa de *Data Leakage*. | ✅ Concluído |
| **Data Preparation** | Imputação de nulos via mediana, *Feature Engineering* (`comprometimento_renda`) e remoção de variáveis viciadas. | ✅ Concluído |
| **Modeling** | Implementação de **Random Forest Classifier** e comparação com outros modelos. | ✅ Concluído |
| **Evaluation** | Validação focada na métrica **Recall** para priorizar a captura de clientes em risco real de default. | ✅ Concluído |

---

## 🎯 Business Understanding

### Problema de Negócio
A PayFlow precisa identificar clientes com alto risco de **inadimplência** para:
- Reduzir perdas financeiras
- Otimizar a concessão de crédito
- Criar estratégias diferenciadas por perfil de risco

---

## 📈 Análise da Métrica Recall

O modelo atual tem **Recall = 0.34**, o que significa que capturamos **34% dos inadimplentes**.

**Implicação:** A cada 100 clientes que se tornariam inadimplentes, identificamos 34 corretamente.

### Threshold de Decisão de Negócio
| Probabilidade | Decisão |
|---------------|---------|
| < 20% | ✅ **APROVAR** automaticamente |
| 20% - 40% | ⚠️ **ANÁLISE MANUAL** |
| > 40% | ❌ **NEGAR** crédito |

---

## 🚀 Como Executar

```bash
# Clone o repositório
git clone https://github.com/LeonardoGMendoza/payflow-risk-model-crispdm.git
cd payflow-risk-model-crispdm

# Crie ambiente virtual
python -m venv venv
source venv/Scripts/activate  # Windows

# Instale dependências
pip install -r requirements.txt

# Execute o notebook de análise
jupyter notebook analise_desafio.ipynb
EOF
