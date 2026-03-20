# 🚚 AI Logistics Optimizer - Previsão & Monitoramento v4.0

> **Status do Projeto:** Proativo | Em produção 🚀
> **Papéis:** AI Scientist, ML Engineer & Quality Assurance (QA).

Este projeto evoluiu de um modelo preditivo estatístico para uma **plataforma completa de Inteligência Logística**. Além de prever o tempo de entrega com **XGBoost**, o sistema monitora condições climáticas em tempo real e automatiza o contato ativo com o cliente.

---

## 💡 Inovações da Versão 4.0

* **Monitoramento Geográfico:** Proativo com auditoria de riscos.
* **Core de IA:** XGBoost Regressor + Motor de Decisão Geográfica.
* **Interface:** Dashboard dinâmico com integração de Mapas (Folium).
* **Tecnologias Adicionais:** * `Open-meteo API`: Dados climáticos sem custo de chave.
  * `Geopy`: Conversão de cidades para coordenadas Lat/Log.
  * `Streamlit Session State`: Persistência de auditoria de dados.

---

## 📊 Performance e Inteligência Proativa

O sistema opera em dois pilares fundamentais:

| Pilar               | Funcionalidade          | Objetivo de Negócio                         |
| :------------------ | :---------------------- | :------------------------------------------- |
| **Preditivo** | XGBoost (MAE 5.07 dias) | Precisão no prazo prometido.                |
| **Auditivo**  | Monitoramento Real-time | Detectar atrasos antes que o rastreio falhe. |

**Impacto:** Redução de Churn e do volume de chamados no SAC através de comunicações preventivas.

---

## 🛡️ Garantia de Qualidade (QA) & Governança

Como especialistas em QA, implementamos uma camada de **Testes de Estresse Geográfico**:

* **Validação de Coordenadas:** Filtro automático para evitar erros de renderização (NaN Handling).
* **Simulação de Crise:** Botão de teste para cenários de tempestade e validação de protocolos.
* **Métricas de IA:** Tempo de auditoria inferior a 0.50s para múltiplos pedidos.

---

## 🚀 Como Executar

1. **Ative o ambiente:** `source venv/Scripts/activate`
2. **Instale as dependências:** `pip install -r requirements.txt`
3. **Execute o Monitor:**

```bash
streamlit run src/app.py
```
