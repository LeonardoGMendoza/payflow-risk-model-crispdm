import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import os

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Tech Challenge: Monitor de Risco NPS",
    page_icon="📊",
    layout="wide"
)

# Custom CSS para visual premium (Texto escuro sobre fundo branco nas métricas)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    /* Estilização dos cards de métricas */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e6e9ef;
    }
    /* Ajuste de Cor dos Títulos das Métricas */
    [data-testid="stMetricLabel"] p {
        color: #555e6d !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    /* Ajuste de Cor dos Valores das Métricas */
    [data-testid="stMetricValue"] div {
        color: #1f77b4 !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# CARREGAMENTO DO MODELO E UTILITÁRIOS (MLOps Robustness)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    """Carrega o modelo e as features garantindo o caminho correto mesmo dentro de /src."""
    try:
        current_dir = Path(__file__).resolve().parent
        root_path = current_dir.parent if current_dir.name == 'src' else current_dir
        
        model_path = root_path / "models" / "modelo_nps_rf.pkl"
        features_path = root_path / "models" / "features_nps.pkl"
        
        if not model_path.exists():
            return None, None
            
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except:
        return None, None

model_nps, feature_list = load_ml_assets()

def calculate_risk(data, model, features):
    """Executa a predição com alinhamento de features e engenharia em tempo real."""
    df_input = pd.DataFrame([data])
    df_input['delay_ratio'] = df_input['delivery_delay_days'] / (df_input['delivery_time_days'] + 1)
    
    df_final = pd.DataFrame(0, index=[0], columns=features)
    for col in df_input.columns:
        if col in df_final.columns:
            df_final[col] = df_input[col].values
            
    prob = model.predict_proba(df_final)[:, 1][0]
    return prob

# ------------------------------------------------------------------------------
# SIDEBAR - ENTRADA DE DADOS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("PayFlow AI Engine")
    st.markdown("---")
    
    st.subheader("📋 Dados do Pedido")
    age = st.slider("Idade do Cliente", 18, 90, 35)
    tenure = st.number_input("Meses de Relacionamento", 0, 240, 12)
    order_val = st.number_input("Valor do Pedido (R$)", 0.0, 5000.0, 250.0)
    items = st.slider("Qtd de Itens", 1, 20, 15)
    
    st.subheader("🚚 Logística")
    delivery_days = st.number_input("Prazo de Entrega (Dias)", 1, 30, 3)
    delay_days = st.number_input("Dias de Atraso", 0, 30, 3)
    
    st.subheader("🎧 Atendimento")
    contacts = st.number_input("Contatos no Suporte", 0, 10, 0)
    
    region = st.selectbox("Região do Cliente", ["Sudeste", "Sul", "Nordeste", "Norte", "Centro-Oeste"])

    st.markdown("---")
    if st.button("🚀 Analisar Risco de Detração", use_container_width=True):
        st.session_state.run_analysis = True
    else:
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False

# ------------------------------------------------------------------------------
# PAINEL PRINCIPAL
# ------------------------------------------------------------------------------
st.title("📊 Tech Challenge: Monitor de Risco de Cliente (NPS)")
st.markdown("Diagnóstico preditivo baseado em IA para antecipar insatisfação.")

if not model_nps:
    st.warning("⚠️ Modelo não carregado. Verifique a pasta 'models'.")
    st.stop()

if st.session_state.run_analysis:
    input_dict = {
        'customer_age': age, 'customer_tenure_months': tenure, 'order_value': order_val,
        'items_quantity': items, 'delivery_time_days': delivery_days, 'delivery_delay_days': delay_days,
        'customer_service_contacts': contacts, f'customer_region_{region}': 1
    }
    
    risk_prob = calculate_risk(input_dict, model_nps, feature_list)
    risk_percent = risk_prob * 100
    
    # Grid de KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probabilidade de Detração", f"{risk_prob:.1%}")
    with col2:
        status = "CRÍTICO" if risk_prob > 0.35 else "SEGURO"
        color = "#ff4b4b" if status == "CRÍTICO" else "#00d4ff"
        st.markdown(f"**Status da Experiência:** <h2 style='color:{color}; margin-top:-15px;'>{status}</h2>", unsafe_allow_html=True)
    with col3:
        ruptura = "Sim" if delay_days >= 3 else "Não"
        st.metric("Ponto de Ruptura Atingido?", ruptura)

    st.markdown("---")
    
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.subheader("🌡️ Termômetro de Risco")
        # PONTEIRO DINÂMICO: A linha vermelha (threshold) agora segue o valor do risco calculado
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_percent,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#1f77b4", 'thickness': 0.6},
                'bgcolor': "rgba(0,0,0,0.1)",
                'borderwidth': 2,
                'bordercolor': "#444",
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(0, 255, 0, 0.15)'},
                    {'range': [35, 70], 'color': 'rgba(255, 255, 0, 0.15)'},
                    {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.15)'}],
                'threshold': {
                    'line': {'color': "red", 'width': 6},
                    'thickness': 0.8,
                    'value': risk_percent # O ponteiro acompanha o valor exato!
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "white", 'family': "Arial"},
            height=380, margin=dict(l=30, r=30, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c_right:
        st.subheader("💡 Recomendações Estratégicas")
        if risk_prob > 0.7:
            st.error("🚨 **Ação Imediata:** Risco extremo. Disparar cupom de desconto ou contato telefônico prioritário.")
        elif risk_prob > 0.35:
            st.warning("⚠️ **Alerta Preventivo:** Atraso identificado. Enviar notificação push proativa.")
        else:
            st.success("✅ **Fidelização:** Cliente satisfeito. Momento ideal para oferta de fidelidade.")

        st.markdown("**Impacto Estimado do Atraso no Risco:**")
        delays = list(range(0, 11))
        probs = [calculate_risk({**input_dict, 'delivery_delay_days': d}, model_nps, feature_list) for d in delays]
        fig_line = px.line(x=delays, y=probs, labels={'x':'Dias de Atraso', 'y':'Probabilidade'}, template="plotly_dark")
        fig_line.add_hline(y=0.35, line_dash="dash", line_color="red", annotation_text="Limite de Ruptura (35%)")
        fig_line.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_line, use_container_width=True)

else:
    st.info("👈 Ajuste as variáveis operacionais e clique em 'Analisar Risco' para iniciar.")
    st.image("https://images.unsplash.com/photo-1551288049-bbbda536ad3a?auto=format&fit=crop&w=1350&q=80")

st.markdown("---")
st.caption("Desenvolvido por Reinaldo Fernandes (RM371717) - Tech Challenge FIAP Fase 1")
st.caption("Desenvolvido por Leonardo Junior Gonzales Mendoza RM 373713 - Tech Challenge FIAP Fase 1")
st.caption("Winny Tavares RM 371471 - Tech Challenge FIAP Fase 1")