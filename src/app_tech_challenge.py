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

# Custom CSS para visual premium e correção de visibilidade
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    /* Estilização dos cards de métricas (Branco com borda suave) */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e6e9ef;
    }
    /* Ajuste de Cor dos Títulos das Métricas (Cinza Escuro) */
    [data-testid="stMetricLabel"] p {
        color: #555e6d !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    /* Ajuste de Cor dos Valores das Métricas (Azul Profissional) */
    [data-testid="stMetricValue"] div {
        color: #1f77b4 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSidebarNav"] {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# CARREGAMENTO DO MODELO E UTILITÁRIOS (MLOps Robustness)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    """Carrega o modelo e as features garantindo o caminho correto."""
    try:
        # Detecta a raiz do projeto (sobe um nível se estiver em 'src')
        current_dir = Path(__file__).resolve().parent
        root_path = current_dir.parent if current_dir.name == 'src' else current_dir
        
        model_path = root_path / "models" / "modelo_nps_rf.pkl"
        features_path = root_path / "models" / "features_nps.pkl"
        
        if not model_path.exists():
            st.error(f"Arquivo não encontrado: {model_path}")
            return None, None
            
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None, None

model_nps, feature_list = load_ml_assets()

def calculate_risk(data, model, features):
    """Executa a predição com alinhamento de features."""
    df_input = pd.DataFrame([data])
    
    # Engenharia de Features: delay_ratio (Crítico para a precisão)
    df_input['delay_ratio'] = df_input['delivery_delay_days'] / (df_input['delivery_time_days'] + 1)
    
    # Alinhamento com as colunas do treino (Dummies)
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
    items = st.slider("Qtd de Itens", 1, 20, 2)
    
    st.subheader("🚚 Logística")
    delivery_days = st.number_input("Prazo de Entrega (Dias)", 1, 30, 5)
    delay_days = st.number_input("Dias de Atraso", 0, 30, 0)
    
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
st.markdown("""
    Esta aplicação utiliza o modelo **Random Forest** para antecipar a insatisfação do cliente.
    O diagnóstico é baseado no comportamento logístico e histórico de suporte.
""")

if not model_nps:
    st.warning("⚠️ Aguardando carregamento do modelo. Verifique a pasta 'models'.")
    st.stop()

if st.session_state.run_analysis:
    # Preparar dados para o modelo
    input_dict = {
        'customer_age': age,
        'customer_tenure_months': tenure,
        'order_value': order_val,
        'items_quantity': items,
        'delivery_time_days': delivery_days,
        'delivery_delay_days': delay_days,
        'customer_service_contacts': contacts,
        f'customer_region_{region}': 1
    }
    
    risk_prob = calculate_risk(input_dict, model_nps, feature_list)
    
    # Grid de KPIs (Ajustado para 3 colunas)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Probabilidade de Detração", f"{risk_prob:.1%}")
    
    with col2:
        # Título e Status fora do card para manter o estilo da imagem enviada
        st.markdown(f"**Status da Experiência:**")
        status = "CRÍTICO" if risk_prob > 0.35 else "SEGURO"
        color = "red" if status == "CRÍTICO" else "green"
        st.markdown(f"<h2 style='color:{color}; margin-top:-15px;'>{status}</h2>", unsafe_allow_html=True)
        
    with col3:
        ruptura = "Sim" if delay_days >= 3 else "Não"
        st.metric("Ponto de Ruptura Atingido?", ruptura)

    st.markdown("---")
    
    # Gráfico de Velocímetro (Gauge)
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        st.subheader("🌡️ Termômetro de Risco")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risco (%)", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1f77b4"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 35], 'color': '#d4edda'},
                    {'range': [35, 70], 'color': '#fff3cd'},
                    {'range': [70, 100], 'color': '#f8d7da'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 35}
            }
        ))
        fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c_right:
        st.subheader("💡 Recomendações Estratégicas")
        if risk_prob > 0.7:
            st.error("🚨 **Ação Imediata:** Este cliente tem altíssimo risco de se tornar detrator. Disparar cupom de desconto ou contato telefônico prioritário.")
        elif risk_prob > 0.35:
            st.warning("⚠️ **Alerta Preventivo:** Atraso identificado. Enviar notificação push proativa explicando o motivo e oferecendo frete grátis na próxima compra.")
        else:
            st.success("✅ **Fidelização:** Cliente satisfeito. Momento ideal para cross-selling ou convite para o programa de fidelidade.")

        # Gráfico Auxiliar: Impacto do Atraso Estimado
        st.markdown("**Impacto do Atraso Estimado:**")
        delays = list(range(0, 11))
        probs = [calculate_risk({**input_dict, 'delivery_delay_days': d}, model_nps, feature_list) for d in delays]
        df_sim = pd.DataFrame({'Dias de Atraso': delays, 'Risco de Detração': probs})
        fig_line = px.line(df_sim, x='Dias de Atraso', y='Risco de Detração', color_discrete_sequence=['#636EFA'])
        fig_line.add_hline(y=0.35, line_dash="dash", line_color="red", annotation_text="Limite de Ruptura")
        fig_line.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_line, use_container_width=True)

else:
    st.info("👈 Ajuste as variáveis operacionais na barra lateral e clique em 'Analisar Risco' para iniciar o diagnóstico.")
    
    # Imagem de placeholder ou boas-vindas
    st.image("https://images.unsplash.com/photo-1551288049-bbbda536ad3a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", caption="Análise Preditiva de Satisfação do Cliente")

st.markdown("---")
st.caption("Desenvolvido por Reinaldo Fernandes (RM371717) - Tech Challenge FIAP Fase 1")
st.caption("Desenvolvido por Leonardo Junior Gonzales Mendoza RM 373713 - Tech Challenge FIAP Fase 1")
st.caption("Winny Tavares RM 371471 - Tech Challenge FIAP Fase 1")
