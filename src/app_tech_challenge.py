import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import os

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Tech Challenge: Monitor de Risco NPS",
    page_icon="📊",
    layout="wide"
)

# Custom CSS premium
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0e1117; }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e6e9ef;
    }
    [data-testid="stMetricLabel"] p {
        color: #555e6d !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    [data-testid="stMetricValue"] div {
        color: #1f77b4 !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a2e;
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #16213e;
        border-radius: 8px;
        color: #aaa;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #1f77b4;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# CARREGAMENTO DO MODELO E DADOS (MLOps Robustness)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    try:
        current_dir = Path(__file__).resolve().parent
        root_path = current_dir.parent if current_dir.name == 'src' else current_dir
        model_path    = root_path / "models" / "modelo_nps_rf.pkl"
        features_path = root_path / "models" / "features_nps.pkl"
        if not model_path.exists():
            return None, None
        model    = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except:
        return None, None

@st.cache_data
def load_dataset():
    try:
        current_dir = Path(__file__).resolve().parent
        root_path = current_dir.parent if current_dir.name == 'src' else current_dir
        csv_path = root_path / "Base de dados Tech Challenge" / "desafio_nps_fase_1.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        df['nps_categoria'] = df['nps_score'].apply(
            lambda s: 'Promotor' if s >= 9 else ('Neutro' if s >= 7 else 'Detrator')
        )
        df['is_detrator'] = (df['nps_score'] < 7).astype(int)
        return df
    except:
        return None

model_nps, feature_list = load_ml_assets()
df_data = load_dataset()

def calculate_risk(data, model, features):
    df_input = pd.DataFrame([data])
    df_input['delay_ratio'] = df_input['delivery_delay_days'] / (df_input['delivery_time_days'] + 1)
    df_final = pd.DataFrame(0, index=[0], columns=features)
    for col in df_input.columns:
        if col in df_final.columns:
            df_final[col] = df_input[col].values
    prob = model.predict_proba(df_final)[:, 1][0]
    return prob

# Paleta de cores
COR_VERDE   = '#2ecc71'
COR_AMARELO = '#f39c12'
COR_VERMELHO= '#e74c3c'
COR_AZUL    = '#3498db'
COR_ROXO    = '#9b59b6'

# ------------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("PayFlow AI Engine")
    st.markdown("---")

    st.subheader("📋 Dados do Pedido")
    age        = st.slider("Idade do Cliente", 18, 90, 35)
    tenure     = st.number_input("Meses de Relacionamento", 0, 240, 12)
    order_val  = st.number_input("Valor do Pedido (R$)", 0.0, 5000.0, 250.0)
    items      = st.slider("Qtd de Itens", 1, 20, 15)

    st.subheader("🚚 Logística")
    delivery_days = st.number_input("Prazo de Entrega (Dias)", 1, 30, 3)
    delay_days    = st.number_input("Dias de Atraso", 0, 30, 3)

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
# PAINEL PRINCIPAL — ABAS
# ------------------------------------------------------------------------------
st.title("📊 Tech Challenge: Monitor de Risco de Cliente (NPS)")
st.markdown("Diagnóstico preditivo baseado em IA para antecipar insatisfação do cliente.")

tab1, tab2, tab3 = st.tabs(["🚀 Predição em Tempo Real", "🔬 Análise Exploratória (EDA)", "📋 Sobre o Projeto"])

# ==============================================================================
# ABA 1 — PREDIÇÃO
# ==============================================================================
with tab1:
    if not model_nps:
        st.warning("⚠️ Modelo não carregado. Verifique a pasta 'models'.")
        st.stop()

    if st.session_state.run_analysis:
        input_dict = {
            'customer_age': age, 'customer_tenure_months': tenure, 'order_value': order_val,
            'items_quantity': items, 'delivery_time_days': delivery_days,
            'delivery_delay_days': delay_days, 'customer_service_contacts': contacts,
            f'customer_region_{region}': 1
        }

        risk_prob    = calculate_risk(input_dict, model_nps, feature_list)
        risk_percent = risk_prob * 100

        # KPIs
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Probabilidade de Detração", f"{risk_prob:.1%}")
        with col2:
            status = "CRÍTICO" if risk_prob > 0.35 else "SEGURO"
            color  = COR_VERMELHO if status == "CRÍTICO" else COR_AZUL
            st.markdown(
                f"**Status da Experiência:** <h2 style='color:{color}; margin-top:-15px;'>{status}</h2>",
                unsafe_allow_html=True
            )
        with col3:
            ruptura = "Sim ⚠️" if delay_days >= 3 else "Não ✅"
            st.metric("Ponto de Ruptura Atingido?", ruptura)

        st.markdown("---")
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.subheader("🌡️ Termômetro de Risco")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#1f77b4", 'thickness': 0.6},
                    'bgcolor': "rgba(0,0,0,0.1)",
                    'borderwidth': 2, 'bordercolor': "#444",
                    'steps': [
                        {'range': [0,  35], 'color': 'rgba(0, 255, 0, 0.15)'},
                        {'range': [35, 70], 'color': 'rgba(255, 255, 0, 0.15)'},
                        {'range': [70,100], 'color': 'rgba(255, 0, 0, 0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 6},
                        'thickness': 0.8,
                        'value': risk_percent
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Inter"},
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
            probs  = [calculate_risk({**input_dict, 'delivery_delay_days': d}, model_nps, feature_list) for d in delays]
            fig_line = px.line(x=delays, y=probs,
                               labels={'x': 'Dias de Atraso', 'y': 'Probabilidade'},
                               template="plotly_dark")
            fig_line.add_hline(y=0.35, line_dash="dash", line_color="red",
                               annotation_text="Limite de Ruptura (35%)")
            fig_line.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("👈 Ajuste as variáveis operacionais na barra lateral e clique em **'Analisar Risco'** para iniciar.")
        st.image("https://images.unsplash.com/photo-1551288049-bbbda536ad3a?auto=format&fit=crop&w=1350&q=80")

# ==============================================================================
# ABA 2 — EDA (CONTRIBUIÇÃO DO LEONARDO)
# ==============================================================================
with tab2:
    st.header("🔬 Análise Exploratória de Dados (EDA)")
    st.markdown(
        "**Contribuição:** Leonardo Junior Gonzales Mendoza — RM 373713  \n"
        "Análise baseada nos dados reais do desafio, seguindo a metodologia **CRISP-DM**."
    )

    if df_data is None:
        st.error("❌ Base de dados não encontrada. Verifique a pasta 'Base de dados Tech Challenge'.")
    else:
        # KPIs do Dataset
        taxa_det  = df_data['is_detrator'].mean()
        atraso_med = df_data['delivery_delay_days'].mean()
        nps_med    = df_data['nps_score'].mean()
        total      = len(df_data)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📦 Total de Registros", f"{total:,}")
        k2.metric("📉 Taxa de Detração", f"{taxa_det:.1%}")
        k3.metric("⏱️ Atraso Médio", f"{atraso_med:.1f} dias")
        k4.metric("⭐ NPS Médio Geral", f"{nps_med:.1f}")

        st.markdown("---")

        # --- Gráfico 1: Distribuição NPS ---
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🎯 Composição do NPS")
            contagem = df_data['nps_categoria'].value_counts()
            fig_pizza = go.Figure(data=[go.Pie(
                labels=contagem.index,
                values=contagem.values,
                hole=0.45,
                marker=dict(
                    colors=[COR_VERMELHO if c == 'Detrator' else COR_AMARELO if c == 'Neutro' else COR_VERDE
                             for c in contagem.index],
                    line=dict(color='#0e1117', width=3)
                ),
                textinfo='label+percent',
                textfont_size=13
            )])
            fig_pizza.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=360,
                annotations=[dict(
                    text=f"{taxa_det:.0%}<br><b>Detratores</b>",
                    x=0.5, y=0.5,
                    font=dict(size=15, color=COR_VERMELHO),
                    showarrow=False
                )],
                legend=dict(orientation='h', y=-0.1)
            )
            st.plotly_chart(fig_pizza, use_container_width=True)

        with col_b:
            st.subheader("📊 Distribuição das Notas NPS")
            fig_hist = px.histogram(
                df_data, x='nps_score', nbins=11,
                color='nps_categoria',
                color_discrete_map={
                    'Promotor': COR_VERDE,
                    'Neutro':   COR_AMARELO,
                    'Detrator': COR_VERMELHO
                },
                template='plotly_dark',
                labels={'nps_score': 'Score NPS', 'count': 'Quantidade'}
            )
            fig_hist.add_vline(x=6.5, line_dash='dash', line_color='white',
                               annotation_text='Linha Detrator', annotation_font_color='white')
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=360, legend_title='Categoria'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # --- Gráfico 2: Boxplot + Curva de Ruptura ---
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("📦 Atraso na Entrega por Categoria NPS")
            fig_box = px.box(
                df_data,
                x='nps_categoria',
                y='delivery_delay_days',
                color='nps_categoria',
                color_discrete_map={
                    'Promotor': COR_VERDE,
                    'Neutro':   COR_AMARELO,
                    'Detrator': COR_VERMELHO
                },
                category_orders={'nps_categoria': ['Detrator', 'Neutro', 'Promotor']},
                template='plotly_dark',
                points='outliers',
                labels={'delivery_delay_days': 'Dias de Atraso', 'nps_categoria': ''}
            )
            fig_box.add_hline(y=3, line_dash='dash', line_color='orange', line_width=2,
                              annotation_text='⚠️ Ponto de Ruptura (3 dias)',
                              annotation_font_color='orange')
            fig_box.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=380, showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with col_d:
            st.subheader("📈 Curva de Risco × Dias de Atraso")
            curva = df_data.groupby('delivery_delay_days')['is_detrator'].mean().reset_index()
            curva.columns = ['delay_days', 'taxa_detracao']
            curva = curva[curva['delay_days'] <= 14]
            curva['taxa_pct'] = curva['taxa_detracao'] * 100

            fig_curva = go.Figure()
            fig_curva.add_trace(go.Scatter(
                x=curva['delay_days'], y=curva['taxa_pct'],
                mode='lines+markers',
                line=dict(color=COR_VERMELHO, width=3),
                marker=dict(size=8, color=COR_VERMELHO),
                fill='tozeroy',
                fillcolor='rgba(231,76,60,0.15)',
                name='Taxa de Detração'
            ))
            fig_curva.add_vline(x=3, line_dash='dash', line_color='orange', line_width=2)
            fig_curva.add_annotation(
                x=3.3, y=curva['taxa_pct'].max() * 0.8,
                text="⚠️ Ruptura<br>(3 dias)",
                font=dict(color='orange', size=12),
                showarrow=False
            )
            fig_curva.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                template='plotly_dark',
                xaxis_title='Dias de Atraso',
                yaxis_title='Taxa de Detração (%)',
                yaxis_ticksuffix='%',
                height=380, showlegend=False
            )
            st.plotly_chart(fig_curva, use_container_width=True)

        st.markdown("---")

        # --- Gráfico 3: Heatmap de Correlação ---
        st.subheader("🔥 Matriz de Correlação — Variáveis vs NPS")

        cols_corr = ['nps_score', 'delivery_delay_days', 'delivery_time_days',
                     'customer_service_contacts', 'complaints_count', 'resolution_time_days',
                     'order_value', 'items_quantity', 'csat_internal_score',
                     'customer_tenure_months', 'delivery_attempts']
        cols_disp = [c for c in cols_corr if c in df_data.columns]
        corr = df_data[cols_disp].corr().round(2)

        fig_heat = px.imshow(
            corr, text_auto=True, aspect='auto',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            template='plotly_dark',
            title=''
        )
        fig_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            height=500,
            font=dict(size=11)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Insight de correlação
        corr_nps = corr['nps_score'].drop('nps_score').sort_values()
        col_ins1, col_ins2 = st.columns(2)
        with col_ins1:
            st.markdown("**🔴 Maior impacto NEGATIVO no NPS:**")
            for var, val in corr_nps.head(3).items():
                st.markdown(f"- `{var}`: **{val:.3f}**")
        with col_ins2:
            st.markdown("**🟢 Maior impacto POSITIVO no NPS:**")
            for var, val in corr_nps.tail(3).items():
                st.markdown(f"- `{var}`: **{val:.3f}**")

        st.markdown("---")

        # --- Gráfico 4: Análise Regional ---
        st.subheader("🌍 Análise Regional de Risco")

        regional = df_data.groupby('customer_region').agg(
            taxa_detracao=('is_detrator', 'mean'),
            atraso_medio=('delivery_delay_days', 'mean'),
            nps_medio=('nps_score', 'mean'),
            volume=('customer_id', 'count')
        ).reset_index().round(2)
        regional['taxa_pct'] = (regional['taxa_detracao'] * 100).round(1)

        fig_reg = make_subplots(rows=1, cols=2,
                                subplot_titles=['Taxa de Detração por Região (%)',
                                                'NPS Médio por Região'])

        cores = [COR_VERMELHO if v == regional['taxa_pct'].max() else COR_AZUL
                 for v in regional['taxa_pct']]

        fig_reg.add_trace(go.Bar(
            x=regional['customer_region'], y=regional['taxa_pct'],
            marker_color=cores, text=regional['taxa_pct'],
            texttemplate='%{text}%', textposition='outside', name='Detração'
        ), row=1, col=1)

        fig_reg.add_trace(go.Bar(
            x=regional['customer_region'], y=regional['nps_medio'],
            marker_color=COR_VERDE, text=regional['nps_medio'],
            texttemplate='%{text:.1f}', textposition='outside', name='NPS'
        ), row=1, col=2)

        fig_reg.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            template='plotly_dark', height=380, showlegend=False
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        st.markdown("---")

        # --- Teste de Hipótese ---
        st.subheader("🧪 Teste de Hipótese — Mann-Whitney U")
        st.markdown(
            "> **H₀:** Não há diferença no NPS entre pedidos no prazo e atrasados.  \n"
            "> **H₁:** Atrasos causam queda significativa no NPS (p < 0.05)."
        )

        nps_prazo     = df_data[df_data['delivery_delay_days'] == 0]['nps_score']
        nps_atrasados = df_data[df_data['delivery_delay_days'] >  0]['nps_score']
        stat, p_val   = stats.mannwhitneyu(nps_prazo, nps_atrasados, alternative='greater')

        r1, r2, r3 = st.columns(3)
        r1.metric("NPS Médio — No Prazo",  f"{nps_prazo.mean():.2f}")
        r2.metric("NPS Médio — Atrasados", f"{nps_atrasados.mean():.2f}")
        r3.metric("p-valor", f"{p_val:.2e}")

        if p_val < 0.05:
            st.success(
                "✅ **Resultado:** Rejeitamos H₀! A diferença é estatisticamente significativa. "
                "O atraso causa uma queda real e comprovada no NPS. Nossa IA está embasada pela estatística! 🚀"
            )
        else:
            st.warning("⚠️ Não rejeitamos H₀ com os dados atuais.")

# ==============================================================================
# ABA 3 — SOBRE O PROJETO
# ==============================================================================
with tab3:
    st.header("📋 Sobre o Projeto")

    st.markdown("""
    ## 🚀 Tech Challenge Fase 1 — FIAP AI Scientist

    Este projeto aplica **Inteligência Artificial** para prever a insatisfação de clientes
    **antes** da pesquisa de NPS ser aplicada, permitindo ações proativas de retenção.

    ---

    ### 🔬 Metodologia CRISP-DM Aplicada

    | Fase | Responsável | Entregável |
    |------|-------------|------------|
    | 1. Entendimento do Negócio | Grupo | Definição do objetivo de NPS |
    | 2. Entendimento dos Dados | **Leonardo** | Notebook EDA completo |
    | 3. Preparação dos Dados | Reinaldo | Feature Engineering |
    | 4. Modelagem | Reinaldo | Random Forest + SMOTE |
    | 5. Avaliação | Grupo | AUC-ROC = 0.92 |
    | 6. Deploy | Reinaldo | Dashboard Streamlit |
    | 7. QA / Testes | Reinaldo | pytest — 3 testes unitários |

    ---

    ### 🤖 Arquitetura da Solução

    ```
    Base de dados → EDA (Notebook) → Feature Engineering → Random Forest
                                                              ↓
                                           Dashboard Streamlit (Produção)
                                                              ↓
                                              Monitor de Risco em Tempo Real
    ```

    ---

    ### 👥 Integrantes do Grupo
    """)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        **🛠️ Reinaldo Fernandes**
        RM 371717
        - Modelagem de IA
        - Dashboard Streamlit
        - Testes unitários (pytest)
        """)
    with m2:
        st.markdown("""
        **📊 Leonardo Jr. G. Mendoza**
        RM 373713
        - Análise Exploratória (EDA)
        - Aba EDA no Dashboard
        - Storytelling com Dados
        """)
    with m3:
        st.markdown("""
        **📋 Winny Tavares**
        RM 371471
        - Documentação
        - Validação de resultados
        """)

st.markdown("---")
st.caption("Desenvolvido por Reinaldo Fernandes (RM371717) | Leonardo Junior Gonzales Mendoza (RM373713) | Winny Tavares (RM371471)")
st.caption("Tech Challenge FIAP Fase 1 — AI Scientist")