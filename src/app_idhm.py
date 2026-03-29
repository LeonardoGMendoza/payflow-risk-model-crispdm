import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
from pathlib import Path

# --- Rigor de Engenharia: Localização de Módulos ---
# Como este arquivo está dentro de 'src/', o 'idhm.py' está no mesmo diretório.
# Adicionamos o diretório atual ao sys.path para garantir que os imports funcionem.
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from idhm import find_idh_file, load_and_clean_idh, run_data_audit

# --- Configuração da Página ---
st.set_page_config(page_title="PayFlow | IDHM Intelligence", layout="wide", page_icon="🌍")

# Estilo para melhorar a estética (UI/UX)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🌍 Inteligência Territorial: Análise de Oportunidades IDHM")
st.markdown("""
Esta aplicação identifica **Territórios Premium** para expansão de crédito, 
utilizando o motor de processamento e auditoria validado via MLOps.
""")

# --- Carga de Dados (Usando o motor idhm.py) ---
@st.cache_data
def get_data():
    # O motor find_idh_file já possui lógica de busca dinâmica
    path = find_idh_file()
    if path:
        df = load_and_clean_idh(path)
        # Garantir tipagem numérica para os indicadores
        for col in ['idhm', 'idhm_educacao', 'idhm_longevidade', 'idhm_renda']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    return None

df = get_data()

if df is not None:
    # --- Sidebar: Filtros de Negócio ---
    st.sidebar.header("🎯 Filtros Estratégicos")
    ufs = sorted(df['nome_da_unidade_da_federacao'].unique())
    selected_uf = st.sidebar.multiselect("Selecione os Estados:", ufs, default=ufs[:5])
    
    idh_min = st.sidebar.slider("Piso de IDH Municipal:", 
                                float(df['idhm'].min()), 
                                float(df['idhm'].max()), 
                                0.65)

    # --- Filtragem ---
    df_filtered = df[(df['nome_da_unidade_da_federacao'].isin(selected_uf)) & 
                     (df['idhm'] >= idh_min)]

    # --- Métricas de Resumo ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Municípios Filtrados", len(df_filtered))
    c2.metric("IDHM Médio", round(df_filtered['idhm'].mean(), 3))
    c3.metric("Maior IDH", round(df_filtered['idhm'].max(), 3))
    c4.metric("Desvio Padrão", round(df_filtered['idhm'].std(), 3))

    st.divider()

    # --- Linha 1 de Gráficos ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top 15 Municípios por IDHM")
        fig_bar = px.bar(
            df_filtered.sort_values(by='idhm', ascending=False).head(15), 
            x='idhm', y='municipio', color='idhm',
            orientation='h',
            color_continuous_scale='Viridis',
            labels={'idhm': 'IDH Municipal', 'municipio': 'Cidade'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("📈 Correlação: Renda vs Educação")
        fig_scatter = px.scatter(
            df_filtered, x='idhm_educacao', y='idhm_renda', 
            size='idhm', color='nome_da_unidade_da_federacao',
            hover_name='municipio',
            title="Sinal de Equilíbrio Social",
            labels={'idhm_educacao': 'IDH Educação', 'idhm_renda': 'IDH Renda'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Linha 2 de Gráficos ---
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📊 Distribuição de IDH por Estado")
        fig_box = px.box(
            df_filtered, x='nome_da_unidade_da_federacao', y='idhm',
            color='nome_da_unidade_da_federacao',
            points="all",
            title="Variabilidade Regional"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col4:
        st.subheader("🔍 Densidade de Indicadores")
        fig_hist = px.histogram(
            df_filtered, x='idhm', nbins=30, 
            color_discrete_sequence=['#636EFA'],
            marginal="rug", title="Frequência de IDHM na Amostra"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- Data Audit & Tabela ---
    st.divider()
    with st.expander("🛠️ Auditoria de Integridade e Dados Brutos"):
        try:
            # Auditando os dados filtrados em tempo real
            if run_data_audit(df_filtered):
                st.success("✅ Pipeline validado: Dados íntegros para decisão financeira.")
        except Exception as e:
            st.warning(f"⚠️ Alerta de Auditoria: {e}")
        
        st.dataframe(df_filtered, use_container_width=True)

else:
    st.error("Erro Crítico: O arquivo 'IDH_2010.xls' não foi localizado. Verifique a pasta 'Base de dados IDH'.")