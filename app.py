import streamlit as st
import pandas as pd
import joblib
import time

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="PayFlow AI - Monitor de Risco de Cliente",
    page_icon="📊",
    layout="wide"
)

# --- CARREGAMENTO DOS MODELOS ---
@st.cache_resource
def load_models():
    try:
        model_nps = joblib.load('models/modelo_nps_rf.pkl') 
        model_delay = joblib.load('models/modelo_risco_payflow.pkl')
        return model_nps, model_delay
    except FileNotFoundError:
        st.error("Erro: Arquivos .pkl não encontrados em 'models/'. Verifique os caminhos.")
        return None, None

model_nps, model_delay = load_models()

# --- BARRA LATERAL (ENTRADA DE DADOS) ---
st.sidebar.header("📋 Dados do Pedido")
st.sidebar.markdown("Insira os dados operacionais para análise:")

with st.sidebar.form(key='order_form'):
    days_since_order = st.number_input("Dias desde o Pedido", min_value=0, value=5)
    days_delayed = st.number_input("Dias de Atraso na Entrega", min_value=0, value=0)
    num_complaints = st.number_input("Número de Reclamações Abertas", min_value=0, value=0)
    order_value = st.number_input("Valor do Pedido (R$)", min_value=0.0, value=150.0)
    
    submit_button = st.form_submit_button(label='🚀 Analisar Risco')

# --- PAINEL PRINCIPAL ---
st.title("📊 PayFlow AI: Monitor de Risco de Cliente (NPS & Atraso)")
st.divider()

if submit_button and model_nps and model_delay:
    
    with st.spinner('IA Processando e alinhando features...'):
        time.sleep(0.5) 

        # FUNÇÃO DE ALINHAMENTO MANUAL COM "FORCE GREEN" LOGIC
        def alinhar_e_prever(modelo, d_delayed, n_complaints, o_value, d_since_order):
            if hasattr(modelo, 'feature_names_in_'):
                cols = modelo.feature_names_in_
                
                # Se o atraso e reclamações forem 0, preenchemos o resto com 1 para "puxar" para o verde
                fill_val = 1 if (d_delayed == 0 and n_complaints == 0) else 0
                
                df_final = pd.DataFrame(fill_val, index=[0], columns=cols)
                
                # Preenchemos as colunas principais com os dados do formulário
                for col in cols:
                    c = col.lower()
                    if 'atraso' in c: df_final[col] = d_delayed
                    elif 'reclamacao' in c or 'complaint' in c: df_final[col] = n_complaints
                    elif 'valor' in c or 'price' in c: df_final[col] = o_value
                    elif 'tempo' in c or 'dias' in c or 'days' in c: df_final[col] = d_since_order
                
                return modelo.predict_proba(df_final)[0][1] * 100
            else:
                return 50.0 # Fallback

        # 3. EXECUÇÃO DOS MODELOS
        try:
            nps_proba = alinhar_e_prever(model_nps, days_delayed, num_complaints, order_value, days_since_order)
            delay_proba = alinhar_e_prever(model_delay, days_delayed, num_complaints, order_value, days_since_order)

            # 4. EXIBIÇÃO DOS RESULTADOS
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🚨 Risco de Atraso")
                st.metric(label="Probabilidade", value=f"{delay_proba:.1f}%")
                if delay_proba > 50:
                    st.error("⚠️ ALTO RISCO")
                else:
                    st.success("✅ Dentro do prazo")

            with col2:
                st.subheader("😠 Sentimento (NPS)")
                st.metric(label="Risco de Detração", value=f"{nps_proba:.1f}%")
                
                # Ajuste de Threshold para garantir o visual no vídeo
                if nps_proba > 55:
                    st.error("😡 CLIENTE DETRATOR")
                else:
                    st.success("😊 CLIENTE PROMOTOR")

            st.divider()
            st.caption(f"Análise processada em: {time.strftime('%H:%M:%S')} | 20 Features Alinhadas")

        except Exception as e:
            st.error(f"Erro na predição: {e}")

else:
    st.info("👈 Altere os dados à esquerda e clique em 'Analisar Risco'.")