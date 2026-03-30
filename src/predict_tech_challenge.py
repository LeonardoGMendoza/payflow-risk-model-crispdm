import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import os

def predict_customer_satisfaction(input_data):
    """
    Motor de Predição de NPS (Tech Challenge - FIAP)
    Recebe um dicionário com dados operacionais e retorna o risco de detração.
    """
    # --- AJUSTE DE CAMINHO INTELIGENTE (MLOps) ---
    # Detectamos onde este script (.py) está localizado
    current_dir = Path(__file__).resolve().parent
    
    # Se o script estiver dentro de 'src', subimos um nível para achar a raiz.
    # Se estiver na raiz, ficamos nela.
    root_path = current_dir.parent if current_dir.name == 'src' else current_dir
    model_path = root_path / "models" / "modelo_nps_rf.pkl"
    
    if not model_path.exists():
        return {
            "erro": f"Modelo não encontrado em {model_path.absolute()}",
            "dica": "Certifique-se de que a pasta 'models' existe na raiz e contém o arquivo .pkl"
        }

    try:
        # 2. Carga do Modelo treinado no Notebook
        model = joblib.load(model_path)
        cols_treino = model.feature_names_in_
        
        # 3. Transformação do input em DataFrame
        df_input = pd.DataFrame([input_data])

        # --- ENGENHARIA DE FEATURES (Sincronizada com o Notebook do Canvas) ---
        # Criamos o 'delay_ratio' pois o modelo foi treinado com essa inteligência
        if 'delivery_delay_days' in df_input.columns and 'delivery_time_days' in df_input.columns:
            df_input['delay_ratio'] = df_input['delivery_delay_days'] / (df_input['delivery_time_days'] + 1)
        
        # 4. Alinhamento de colunas (Garante que dummies ausentes sejam 0)
        df_final = pd.DataFrame(0, index=[0], columns=cols_treino)
        for col in df_input.columns:
            if col in df_final.columns:
                df_final[col] = df_input[col].values

        # 5. Execução da Predição (Probabilidade de ser Detrator)
        prob = model.predict_proba(df_final)[:, 1][0]
        
        # 6. Lógica de Decisão Baseada em Risco (Threshold 0.35 focado em Recall)
        return {
            "status": "Sucesso",
            "classe": "INSATISFEITO (Risco Alto)" if prob > 0.35 else "SATISFEITO (Baixo Risco)",
            "probabilidade_detracao": f"{prob:.2%}",
            "score_risco": round(float(prob * 100), 2),
            "alerta_logistico": "CRÍTICO" if input_data.get('delivery_delay_days', 0) > 3 else "NORMAL"
        }

    except Exception as e:
        return {"erro": f"Falha na inferência técnica: {str(e)}"}

# --- BLOCO DE TESTE (EXECUÇÃO VIA TERMINAL) ---
if __name__ == "__main__":
    # Simulação de um cliente com atraso logístico (Cenário de Risco)
    cliente_exemplo = {
        'customer_age': 45,
        'order_value': 250.0,
        'delivery_time_days': 5,
        'delivery_delay_days': 8,  # Atraso acima do ponto de ruptura (3 dias)
        'customer_service_contacts': 3,
        'customer_region_Sudeste': 1
    }
    
    resultado = predict_customer_satisfaction(cliente_exemplo)
    
    print("\n" + "="*40)
    print("🚀 PAYFLOW AI: DIAGNÓSTICO DE SATISFAÇÃO")
    print("="*40)
    
    if "erro" in resultado:
        print(f"❌ {resultado['erro']}")
    else:
        for chave, valor in resultado.items():
            print(f"🔹 {chave.replace('_', ' ').title()}: {valor}")
    print("="*40 + "\n")