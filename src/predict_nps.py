import joblib
import pandas as pd
from pathlib import Path
import os

def predict_customer_satisfaction(input_data):
    """
    Motor de Predição de NPS (Tech Challenge)
    Recebe um dicionário com dados logísticos e retorna o risco de detração.
    """
    # Localização do modelo (ajuste o nome do arquivo .pkl se necessário)
    model_path = Path("models/modelo_nps_rf.pkl") 
    
    if not model_path.exists():
        return {"erro": f"Modelo não encontrado em {model_path.absolute()}"}

    try:
        model = joblib.load(model_path)
        
        # Alinhamento de Features (Padrão Sênior)
        cols_treino = model.feature_names_in_
        df_input = pd.DataFrame([input_data])
        df_final = pd.DataFrame(0, index=[0], columns=cols_treino)

        for col in df_input.columns:
            if col in df_final.columns:
                df_final[col] = df_input[col].values

        # Predição
        prob = model.predict_proba(df_final)[:, 1][0]
        
        return {
            "classe": "DETRATOR" if prob > 0.5 else "PROMOTOR/NEUTRO",
            "probabilidade_detracao": f"{prob:.2%}",
            "alerta_logistico": "ALTO" if input_data.get('delivery_delay_days', 0) > 5 else "NORMAL",
            "status": "Sucesso"
        }
    except Exception as e:
        return {"erro": f"Falha na inferência: {str(e)}"}