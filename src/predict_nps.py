import joblib
import pandas as pd
import os

def fazer_predicao_nps(dados_cliente):
    # Caminhos para a pasta models na raiz
    path_mod = 'models/modelo_nps_rf.pkl'
    path_feat = 'models/features_nps.pkl'

    if not os.path.exists(path_mod):
        return "Erro: Modelo não encontrado. Verifique a pasta /models.", 0

    model = joblib.load(path_mod)
    features = joblib.load(path_feat)
    
    df = pd.DataFrame([dados_cliente])
    # Alinhando as colunas com o que o modelo espera
    df = pd.get_dummies(df).reindex(columns=features, fill_value=0)
    
    prob = model.predict_proba(df)[0][1]
    status = "DETRATOR 🚩" if prob > 0.5 else "PROMOTOR/NEUTRO ✅"
    
    return status, prob

if __name__ == "__main__":
    print("\n--- 🚀 EXECUTANDO PREDITOR DE NPS (TECH CHALLENGE) ---")
    
    # Exemplo de teste:
    exemplo = {
        'delivery_delay_days': 10, 
        'complaints_count': 1, 
        'repeat_purchase_30d': 0
    }
    
    res, score = fazer_predicao_nps(exemplo)
    print(f"Resultado Previsto: {res}")
    print(f"Risco de Insatisfação: {score:.2%}")