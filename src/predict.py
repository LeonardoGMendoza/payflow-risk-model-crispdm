"""
Script de predição para modelo de risco PayFlow
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class RiskPredictor:
    """
    Preditor de risco de crédito com threshold de decisão
    """
    def __init__(self, model_path='models/modelo_risco_payflow.pkl'):
        if Path(model_path).exists():
            self.model = joblib.load(model_path)
            print("✅ Modelo carregado com sucesso!")
        else:
            self.model = None
            print("⚠️ Modelo não encontrado. Treine o modelo primeiro.")
        
        self.thresholds = {
            'auto_approve': 0.2,
            'manual_review': 0.4,
        }
    
    def predict(self, input_data):
        if self.model is None:
            return {"erro": "Modelo não carregado"}
        
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data
        
        prob = self.model.predict_proba(input_df)[:, 1][0]
        
        if prob < self.thresholds['auto_approve']:
            decisao = "✅ APROVAR automaticamente"
        elif prob < self.thresholds['manual_review']:
            decisao = "⚠️ ANÁLISE MANUAL necessária"
        else:
            decisao = "❌ NEGAR crédito"
        
        return {
            'probabilidade_inadimplencia': f'{prob:.2%}',
            'score_risco': float(prob),
            'decisao_negocio': decisao
        }

if __name__ == "__main__":
    predictor = RiskPredictor()
    cliente = {
        'score_credito': 650,
        'utilizacao_credito': 0.45,
        'dias_atraso_max_12m': 15,
        'comprometimento_renda': 0.30
    }
    resultado = predictor.predict(cliente)
    print("\n🔍 ANÁLISE DE CRÉDITO")
    print("=" * 40)
    for chave, valor in resultado.items():
        print(f"{chave}: {valor}")
