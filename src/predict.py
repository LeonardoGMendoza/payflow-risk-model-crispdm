import joblib
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def run_prediction(input_data):
    """
    Pipeline de inferência: Carrega modelo, alinha features e aplica lógica de negócio.
    """
    # 1. Localização dinâmica do modelo
    current_dir = Path(os.getcwd()).absolute()
    root_path = current_dir if (current_dir / "models").exists() else current_dir.parent
    model_path = root_path / "models" / "modelo_risco_payflow.pkl"

    if not model_path.exists():
        return {"erro": f"Modelo não encontrado em {model_path}"}

    try:
        # 2. Carregar o modelo (XGBoost/RF)
        model = joblib.load(model_path)
        
        # 3. Identificar colunas esperadas pelo modelo
        # Se for XGBoost, usamos feature_names_in_, se for RF, usamos n_features_in_ ou similar
        try:
            cols_treino = model.feature_names_in_
        except AttributeError:
            # Fallback para versões específicas ou outros modelos de árvore
            cols_treino = model.get_booster().feature_names if hasattr(model, 'get_booster') else []

        # 4. Preparar o Input (Garantir DataFrame)
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        else:
            df_input = input_data.copy()

        # 5. FEATURE ENGINEERING (O "Pulo do Gato" do Professor)
        if 'comprometimento_renda' not in df_input.columns:
            # Evitar divisão por zero
            df_input['comprometimento_renda'] = df_input['valor_solicitado'] / (df_input['renda_mensal'] + 1)

        # 6. ALINHAMENTO DE FEATURES (Resolve o erro de nomes de colunas)
        # Criamos um DataFrame vazio com as colunas do treino (preenchido com 0)
        df_final = pd.DataFrame(0, index=[0], columns=cols_treino)
        
        # Preenchemos com os dados que o cliente enviou
        for col in df_input.columns:
            if col in df_final.columns:
                df_final[col] = df_input[col].values

        # 7. PREDIÇÃO DE PROBABILIDADE
        prob = model.predict_proba(df_final)[:, 1][0]

        # 8. LÓGICA DE DECISÃO (Estratégia de Negócio)
        # Thresholds baseados na análise de Recall vs Precision
        if prob < 0.20:
            status = "APROVADO"
            recomendacao = "Risco baixo. Concessão automática sugerida."
        elif prob < 0.50:
            status = "ANALISE_MANUAL"
            recomendacao = "Risco moderado. Revisar garantias ou documentos extras."
        else:
            status = "NEGADO"
            recomendacao = "Risco alto. Probabilidade de inadimplência acima do limite."

        return {
            "score_risco_default": f"{prob:.2%}",
            "decisao_politica": status,
            "recomendacao_operacional": recomendacao,
            "info_adicional": {
                "comprometimento_calculado": f"{df_input['comprometimento_renda'].iloc[0]:.2%}",
                "modelo_versao": "XGBoost_SOTA_v1"
            }
        }

    except Exception as e:
        return {"erro": f"Falha técnica na inferência: {str(e)}"}

# --- BLOCO DE TESTE ---
if __name__ == "__main__":
    print("\n🚀 Testando Motor de Decisão PayFlow AI...")
    
    # Exemplo de cliente (Mesmo faltando colunas, o código agora é resiliente)
    cliente_exemplo = {
        "score_credito": 450,
        "valor_solicitado": 2500,
        "renda_mensal": 3000,
        "dias_atraso_max_12m": 45,
        "utilizacao_credito": 0.85,
        "idade": 30,
        "tempo_emprego_anos": 2
        # As colunas faltantes (como 'autonomo') serão preenchidas com 0 pelo alinhamento
    }
    
    resultado = run_prediction(cliente_exemplo)
    
    print("-" * 50)
    print(json.dumps(resultado, indent=4, ensure_ascii=False))
    print("-" * 50)