import joblib
import pandas as pd
import os
from pathlib import Path

def run_prediction(input_data):
    """
    Carrega o modelo da pasta 'models' e realiza a predição.
    """
    # 1. Localização dinâmica: O script está em 'src', o modelo em 'models'
    # Subimos um nível (..) para a raiz e entramos em 'models'
    current_dir = Path(os.getcwd()).absolute()
    
    # Se rodarmos da raiz, o caminho é 'models/...'
    # Se rodarmos de dentro da 'src', o caminho é '../models/...'
    root_path = current_dir if (current_dir / "models").exists() else current_dir.parent
    model_path = root_path / "models" / "modelo_risco_payflow.pkl"

    if not model_path.exists():
        return f"❌ Erro: Modelo não encontrado em {model_path}"

    try:
        # 2. Carregar o artefato
        model = joblib.load(model_path)
        
        # 3. Realizar a predição
        # Nota: O input_data deve ser um DataFrame formatado com as mesmas colunas do treino
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]
        
        return {
            "target": prediction[0],
            "probabilidade_inadimplencia": f"{probability[0]:.2%}",
            "status": "Risco Alto" if prediction[0] == 1 else "Risco Baixo"
        }
    except Exception as e:
        return f"❌ Erro na predição: {e}"

# Exemplo de uso para teste rápido
if __name__ == "__main__":
    print("🚀 Testando motor de predição...")
    # Aqui você passaria um DataFrame com os dados do cliente
    # print(run_prediction(dados_cliente))