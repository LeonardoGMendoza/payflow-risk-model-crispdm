import pytest
import os
import sys

# Garante que o Python encontre a pasta 'src' estando na raiz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ajustado para importar a função correta do seu src/predict.py
from src.predict import run_prediction

def test_modelo_carregado():
    """ Verifica se a predição básica funciona (indica que o modelo carregou) """
    cliente_fake = {
        'idade': 30,
        'renda_mensal': 5000,
        'valor_solicitado': 10000,
        'score_credito': 600,
        'tempo_emprego_anos': 3
    }
    resultado = run_prediction(cliente_fake)
    
    assert isinstance(resultado, dict), "Deveria retornar um dicionário"
    assert "decisao_politica" in resultado, "Deveria ter a chave de decisão"
    assert "erro" not in resultado, f"Erro inesperado: {resultado.get('erro')}"

def test_calculo_comprometimento():
    """ Verifica se o motor de decisão está calculando o comprometimento """
    cliente_fake = {
        'renda_mensal': 2000,
        'valor_solicitado': 1000
    }
    resultado = run_prediction(cliente_fake)
    
    assert "info_adicional" in resultado
    assert "comprometimento_calculado" in resultado["info_adicional"]