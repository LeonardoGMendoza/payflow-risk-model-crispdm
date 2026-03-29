import pytest
import os
import sys

# Garante que o Python encontre a pasta 'src' estando na raiz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict_nps import predict_customer_satisfaction

def test_predict_nps_flow():
    """ Teste de fluxo completo de predição """
    cliente_teste = {
        "delivery_delay_days": 10,
        "total_order_value": 500
    }
    resultado = predict_customer_satisfaction(cliente_teste)
    
    assert isinstance(resultado, dict)
    assert "classe" in resultado
    assert "probabilidade_detracao" in resultado

def test_nps_critical_delay_alert():
    """ Teste de lógica de alerta logístico """
    cliente_atrasado = {"delivery_delay_days": 8}
    resultado = predict_customer_satisfaction(cliente_atrasado)
    
    if "erro" not in resultado:
        assert resultado["alerta_logistico"] == "ALTO"

def test_nps_resilience():
    """ Teste de robustez com dados parciais """
    cliente_vazio = {"delivery_delay_days": 1}
    resultado = predict_customer_satisfaction(cliente_vazio)
    
    assert "erro" not in resultado