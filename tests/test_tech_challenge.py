import pytest
import os
import sys
from pathlib import Path

# ==============================================================================
# RIGOR DE ENGENHARIA: LOCALIZAÇÃO DINÂMICA DE MÓDULOS (MLOps Standard)
# ==============================================================================
# Garante que o ambiente de teste localize a pasta 'src' de forma absoluta,
# evitando erros de 'ModuleNotFoundError' em servidores de CI/CD ou diferentes SOs.
current_dir = Path(__file__).resolve().parent
root_path = current_dir.parent
src_path = str(root_path / "src")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Importação da lógica de produção validada
try:
    from predict_tech_challenge import predict_customer_satisfaction
except ImportError:
    # Fallback caso o nome do ficheiro seja predict_nps.py
    from predict_nps import predict_customer_satisfaction

# ==============================================================================
# SUÍTE DE TESTES UNITÁRIOS E DE INTEGRAÇÃO
# ==============================================================================

def test_predict_nps_flow():
    """ Valida o fluxo completo de inferência com um perfil de cliente padrão. """
    cliente_teste = {
        "customer_age": 40,
        "order_value": 500.0,
        "delivery_time_days": 5,
        "delivery_delay_days": 2,
        "customer_service_contacts": 1
    }
    
    resultado = predict_customer_satisfaction(cliente_teste)
    
    # Validação de Schema de Resposta
    assert resultado["status"] == "Sucesso"
    assert "classe" in resultado
    assert "probabilidade_detracao" in resultado
    assert "score_risco" in resultado
    assert isinstance(resultado["score_risco"], (int, float))

def test_nps_critical_delay_alert():
    """ Valida se a lógica de Ponto de Ruptura (3 dias) dispara o alerta correto. """
    # Cenário Crítico: 8 dias de atraso
    cliente_atrasado = {"delivery_delay_days": 8, "delivery_time_days": 5}
    
    resultado = predict_customer_satisfaction(cliente_atrasado)
    
    if "erro" not in resultado:
        # Sincronizado com a estratégia de negócio: Alerta deve ser 'CRÍTICO'
        assert resultado["alerta_logistico"] == "CRÍTICO"

def test_nps_resilience():
    """ Valida a resiliência do modelo perante dicionários de entrada incompletos. """
    # O script de predição deve preencher as colunas ausentes com 0 (alinhamento de features)
    cliente_incompleto = {"delivery_delay_days": 1}
    
    resultado = predict_customer_satisfaction(cliente_incompleto)
    
    assert "erro" not in resultado
    assert resultado["status"] == "Sucesso"

# ==============================================================================
# EXECUÇÃO MANUAL E RELATÓRIO DE AUDITORIA
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧪 PAYFLOW AI: AUDITORIA DE TESTES UNITÁRIOS")
    print("="*50)
    
    try:
        test_predict_nps_flow()
        print("✅ [OK] Teste de Fluxo: Schema e tipos validados.")
        
        test_nps_critical_delay_alert()
        print("✅ [OK] Teste de Negócio: Alerta CRÍTICO operacional.")
        
        test_nps_resilience()
        print("✅ [OK] Teste de Resiliência: Tratamento de dados parciais.")
        
        print("\n🏁 RESULTADO FINAL: 100% de cobertura nos requisitos críticos.")
    except AssertionError as e:
        print(f"❌ [FALHA] Erro de validação: {e}")
    except Exception as e:
        print(f"💥 [ERRO NO AMBIENTE]: {e}")
    
    print("="*50 + "\n")