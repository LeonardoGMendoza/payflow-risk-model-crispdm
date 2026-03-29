import pytest
import sys
import os
from pathlib import Path

# --- Rigor de Engenharia: Localização Dinâmica de Módulos ---
# Estrutura detectada:
# Raiz/
#  ├── src/idhm.py
#  └── tests/test_idhm.py (Este arquivo)

# Utilizamos resolve() para obter o caminho absoluto e evitar erros de contexto.
current_file = Path(__file__).resolve()

# Como o arquivo está em tests/test_idhm.py:
# .parent é a pasta 'tests/'
# .parent.parent é a RAIZ do projeto onde reside a pasta 'src'
root_path = current_file.parent.parent
src_path = str(root_path / "src")

# Injetamos o caminho absoluto da pasta 'src' no início do PATH para prioridade total
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Tentativa de importação absoluta do motor IDH
try:
    from idhm import find_idh_file, load_and_clean_idh, run_data_audit
except ImportError as e:
    # Diagnóstico técnico para MLOps
    print(f"\n❌ Erro de Importação: O motor 'idhm.py' não foi localizado em {src_path}")
    print(f"Diretório atual de execução: {os.getcwd()}")
    raise e

def test_file_discovery():
    """Valida se o motor de busca localiza a planilha IDH_2010.xls na pasta de base de dados."""
    path = find_idh_file()
    assert path is not None, "❌ Erro: O motor de busca não localizou a planilha IDH_2010.xls."
    assert path.exists(), "❌ Erro: O caminho foi detectado, mas o arquivo físico não existe no disco."

def test_data_cleaning_logic():
    """Valida se a limpeza de colunas (snake_case) está operante e conforme o schema."""
    path = find_idh_file()
    df = load_and_clean_idh(path)
    
    # Lista de colunas esperadas após o processamento rigoroso no idhm.py
    expected_cols = [
        'nome_da_unidade_da_federacao', 
        'municipio', 
        'idhm', 
        'idhm_educacao', 
        'idhm_longevidade', 
        'idhm_renda'
    ]
    
    # Validação de Schema e Tipagem (Indispensável para MLOps)
    assert list(df.columns) == expected_cols, "❌ Erro: A renomeação ou seleção de colunas falhou."
    assert df['idhm'].dtype == 'float64', "❌ Erro: A coluna IDHM deveria ser do tipo numérico (float)."

def test_audit_integrity():
    """Valida se a função de auditoria aprova a base carregada e detecta a escala técnica."""
    path = find_idh_file()
    df = load_and_clean_idh(path)
    
    # O motor de auditoria deve retornar True se os dados respeitarem as regras de MLOps (escala 0 a 1)
    assert run_data_audit(df) is True, "❌ Erro: A base de dados falhou na auditoria de integridade técnica."