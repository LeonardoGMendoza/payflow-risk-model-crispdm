import os
import pandas as pd
from pathlib import Path

def get_project_root():
    """
    Retorna a raiz do projeto de forma absoluta.
    Utiliza Path(__file__) para garantir que o caminho seja resolvido
    corretamente independentemente de onde o script seja invocado.
    """
    return Path(__file__).resolve().parent.parent

def find_idh_file(filename='IDH_2010.xls'):
    """
    Localiza a planilha na pasta 'Base de dados IDH' a partir da raiz.
    Retorna o objeto Path absoluto para o arquivo ou None se não for encontrado.
    """
    root = get_project_root()
    caminho = root / "Base de dados IDH" / filename
    
    if caminho.exists():
        return caminho
    return None

def load_and_clean_idh(file_path):
    """
    Executa o pipeline de Extração e Limpeza (ETL) seguindo o rigor de MLOps.
    
    Passos:
    1. Carga via engine 'xlrd' (obrigatório para ficheiros .xls legados).
    2. Seleção de 6 features estratégicas para a análise de mercado.
    3. Normalização de nomes para snake_case (evita problemas de encoding e espaços).
    """
    # Carga dos dados brutos com motor de compatibilidade
    df_raw = pd.read_excel(file_path, engine='xlrd')
    
    # Seleção de colunas com base no Business Understanding (Pergunta Norteadora)
    df_clean = df_raw[[
        'Nome da Unidade da Federação', 
        'Município', 
        'IDHM', 
        'IDHM Educação', 
        'IDHM Longevidade', 
        'IDHM Renda'
    ]].copy()
    
    # Renomeação padronizada para garantir reprodutibilidade
    df_clean.columns = [
        'nome_da_unidade_da_federacao', 
        'municipio', 
        'idhm', 
        'idhm_educacao', 
        'idhm_longevidade', 
        'idhm_renda'
    ]
    
    return df_clean

def run_data_audit(df_idhm, df_uf=None):
    """
    Executa a auditoria automática de qualidade (Data Quality Audit).
    
    Validações:
    - Schema: Verifica se a limpeza de colunas ocorreu conforme o esperado.
    - Integrity: Garante que não há valores nulos no indicador principal (IDHM).
    - Domain: Valida se o IDH está na escala probabilística correta entre 0 e 1.
    - Geography: Valida se a base contém as 27 unidades federativas brasileiras.
    """
    # Validação de Schema (Garante que o pipeline de limpeza não quebrou)
    assert 'nome_da_unidade_da_federacao' in df_idhm.columns, "Erro de Schema: Colunas não mapeadas."
    
    # Validação de Dados Ausentes (Zero Tolerance para nulos no target)
    assert df_idhm['idhm'].isnull().sum() == 0, "Erro de Integridade: Detectados valores nulos no IDHM."
    
    # Validação de Escala Matemática (Regra de Negócio do IDH)
    assert df_idhm['idhm'].max() <= 1.0 and df_idhm['idhm'].min() >= 0, "Erro de Domínio: Escala IDH inválida."
    
    # Validação Geográfica (Opcional, se o agrupamento for fornecido)
    if df_uf is not None:
        assert len(df_uf) == 27, f"Erro Geográfico: Encontradas {len(df_uf)} UFs, esperado 27."
    
    return True