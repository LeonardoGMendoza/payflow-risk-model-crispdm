import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de estilo para um visual profissional e moderno (Modern UI)
# Alinhado com as diretrizes de Visual Storytelling discutidas na Aula 1.4
plt.style.use('ggplot')
sns.set_theme(style="whitegrid", palette="viridis")

def run_visual_eda(df):
    """
    Executa a análise visual de dados (EDA) para o projeto PayFlow.
    Este módulo isola a lógica gráfica do notebook principal para manter o código limpo.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados de crédito (ex: df_final_real).
    """
    print("\n[INFO] Iniciando Visual Storytelling dos Dados (Maturidade Analítica)...")

    if df is None or df.empty:
        print("[ERRO] O DataFrame fornecido está vazio ou é inválido.")
        return

    # 1. DISTRIBUIÇÃO E DENSIDADE (KDE Plot)
    # Recomendado para comparar distribuições de classes (Default vs Pagador)
    # Mostra onde o risco está concentrado sem os ruídos de um histograma comum.
    if 'score_credito' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x='score_credito', hue='default_90d', 
                    fill=True, common_norm=False, palette='magma', alpha=.5, linewidth=2)
        plt.title('Densidade de Probabilidade: Score de Crédito vs Risco', fontsize=14)
        plt.xlabel('Score de Crédito')
        plt.ylabel('Densidade (Frequência Relativa)')
        plt.tight_layout()
        plt.show()

    # 2. COMPARAÇÃO E OUTLIERS (Boxplot)
    # Essencial para diagnosticar se o comprometimento de renda influencia a inadimplência.
    # O Boxplot permite ver a mediana e a dispersão (quartis) simultaneamente.
    if 'comprometimento_renda' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='default_90d', y='comprometimento_renda', palette='Set2')
        plt.title('Análise de Comprometimento de Renda: Diagnóstico de Risco', fontsize=14)
        plt.xlabel('Inadimplente (1) vs Bom Pagador (0)')
        plt.ylabel('Comprometimento de Renda (%)')
        
        # Ajuste de escala: limitamos ao quantil 95% para os outliers não "esmagarem" o gráfico
        limit = df['comprometimento_renda'].quantile(0.95)
        plt.ylim(0, limit)
        plt.tight_layout()
        plt.show()

    # 3. RELAÇÕES (Heatmap de Correlação)
    # Identifica visualmente quais variáveis têm maior relação com o Target (default_90d).
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    # Máscara para ocultar a parte superior da matriz, tornando a leitura mais limpa
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0, square=True)
    plt.title('Matriz de Correlação: Diagnóstico de Variáveis Preditoras', fontsize=14)
    plt.tight_layout()
    plt.show()

    print("[SUCCESS] Gráficos gerados com sucesso seguindo o rigor técnico da Pós.")

if __name__ == "__main__":
    # Mensagem de instrução caso o script seja executado isoladamente
    print("Módulo de Visualização Analítica carregado.")
    print("Uso: Importe 'run_visual_eda' no seu Notebook e passe seu DataFrame como argumento.")