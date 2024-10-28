import streamlit as st
import pandas as pd
from datetime import datetime
from src.components.metrics import *
from src.components.charts import *
from src.utils.plotting import *
from src.config.constants import *

def show_dashboard(df: pd.DataFrame, results: pd.DataFrame):
    """
    Renderiza a página do dashboard.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        results (pd.DataFrame): DataFrame com resultados dos modelos
    """
    # Cabeçalho
    st.title("🎯 Dashboard ML Ensemble Analysis")
    
    # Sidebar com filtros
    with st.sidebar:
        st.subheader("Filtros")
        
        # Seleção de modelos
        selected_models = st.multiselect(
            "Selecione os modelos",
            options=results['modelo'].unique(),
            default=results['modelo'].unique()
        )
        
        # Seleção de métricas
        selected_metrics = st.multiselect(
            "Selecione as métricas",
            options=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC'],
            default=['Acurácia', 'F1-Score']
        )
        
        # Atualizar resultados com base nos filtros
        filtered_results = results[
            results['modelo'].isin(selected_models)
        ]

    # Layout principal
    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        MetricCard(
            title="Melhor Acurácia",
            value=filtered_results['Acurácia'].max(),
            delta=f"+{(filtered_results['Acurácia'].max() - filtered_results['Acurácia'].mean()):.2%}",
            suffix="%",
            description=f"Modelo: {filtered_results.loc[filtered_results['Acurácia'].idxmax(), 'modelo']}"
        ).render()
    
    with col2:
        MetricCard(
            title="F1-Score Médio",
            value=filtered_results['F1-Score'].mean(),
            suffix="",
            description="Média entre modelos"
        ).render()
    
    with col3:
        MetricCard(
            title="Tempo Médio",
            value=filtered_results['tempo_treinamento'].mean(),
            suffix="s",
            description="Tempo de treinamento"
        ).render()
    
    with col4:
        MetricCard(
            title="ROC AUC",
            value=filtered_results['AUC-ROC'].mean(),
            description="Média entre modelos"
        ).render()

    # Gráficos principais
    st.markdown("### 📊 Análise de Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_charts = ModelPerformanceCharts(filtered_results)
        metrics_fig = model_charts.metrics_comparison()
        st.plotly_chart(metrics_fig, use_container_width=True)
    
    with col2:
        times_fig = model_charts.training_times()
        st.plotly_chart(times_fig, use_container_width=True)

    # Análise de dados
    st.markdown("### 📈 Análise dos Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_charts = DataAnalysisCharts(df)
        dist_fig = data_charts.class_distribution()
        st.plotly_chart(dist_fig, use_container_width=True)
    
    with col2:
        missing_fig = data_charts.missing_values()
        st.plotly_chart(missing_fig, use_container_width=True)

    # Comparação detalhada de modelos
    st.markdown("### 🔍 Comparação Detalhada de Modelos")
    
    # Tabela de comparação
    comparison_df = filtered_results[
        ['modelo', 'Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC', 'tempo_treinamento']
    ].sort_values('F1-Score', ascending=False)
    
    st.dataframe(
        comparison_df.style.format({
            'Acurácia': '{:.2%}',
            'Precisão': '{:.2%}',
            'Recall': '{:.2%}',
            'F1-Score': '{:.2%}',
            'AUC-ROC': '{:.2%}',
            'tempo_treinamento': '{:.2f}s'
        }),
        use_container_width=True
    )

    # Gráfico radar
    st.markdown("### 📊 Análise Radar")
    radar_fig = model_charts.model_comparison_radar()
    st.plotly_chart(radar_fig, use_container_width=True)

    # Seção de insights
    st.markdown("### 💡 Principais Insights")
    
    with st.expander("Ver detalhes"):
        # Melhor modelo
        best_model = filtered_results.loc[filtered_results['F1-Score'].idxmax(), 'modelo']
        best_score = filtered_results['F1-Score'].max()
        
        st.markdown(f"""
        **Melhor Modelo**: {best_model}
        - F1-Score: {best_score:.2%}
        - Características do modelo: {MODELS_INFO[best_model]['desc']}
        """)
        
        # Tendências observadas
        st.markdown("""
        **Tendências Observadas**:
        - Relação entre tempo de treinamento e performance
        - Balanceamento entre precisão e recall
        - Impacto da estratégia de pré-processamento
        """)
        
        # Recomendações
        st.markdown("""
        **Recomendações**:
        1. Otimização de hiperparâmetros para modelos específicos
        2. Investigação de features importantes
        3. Análise de casos de erro comum
        """)

    # Download de relatório
    st.markdown("### 📥 Download do Relatório")
    
    if st.button("Gerar Relatório PDF"):
        # TODO: Implementar geração de PDF
        st.info("Funcionalidade em desenvolvimento")

    # Timestamp
    create_timestamp_info()

    # Footer com informações adicionais
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <small>
            Dashboard atualizado em tempo real. Dados processados usando Python e Streamlit.<br>
            Desenvolvido pela equipe AlphaEdTech
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Para teste local
    import pandas as pd
    
    try:
        df = pd.read_parquet('../data/train.parquet')
        results = pd.read_parquet('../resultados_modelos.parquet')
        show_dashboard(df, results)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")