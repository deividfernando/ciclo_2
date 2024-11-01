import streamlit as st
import pandas as pd
from datetime import datetime
from src.components.metrics import *
from src.components.charts import *
from src.utils.plotting import *

def show_results(results: pd.DataFrame):
    """
    Renderiza a página de resultados e comparações.
    
    Args:
        results (pd.DataFrame): DataFrame com os resultados dos modelos
    """
    # Cabeçalho
    st.title("📈 Resultados e Comparações")
    
    # Sidebar com filtros
    with st.sidebar:
        st.subheader("Filtros")
        
        # Seleção de modelos para comparação
        selected_models = st.multiselect(
            "Selecione os modelos para comparar",
            options=results['modelo'].unique(),
            default=results['modelo'].unique()
        )
        
        # Seleção de métricas
        selected_metrics = st.multiselect(
            "Métricas para visualização",
            options=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC'],
            default=['Acurácia', 'F1-Score', 'AUC-ROC']
        )
        
        # Opções de visualização
        show_details = st.checkbox("Mostrar Detalhes", value=True)
        show_statistics = st.checkbox("Mostrar Estatísticas", value=True)
        
        # Filtrar resultados
        filtered_results = results[results['modelo'].isin(selected_models)]

    # Resumo das métricas
    st.markdown("### 📊 Resumo das Métricas")
    metrics_grid = create_model_metrics_summary(filtered_results)
    metrics_grid.render()

    # Análise Comparativa
    st.markdown("### 🔄 Análise Comparativa")
    
    tabs = st.tabs(["Métricas", "Performance", "Análise Detalhada"])
    
    # Tab de Métricas
    with tabs[0]:
        model_charts = ModelPerformanceCharts(filtered_results)
        
        # Gráfico de métricas
        st.plotly_chart(model_charts.metrics_comparison(), use_container_width=True)
        
        if show_statistics:
            # Tabela detalhada
            st.markdown("#### 📋 Métricas Detalhadas")
            detailed_metrics = filtered_results[
                ['modelo', 'Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
            ].sort_values('F1-Score', ascending=False)
            
            st.dataframe(
                detailed_metrics.style.format({
                    col: '{:.4f}' for col in detailed_metrics.columns if col != 'modelo'
                }),
                use_container_width=True
            )
            
            # Análise estatística
            st.markdown("#### 📊 Análise Estatística")
            stats_df = detailed_metrics.describe()
            st.dataframe(stats_df.style.format('{:.4f}'), use_container_width=True)
    
    # Tab de Performance
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Tempo de treinamento
            st.markdown("#### ⏱️ Tempo de Treinamento")
            st.plotly_chart(model_charts.training_times(), use_container_width=True)
        
        with col2:
            # Gráfico radar
            st.markdown("#### 🎯 Comparação Radar")
            st.plotly_chart(model_charts.model_comparison_radar(), use_container_width=True)
        
        if show_details:
            st.markdown("#### 💡 Insights de Performance")
            
            # Melhor modelo em cada métrica
            best_models = pd.DataFrame({
                'Métrica': selected_metrics,
                'Melhor Modelo': [
                    filtered_results.loc[filtered_results[metric].idxmax(), 'modelo']
                    for metric in selected_metrics
                ],
                'Valor': [
                    filtered_results[metric].max()
                    for metric in selected_metrics
                ]
            })
            
            st.dataframe(
                best_models.style.format({
                    'Valor': '{:.4f}'
                }),
                use_container_width=True
            )
    
    # Tab de Análise Detalhada
    with tabs[2]:
        # Seleção de modelo específico
        selected_model = st.selectbox(
            "Selecione um modelo para análise detalhada",
            options=selected_models
        )
        
        if selected_model:
            model_data = filtered_results[filtered_results['modelo'] == selected_model].iloc[0]
            
            # Métricas específicas do modelo
            st.markdown("#### 📊 Métricas Detalhadas")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                MetricCard(
                    title="Acurácia",
                    value=model_data['Acurácia'],
                    color="#1f77b4"
                ).render()
            
            with col2:
                MetricCard(
                    title="F1-Score",
                    value=model_data['F1-Score'],
                    color="#2ca02c"
                ).render()
            
            with col3:
                MetricCard(
                    title="AUC-ROC",
                    value=model_data['AUC-ROC'],
                    color="#ff7f0e"
                ).render()
            
            # Análise por classe
            st.markdown("#### 📈 Análise por Classe")
            
            class_metrics = pd.DataFrame({
                'Classe': ['0', '1'],
                'Precisão': [
                    model_data['Precisão_Classe_0'],
                    model_data['Precisão_Classe_1']
                ],
                'Recall': [
                    model_data['Recall_Classe_0'],
                    model_data['Recall_Classe_1']
                ],
                'F1-Score': [
                    model_data['F1-Score_Classe_0'],
                    model_data['F1-Score_Classe_1']
                ]
            })
            
            st.dataframe(
                class_metrics.style.format({
                    col: '{:.4f}' for col in class_metrics.columns if col != 'Classe'
                }),
                use_container_width=True
            )

    # Seção de Download
    st.markdown("### 📥 Download dos Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download CSV
        if st.button("Download CSV"):
            csv = filtered_results.to_csv(index=False)
            st.download_button(
                label="Confirmar Download CSV",
                data=csv,
                file_name=f"resultados_modelos_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        # Download relatório
        if st.button("Gerar Relatório"):
            report = f"""
            # Relatório de Resultados - {datetime.now().strftime('%Y-%m-%d')}
            
            ## Resumo dos Modelos Analisados
            
            {filtered_results[['modelo', 'Acurácia', 'F1-Score', 'AUC-ROC']].to_markdown()}
            
            ## Métricas por Classe
            
            ### Classe 0
            {filtered_results[['modelo', 'Precisão_Classe_0', 'Recall_Classe_0', 'F1-Score_Classe_0']].to_markdown()}
            
            ### Classe 1
            {filtered_results[['modelo', 'Precisão_Classe_1', 'Recall_Classe_1', 'F1-Score_Classe_1']].to_markdown()}
            
            ## Performance
            
            Tempos de treinamento:
            {filtered_results[['modelo', 'tempo_treinamento']].to_markdown()}
            
            ## Conclusões
            
            - Melhor modelo em Acurácia: {filtered_results.loc[filtered_results['Acurácia'].idxmax(), 'modelo']}
            - Melhor modelo em F1-Score: {filtered_results.loc[filtered_results['F1-Score'].idxmax(), 'modelo']}
            - Modelo mais rápido: {filtered_results.loc[filtered_results['tempo_treinamento'].idxmin(), 'modelo']}
            """
            
            st.download_button(
                label="Baixar Relatório",
                data=report,
                file_name=f"relatorio_resultados_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )

    # Conclusões
    st.markdown("### 💡 Conclusões")
    
    with st.expander("Ver análise completa"):
        # Melhor modelo geral
        best_model = filtered_results.loc[filtered_results['F1-Score'].idxmax(), 'modelo']
        best_score = filtered_results['F1-Score'].max()
        
        st.markdown(f"""
        #### 🏆 Melhor Modelo: {best_model}
        
        - F1-Score: {best_score:.4f}
        - Tempo de treinamento: {filtered_results.loc[filtered_results['F1-Score'].idxmax(), 'tempo_treinamento']:.2f}s
        
        #### 📈 Análise Geral
        
        - Distribuição das métricas entre os modelos
        - Trade-off entre performance e tempo de treinamento
        - Comportamento por classe
        
        """)

    # Timestamp
    create_timestamp_info()

if __name__ == "__main__":
    # Para teste local
    try:
        results = pd.read_parquet('../resultados_modelos.parquet')
        show_results(results)
    except Exception as e:
        st.error(f"Erro ao carregar resultados: {str(e)}")