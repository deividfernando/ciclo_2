import streamlit as st
import pandas as pd
import numpy as np
from src.components.metrics import (
    MetricCard,
    MetricGrid,
    create_data_quality_metrics
)
from src.components.charts import (
    DataAnalysisCharts,
    create_interactive_chart,
    create_chart_grid
)
from src.utils.data_loader import get_data_info, create_data_summary

def show_analysis(df: pd.DataFrame):
    """
    Renderiza a página de análise de dados.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados para análise
    """
    # Cabeçalho
    st.title("📊 Análise Exploratória dos Dados")
    
    # Sidebar com opções de análise
    with st.sidebar:
        st.subheader("Opções de Análise")
        
        # Seleção de features
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        selected_features = st.multiselect(
            "Selecione as features para análise",
            options=numeric_cols,
            default=list(numeric_cols[:5])
        )
        
        # Tipo de visualização
        viz_type = st.selectbox(
            "Tipo de Visualização",
            options=['Distribuições', 'Correlações', 'Box Plots', 'Scatter Plots']
        )
        
        # Opções adicionais
        show_statistics = st.checkbox("Mostrar Estatísticas", value=True)
        show_outliers = st.checkbox("Destacar Outliers", value=False)

    # Overview dos dados
    st.markdown("### 📋 Visão Geral dos Dados")
    
    # Métricas básicas em grid
    metrics_grid = create_data_quality_metrics(df)
    metrics_grid.render()
    
    # Análise de tipos de dados
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Tipos de Dados")
        dtypes_counts = df.dtypes.value_counts()
        st.bar_chart(dtypes_counts)
    
    with col2:
        st.markdown("#### Valores Nulos")
        null_pcts = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        st.bar_chart(null_pcts.head(10))

    # Análise detalhada baseada na seleção
    st.markdown(f"### 📈 {viz_type}")
    
    data_charts = DataAnalysisCharts(df)
    
    if viz_type == 'Distribuições':
        if selected_features:
            dist_figs = data_charts.feature_distributions(selected_features)
            st.plotly_chart(dist_figs, use_container_width=True)
            
            if show_statistics:
                st.markdown("#### 📊 Estatísticas Descritivas")
                stats_df = df[selected_features].describe()
                st.dataframe(stats_df.style.format("{:.2f}"))
                
                # Teste de normalidade
                from scipy import stats
                st.markdown("#### 🔍 Teste de Normalidade (Shapiro-Wilk)")
                for feature in selected_features:
                    stat, p_value = stats.shapiro(df[feature].dropna())
                    st.write(f"**{feature}**: p-value = {p_value:.4f}")
    
    elif viz_type == 'Correlações':
        if len(selected_features) > 1:
            st.markdown("#### 🔄 Matriz de Correlação")
            corr_fig = data_charts.correlation_matrix()
            st.plotly_chart(corr_fig, use_container_width=True)
            
            if show_statistics:
                st.markdown("#### 📊 Coeficientes de Correlação")
                corr_matrix = df[selected_features].corr()
                st.dataframe(corr_matrix.style.format("{:.2f}"))
    
    elif viz_type == 'Box Plots':
        if selected_features:
            st.markdown("#### 📦 Box Plots")
            for feature in selected_features:
                fig = create_interactive_chart(
                    data=df,
                    x='y',
                    y=feature,
                    chart_type='box',
                    title=f'Box Plot - {feature}'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if show_statistics and show_outliers:
                    # Identificação de outliers
                    Q1 = df[feature].quantile(0.25)
                    Q3 = df[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[
                        (df[feature] < (Q1 - 1.5 * IQR)) | 
                        (df[feature] > (Q3 + 1.5 * IQR))
                    ][feature]
                    st.write(f"Número de outliers em {feature}: {len(outliers)}")
    
    elif viz_type == 'Scatter Plots':
        if len(selected_features) >= 2:
            st.markdown("#### 📊 Scatter Plots")
            for i in range(len(selected_features)-1):
                for j in range(i+1, len(selected_features)):
                    fig = create_interactive_chart(
                        data=df,
                        x=selected_features[i],
                        y=selected_features[j],
                        chart_type='scatter',
                        color='y' if 'y' in df.columns else None,
                        title=f'Scatter Plot - {selected_features[i]} vs {selected_features[j]}'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Análise de Features
    st.markdown("### 🎯 Análise de Features")
    
    # Seletor de feature específica
    selected_feature = st.selectbox(
        "Selecione uma feature para análise detalhada",
        options=selected_features if selected_features else numeric_cols
    )
    
    if selected_feature:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### Distribuição de {selected_feature}")
            hist_fig = create_interactive_chart(
                data=df,
                x=selected_feature,
                chart_type='histogram',
                title=f'Distribuição de {selected_feature}'
            )
            st.plotly_chart(hist_fig, use_container_width=True)
        
        with col2:
            if 'y' in df.columns:
                st.markdown(f"#### {selected_feature} por Classe")
                box_fig = create_interactive_chart(
                    data=df,
                    x='y',
                    y=selected_feature,
                    chart_type='box',
                    title=f'{selected_feature} por Classe'
                )
                st.plotly_chart(box_fig, use_container_width=True)
        
        # Estatísticas detalhadas
        if show_statistics:
            st.markdown(f"#### 📊 Estatísticas de {selected_feature}")
            
            stats = pd.DataFrame({
                "Estatística": [
                    "Média", "Mediana", "Desvio Padrão",
                    "Mínimo", "Máximo", "Assimetria", "Curtose"
                ],
                "Valor": [
                    df[selected_feature].mean(),
                    df[selected_feature].median(),
                    df[selected_feature].std(),
                    df[selected_feature].min(),
                    df[selected_feature].max(),
                    df[selected_feature].skew(),
                    df[selected_feature].kurtosis()
                ]
            })
            
            st.dataframe(stats.set_index('Estatística').style.format("{:.4f}"))

    # Análise Bivariada
    if len(selected_features) >= 2:
        st.markdown("### 🔄 Análise Bivariada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature_x = st.selectbox("Feature X", options=selected_features, key="feature_x")
        
        with col2:
            feature_y = st.selectbox("Feature Y", options=selected_features, key="feature_y")
        
        if feature_x != feature_y:
            scatter_fig = create_interactive_chart(
                data=df,
                x=feature_x,
                y=feature_y,
                chart_type='scatter',
                color='y' if 'y' in df.columns else None,
                title=f'Relação entre {feature_x} e {feature_y}'
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            if show_statistics:
                correlation = df[feature_x].corr(df[feature_y])
                st.write(f"Correlação de Pearson: {correlation:.4f}")

    # Download dos dados analisados
    st.markdown("### 📥 Download da Análise")
    
    if st.button("Preparar Relatório de Análise"):
        # Criar relatório em formato markdown
        report = f"""
        # Relatório de Análise de Dados
        
        ## Visão Geral
        - Total de registros: {len(df)}
        - Total de features: {len(df.columns)}
        - Features analisadas: {', '.join(selected_features)}
        
        ## Estatísticas Descritivas
        ```
        {df[selected_features].describe().to_markdown()}
        ```
        
        ## Correlações
        ```
        {df[selected_features].corr().to_markdown()}
        ```
        
        ## Análise de Valores Nulos
        ```
        {df[selected_features].isnull().sum().to_markdown()}
        ```
        """
        
        st.download_button(
            label="Download Relatório",
            data=report,
            file_name="analise_dados.md",
            mime="text/markdown"
        )

    # Timestamp
    st.markdown("""
    <div style='text-align: right; color: #666; padding: 20px;'>
        <small>
            Última atualização: {}
        </small>
    </div>
    """.format(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")), 
    unsafe_allow_html=True)

if __name__ == "__main__":
    # Para teste local
    try:
        df = pd.read_parquet('../data/train.parquet')
        show_analysis(df)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")