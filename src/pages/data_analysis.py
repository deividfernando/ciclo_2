import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from src.components.metrics import (
    MetricCard,
    MetricGrid,
    create_data_quality_metrics
)
from src.components.charts import (
    DataAnalysisCharts,
    create_chart_grid
)
from src.utils.data_loader import get_data_info, create_data_summary
from src.config.styles import COLOR_PALETTES, PLOTLY_TEMPLATE
import plotly.graph_objects as go

def prepare_data_for_plotting(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Prepara os dados para plotagem, convertendo tipos complexos para tipos Python nativos.
    
    Args:
        df (pd.DataFrame): DataFrame original
        column_name (str): Nome da coluna para processar
        
    Returns:
        pd.DataFrame: DataFrame com tipos de dados compatíveis com Plotly
    """
    # Criar uma cópia para não modificar o DataFrame original
    df_plot = df.copy()
    
    # Converter tipos de dados complexos
    if df_plot[column_name].dtype.name == 'category':
        df_plot[column_name] = df_plot[column_name].astype(str)
    elif df_plot[column_name].dtype.name.startswith('datetime'):
        df_plot[column_name] = df_plot[column_name].astype(str)
    elif df_plot[column_name].dtype.name.startswith('float'):
        df_plot[column_name] = df_plot[column_name].astype('float64')
    elif df_plot[column_name].dtype.name.startswith('int'):
        df_plot[column_name] = df_plot[column_name].astype('int64')
    
    return df_plot

def create_interactive_chart(data: pd.DataFrame,
                           x: str,
                           y: str = None,
                           chart_type: str = 'scatter',
                           color: str = None,
                           size: str = None,
                           title: str = None,
                           **kwargs) -> go.Figure:
    """
    Cria um gráfico interativo customizado.

    Args:
        data (pd.DataFrame): DataFrame com os dados
        x (str): Coluna para eixo x
        y (str, optional): Coluna para eixo y. Opcional para histogramas
        chart_type (str): Tipo de gráfico ('scatter', 'line', 'bar', 'histogram')
        color (str, optional): Coluna para cor
        size (str, optional): Coluna para tamanho
        title (str, optional): Título do gráfico
        **kwargs: Argumentos adicionais para o gráfico

    Returns:
        go.Figure: Figura do Plotly
    """
    fig = go.Figure()

    # Garantir que os dados estão em formato compatível
    if x in data.columns:
        data = prepare_data_for_plotting(data, x)
    if y and y in data.columns:
        data = prepare_data_for_plotting(data, y)

    if chart_type == 'histogram':
        fig.add_trace(go.Histogram(
            x=data[x],
            name=x,
            marker_color=color if color else COLOR_PALETTES['main'][0],
            **kwargs
        ))
    elif chart_type == 'scatter':
        if y is None:
            raise ValueError("y parameter is required for scatter plots")
        fig.add_trace(go.Scatter(
            x=data[x],
            y=data[y],
            mode='markers',
            marker=dict(
                color=data[color] if color else COLOR_PALETTES['main'][0],
                size=data[size] if size else 8
            ),
            name=y,
            **kwargs
        ))
    elif chart_type == 'line':
        if y is None:
            raise ValueError("y parameter is required for line plots")
        fig.add_trace(go.Scatter(
            x=data[x],
            y=data[y],
            mode='lines',
            line=dict(color=COLOR_PALETTES['main'][0]),
            name=y,
            **kwargs
        ))
    elif chart_type == 'bar':
        if y is None:
            raise ValueError("y parameter is required for bar plots")
        fig.add_trace(go.Bar(
            x=data[x],
            y=data[y],
            marker_color=COLOR_PALETTES['main'][0],
            name=y,
            **kwargs
        ))
    elif chart_type == 'box':
        if y is None:
            raise ValueError("y parameter is required for box plots")
        fig.add_trace(go.Box(
            x=data[x],
            y=data[y],
            name=y,
            marker_color=COLOR_PALETTES['main'][0],
            **kwargs
        ))

    fig.update_layout(
        title=title or f'Análise de {x}' + (f' vs {y}' if y else ''),
        template=PLOTLY_TEMPLATE
    )

    return fig

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
        dtypes_counts = df.dtypes.astype(str).value_counts()
        fig_dtypes = create_interactive_chart(
            data=pd.DataFrame({
                'tipo': dtypes_counts.index.astype(str),
                'count': dtypes_counts.values.astype(int)
            }),
            x='tipo',
            y='count',
            chart_type='bar',
            title='Distribuição de Tipos de Dados'
        )
        st.plotly_chart(fig_dtypes, use_container_width=True)
    
    with col2:
        st.markdown("#### Valores Nulos")
        null_pcts = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        fig_nulls = create_interactive_chart(
            data=pd.DataFrame({
                'coluna': null_pcts.index.astype(str),
                'percentual': null_pcts.values.astype(float)
            }),
            x='coluna',
            y='percentual',
            chart_type='bar',
            title='Percentual de Valores Nulos'
        )
        st.plotly_chart(fig_nulls, use_container_width=True)

    # Análise detalhada baseada na seleção
    st.markdown(f"### 📈 {viz_type}")
    
    data_charts = DataAnalysisCharts(df)
    
    if viz_type == 'Distribuições':
        if selected_features:
            for feature in selected_features:
                plot_data = prepare_data_for_plotting(df, feature)
                fig = create_interactive_chart(
                    data=plot_data,
                    x=feature,
                    chart_type='histogram',
                    title=f'Distribuição de {feature}'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if show_statistics:
                st.markdown("#### 📊 Estatísticas Descritivas")
                stats_df = df[selected_features].describe()
                st.dataframe(stats_df.style.format("{:.2f}"))
                
                # Teste de normalidade
                st.markdown("#### 🔍 Teste de Normalidade (Shapiro-Wilk)")
                for feature in selected_features:
                    try:
                        # Remover valores nulos e infinitos
                        data_clean = df[feature].dropna()
                        data_clean = data_clean[~np.isinf(data_clean)]
                        
                        if len(data_clean) < 3:
                            st.write(f"**{feature}**: Dados insuficientes para o teste")
                            continue
                            
                        statistic, p_value = scipy_stats.shapiro(data_clean)
                        st.write(f"""
                        **{feature}**: 
                        - p-value = {p_value:.4f}
                        - statistic = {statistic:.4f}
                        - Conclusão: {'Distribuição normal (p > 0.05)' if p_value > 0.05 else 'Distribuição não normal (p ≤ 0.05)'}
                        """)
                    except Exception as e:
                        st.write(f"**{feature}**: Erro ao realizar teste - {str(e)}")
    
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
                plot_data = prepare_data_for_plotting(df, feature)
                if 'y' in df.columns:
                    plot_data = prepare_data_for_plotting(plot_data, 'y')
                fig = create_interactive_chart(
                    data=plot_data,
                    x='y' if 'y' in df.columns else None,
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
                    plot_data = prepare_data_for_plotting(df, selected_features[i])
                    plot_data = prepare_data_for_plotting(plot_data, selected_features[j])
                    if 'y' in df.columns:
                        plot_data = prepare_data_for_plotting(plot_data, 'y')
                    fig = create_interactive_chart(
                        data=plot_data,
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
            plot_data = prepare_data_for_plotting(df, selected_feature)
            hist_fig = create_interactive_chart(
                data=plot_data,
                x=selected_feature,
                chart_type='histogram',
                title=f'Distribuição de {selected_feature}'
            )
            st.plotly_chart(hist_fig, use_container_width=True)
        
        with col2:
            if 'y' in df.columns:
                st.markdown(f"#### {selected_feature} por Classe")
                plot_data = prepare_data_for_plotting(df, selected_feature)
                plot_data = prepare_data_for_plotting(plot_data, 'y')
                box_fig = create_interactive_chart(
                    data=plot_data,
                    x='y',
                    y=selected_feature,
                    chart_type='box',
                    title=f'{selected_feature} por Classe'
                )
                st.plotly_chart(box_fig, use_container_width=True)
        
        # Estatísticas detalhadas
        if show_statistics:
                st.markdown(f"#### 📊 Estatísticas de {selected_feature}")
                
                stats_df = pd.DataFrame({
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
                
                st.dataframe(stats_df.set_index('Estatística').style.format("{:.4f}"))

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