import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from src.config.styles import *


def create_chart_grid(charts: List[Tuple[str, go.Figure]],
                          cols: int = 2,
                          titles: Optional[List[str]] = None):
        """
        Cria um grid de gráficos.

        Args:
            charts (List[Tuple[str, go.Figure]]): Lista de tuplas (título, figura)
            cols (int): Número de colunas
            titles (List[str], optional): Lista de títulos alternativos
        """
        for i in range(0, len(charts), cols):
            row_charts = charts[i:i+cols]
            columns = st.columns(cols)

            for j, (title, fig) in enumerate(row_charts):
                with columns[j]:
                    ChartCard(
                        title=titles[i+j] if titles else title
                    ).render(fig)
class ChartCard:
    """Classe base para criar cards de gráficos com estilo consistente"""

    def __init__(self, title: str, description: Optional[str] = None):
        """
        Inicializa um card de gráfico.

        Args:
            title (str): Título do gráfico
            description (str, optional): Descrição do gráfico
        """
        self.title = title
        self.description = description
        self.style = COMPONENT_STYLES['chart_card']

    def render(self, fig: go.Figure):
        """
        Renderiza o gráfico em um card estilizado.

        Args:
            fig (go.Figure): Figura do Plotly para renderizar
        """
        st.markdown(f"""
        <div style="
            background: {self.style['background']};
            border-radius: {self.style['border_radius']};
            padding: {self.style['padding']};
            box-shadow: {self.style['shadow']};
            margin: 1rem 0;
        ">
            <h3 style="margin-bottom: 1rem;">{self.title}</h3>
            {f'<p style="color: #666; margin-bottom: 1rem;">{self.description}</p>'
             if self.description else ''}
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig, use_container_width=True)
        
    


class ModelPerformanceCharts:
    """Classe para criar gráficos de performance dos modelos"""

    def __init__(self, results_df: pd.DataFrame):
        """
        Inicializa os gráficos de performance.

        Args:
            results_df (pd.DataFrame): DataFrame com resultados dos modelos
        """
        self.results = results_df
        self.colors = COLOR_PALETTES['main']
        self.template = PLOTLY_TEMPLATE

    def metrics_comparison(self) -> go.Figure:
        """Cria gráfico comparativo de métricas"""
        metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']

        fig = go.Figure()

        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=self.results['modelo'],
                y=self.results[metric],
                text=self.results[metric].round(3),
                textposition='auto',
                marker_color=self.colors[i % len(self.colors)]
            ))

        fig.update_layout(
            title='Comparação de Métricas por Modelo',
            barmode='group',
            template=self.template
        )

        return fig
    
    

    def training_times(self) -> go.Figure:
        """Cria gráfico de tempos de treinamento"""
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=self.results['modelo'],
            y=self.results['tempo_treinamento'],
            text=self.results['tempo_treinamento'].round(2),
            textposition='auto',
            marker_color=self.colors[0]
        ))

        fig.update_layout(
            title='Tempo de Treinamento por Modelo',
            yaxis_title='Tempo (segundos)',
            template=self.template
        )

        return fig

    def model_comparison_radar(self) -> go.Figure:
        """Cria gráfico radar para comparação de modelos"""
        metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']

        fig = go.Figure()

        for index, row in self.results.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[metric] for metric in metrics],
                theta=metrics,
                name=f"{row['modelo']} {index}",
                line_color=self.colors[index % len(self.colors)]
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title='Comparação Radar dos Modelos',
            template=self.template
        )

        return fig


class DataAnalysisCharts:
    """Classe para criar gráficos de análise de dados"""

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa os gráficos de análise.

        Args:
            df (pd.DataFrame): DataFrame para análise
        """
        self.df = df
        self.colors = COLOR_PALETTES['main']
        self.template = PLOTLY_TEMPLATE

    def class_distribution(self) -> go.Figure:
        """Cria gráfico de distribuição de classes"""
        class_counts = self.df['y'].value_counts()

        fig = go.Figure(data=[
            go.Pie(
                labels=class_counts.index,
                values=class_counts.values,
                hole=0.4,
                marker_colors=self.colors
            )
        ])

        fig.update_layout(
            title='Distribuição das Classes',
            template=self.template
        )

        return fig

    def missing_values(self) -> go.Figure:
        """Cria gráfico de valores faltantes"""
        missing = self.df.isnull().sum().sort_values(ascending=True)
        missing_pct = (missing / len(self.df) * 100).round(2)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=missing.index,
            x=missing_pct,
            orientation='h',
            text=missing_pct.apply(lambda x: f'{x:.1f}%'),
            textposition='auto',
            marker_color=self.colors[0]
        ))

        fig.update_layout(
            title='Análise de Valores Faltantes',
            xaxis_title='Porcentagem',
            yaxis_title='Features',
            template=self.template
        )

        return fig

    def correlation_matrix(self) -> go.Figure:
        """Cria matriz de correlação"""
        corr = self.df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))

        fig.update_layout(
            title='Matriz de Correlação',
            template=self.template
        )

        return fig

    def feature_distributions(self, features: List[str]) -> go.Figure:
        """
        Cria gráfico de distribuições das features.

        Args:
            features (List[str]): Lista de features para plotar
        """
        fig = make_subplots(rows=len(features), cols=1,
                            subplot_titles=features)

        for i, feature in enumerate(features, 1):
            fig.add_trace(
                go.Histogram(
                    x=self.df[feature],
                    name=feature,
                    marker_color=self.colors[i % len(self.colors)]
                ),
                row=i,
                col=1
            )

        fig.update_layout(
            title='Distribuição das Features',
            showlegend=False,
            template=self.template,
            height=300 * len(features)
        )

        return fig
    
    


class MLMetricsCharts:
    """Classe para criar gráficos de métricas de ML"""

    def __init__(self):
        self.colors = COLOR_PALETTES['main']
        self.template = PLOTLY_TEMPLATE

    def confusion_matrix(self, cm: np.ndarray, classes: List[str]) -> go.Figure:
        """
        Cria matriz de confusão.

        Args:
            cm (np.ndarray): Matriz de confusão
            classes (List[str]): Lista de classes
        """
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))

        fig.update_layout(
            title='Matriz de Confusão',
            xaxis_title='Previsto',
            yaxis_title='Real',
            template=self.template
        )

        return fig

    def roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
        """
        Cria curva ROC.

        Args:
            fpr (np.ndarray): False positive rate
            tpr (np.ndarray): True positive rate
            auc (float): Area under curve
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            name=f'ROC (AUC = {auc:.3f})',
            mode='lines',
            line=dict(color=self.colors[0])
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title='Curva ROC',
            xaxis_title='Taxa de Falsos Positivos',
            yaxis_title='Taxa de Verdadeiros Positivos',
            template=self.template
        )

        return fig

    def learning_curve(self, train_sizes: np.ndarray, train_scores: np.ndarray,
                       val_scores: np.ndarray) -> go.Figure:
        """
        Cria curva de aprendizado.

        Args:
            train_sizes (np.ndarray): Tamanhos de treino
            train_scores (np.ndarray): Scores de treino
            val_scores (np.ndarray): Scores de validação
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(train_scores, axis=1),
            name='Treino',
            mode='lines+markers',
            line=dict(color=self.colors[0])
        ))

        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(val_scores, axis=1),
            name='Validação',
            mode='lines+markers',
            line=dict(color=self.colors[1])
        ))

        fig.update_layout(
            title='Curva de Aprendizado',
            xaxis_title='Tamanho do Conjunto de Treino',
            yaxis_title='Score',
            template=self.template
        )

        return fig

    def feature_importance(self, features: List[str],
                           importances: List[float]) -> go.Figure:
        """
        Cria gráfico de importância das features.

        Args:
            features (List[str]): Lista de features
            importances (List[float]): Lista de importâncias
        """
        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker_color=self.colors[0]
        ))

        fig.update_layout(
            title='Importância das Features',
            xaxis_title='Importância',
            yaxis_title='Feature',
            template=self.template
        )

        return fig

    


def create_interactive_chart(data: pd.DataFrame,
                             x: str,
                             y: str,
                             chart_type: str = 'scatter',
                             color: Optional[str] = None,
                             size: Optional[str] = None,
                             title: Optional[str] = None,
                             **kwargs) -> go.Figure:
    """
    Cria um gráfico interativo customizado.

    Args:
        data (pd.DataFrame): DataFrame com os dados
        x (str): Coluna para eixo x
        y (str): Coluna para eixo y
        chart_type (str): Tipo de gráfico ('scatter', 'line', 'bar', etc)
        color (str, optional): Coluna para cor
        size (str, optional): Coluna para tamanho
        title (str, optional): Título do gráfico
        **kwargs: Argumentos adicionais para o gráfico

    Returns:
        go.Figure: Figura do Plotly
    """
    fig = go.Figure()

    if chart_type == 'scatter':
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
        fig.add_trace(go.Scatter(
            x=data[x],
            y=data[y],
            mode='lines',
            line=dict(color=COLOR_PALETTES['main'][0]),
            name=y,
            **kwargs
        ))
    elif chart_type == 'bar':
        fig.add_trace(go.Bar(
            x=data[x],
            y=data[y],
            marker_color=COLOR_PALETTES['main'][0],
            name=y,
            **kwargs
        ))

    fig.update_layout(
        title=title or f'{y} vs {x}',
        template=PLOTLY_TEMPLATE
    )

    return fig


def create_comparison_subplot(data1: Dict[str, float],
                              data2: Dict[str, float],
                              title1: str,
                              title2: str,
                              main_title: str) -> go.Figure:
    """
    Cria subplots para comparação de métricas.

    Args:
        data1 (Dict[str, float]): Primeiro conjunto de dados
        data2 (Dict[str, float]): Segundo conjunto de dados
        title1 (str): Título do primeiro subplot
        title2 (str): Título do segundo subplot
        main_title (str): Título principal

    Returns:
        go.Figure: Figura do Plotly com subplots
    """
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[title1, title2])

    # Primeiro subplot
    fig.add_trace(
        go.Bar(
            x=list(data1.keys()),
            y=list(data1.values()),
            marker_color=COLOR_PALETTES['main'][0],
            showlegend=False
        ),
        row=1, col=1
    )

    # Segundo subplot
    fig.add_trace(
        go.Bar(
            x=list(data2.keys()),
            y=list(data2.values()),
            marker_color=COLOR_PALETTES['main'][1],
            showlegend=False
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=main_title,
        template=PLOTLY_TEMPLATE
    )

    return fig


def create_trend_analysis(values: List[float],
                          dates: Optional[List[str]] = None,
                          title: str = "Análise de Tendência",
                          show_stats: bool = True) -> go.Figure:
    """
    Cria gráfico de análise de tendência com estatísticas.

    Args:
        values (List[float]): Lista de valores
        dates (List[str], optional): Lista de datas
        title (str): Título do gráfico
        show_stats (bool): Se deve mostrar estatísticas

    Returns:
        go.Figure: Figura do Plotly
    """
    x = dates if dates else list(range(len(values)))
    y = values

    fig = go.Figure()

    # Linha principal
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name='Valores',
        line=dict(color=COLOR_PALETTES['main'][0])
    ))

    if show_stats:
        # Média móvel
        moving_avg = pd.Series(values).rolling(window=3).mean()
        fig.add_trace(go.Scatter(
            x=x,
            y=moving_avg,
            mode='lines',
            name='Média Móvel (3)',
            line=dict(color=COLOR_PALETTES['main'][1], dash='dash')
        ))

        # Linha de tendência
        z = np.polyfit(range(len(values)), values, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=x,
            y=p(range(len(values))),
            mode='lines',
            name='Tendência',
            line=dict(color=COLOR_PALETTES['main'][2], dash='dot')
        ))

    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        showlegend=True
    )

    return fig


def create_downloadable_chart(fig: go.Figure,
                              filename: str = "chart.html",
                              button_text: str = "Download Gráfico"):
    """
    Cria um botão para download do gráfico em HTML.

    Args:
        fig (go.Figure): Figura do Plotly
        filename (str): Nome do arquivo para download
        button_text (str): Texto do botão
    """
    html = f"""
    <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div id="chart"></div>
            <script>
                var data = {fig.to_json()};
                Plotly.newPlot('chart', data.data, data.layout);
            </script>
        </body>
    </html>
    """

    st.download_button(
        label=button_text,
        data=html,
        file_name=filename,
        mime="text/html"
    )
