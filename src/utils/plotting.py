import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from src.config.constants import CHART_CONFIG, DEFAULT_METRICS
from src.config.styles import COLOR_PALETTES, PLOTLY_TEMPLATE

class ModelVisualization:
    """Classe para criar visualizações relacionadas aos modelos de ML"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results = results_df
        self.colors = COLOR_PALETTES['main']
        self.template = PLOTLY_TEMPLATE

    def plot_model_metrics_comparison(self, height: int = CHART_CONFIG['default_height']) -> go.Figure:
        """
        Cria gráfico comparativo das métricas entre modelos.
        
        Args:
            height (int): Altura do gráfico
            
        Returns:
            go.Figure: Figura do Plotly
        """
        fig = go.Figure()
        
        for metric in DEFAULT_METRICS:
            fig.add_trace(go.Bar(
                name=metric,
                x=self.results['modelo'].unique(),
                y=self.results[metric],
                text=self.results[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Comparação de Métricas entre Modelos',
            barmode='group',
            height=height,
            showlegend=True,
            hovermode='x unified',
            template=self.template,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig

    def plot_training_times(self, height: int = CHART_CONFIG['default_height']) -> go.Figure:
        """
        Cria gráfico de barras dos tempos de treinamento.
        
        Args:
            height (int): Altura do gráfico
            
        Returns:
            go.Figure: Figura do Plotly
        """
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
            xaxis_title='Modelo',
            yaxis_title='Tempo (segundos)',
            height=height,
            template=self.template
        )
        
        return fig

    def plot_feature_importance(self, model_name: str, features: list, importances: list, 
                              height: int = CHART_CONFIG['default_height']) -> go.Figure:
        """
        Cria gráfico de importância das features.
        
        Args:
            model_name (str): Nome do modelo
            features (list): Lista de features
            importances (list): Lista de importâncias
            height (int): Altura do gráfico
            
        Returns:
            go.Figure: Figura do Plotly
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker_color=self.colors[0]
        ))
        
        fig.update_layout(
            title=f'Importância das Features - {model_name}',
            xaxis_title='Importância',
            yaxis_title='Feature',
            height=height,
            template=self.template
        )
        
        return fig

class DataVisualization:
    """Classe para criar visualizações relacionadas aos dados"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.colors = COLOR_PALETTES['main']
        self.template = PLOTLY_TEMPLATE

    def plot_class_distribution(self, height: int = CHART_CONFIG['default_height']) -> go.Figure:
        """
        Cria gráfico de distribuição das classes.
        
        Args:
            height (int): Altura do gráfico
            
        Returns:
            go.Figure: Figura do Plotly
        """
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
            height=height,
            template=self.template
        )
        
        return fig

    def plot_feature_distributions(self, features: list, 
                                 height: int = CHART_CONFIG['default_height']) -> go.Figure:
        """
        Cria gráfico de distribuições das features.
        
        Args:
            features (list): Lista de features para plotar
            height (int): Altura do gráfico
            
        Returns:
            go.Figure: Figura do Plotly
        """
        num_features = len(features)
        fig = make_subplots(rows=num_features, cols=1, 
                           subplot_titles=features,
                           height=height * num_features // 2)
        
        for i, feature in enumerate(features, 1):
            fig.add_trace(
                go.Histogram(
                    x=self.df[feature],
                    name=feature,
                    marker_color=self.colors[i % len(self.colors)]
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            title='Distribuição das Features',
            showlegend=False,
            template=self.template
        )
        
        return fig

    def plot_correlation_matrix(self, height: int = CHART_CONFIG['default_height']) -> go.Figure:
        """
        Cria matriz de correlação.
        
        Args:
            height (int): Altura do gráfico
            
        Returns:
            go.Figure: Figura do Plotly
        """
        corr_matrix = self.df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title='Matriz de Correlação',
            height=height,
            template=self.template
        )
        
        return fig

    def plot_missing_values(self, height: int = CHART_CONFIG['default_height']) -> go.Figure:
        """
        Cria gráfico de valores faltantes.
        
        Args:
            height (int): Altura do gráfico
            
        Returns:
            go.Figure: Figura do Plotly
        """
        missing = self.df.isnull().sum().sort_values(ascending=True)
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=missing.index,
            x=missing_pct,
            orientation='h',
            marker_color=self.colors[0],
            name='Porcentagem de Valores Faltantes',
            text=missing_pct.apply(lambda x: f'{x:.1f}%'),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Análise de Valores Faltantes',
            xaxis_title='Porcentagem de Valores Faltantes',
            yaxis_title='Feature',
            height=height,
            template=self.template
        )
        
        return fig

def create_model_comparison_chart(results: pd.DataFrame) -> go.Figure:
    """
    Cria gráfico comparativo entre modelos.
    
    Args:
        results (pd.DataFrame): DataFrame com resultados dos modelos
        
    Returns:
        go.Figure: Figura do Plotly
    """
    model_viz = ModelVisualization(results)
    return model_viz.plot_model_metrics_comparison()

def create_feature_importance_chart(model_name: str, features: list, importances: list) -> go.Figure:
    """
    Cria gráfico de importância das features.
    
    Args:
        model_name (str): Nome do modelo
        features (list): Lista de features
        importances (list): Lista de importâncias
        
    Returns:
        go.Figure: Figura do Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=features,
        x=importances,
        orientation='h',
        marker_color=COLOR_PALETTES['main'][0]
    ))
    
    fig.update_layout(
        title=f'Importância das Features - {model_name}',
        xaxis_title='Importância',
        yaxis_title='Feature',
        height=CHART_CONFIG['default_height'],
        template=PLOTLY_TEMPLATE
    )
    
    return fig

def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> go.Figure:
    """
    Cria matriz de confusão.
    
    Args:
        cm (np.ndarray): Matriz de confusão
        class_names (list): Nomes das classes
        
    Returns:
        go.Figure: Figura do Plotly
    """
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        showscale=True
    )
    
    fig.update_layout(
        title='Matriz de Confusão',
        xaxis_title='Previsto',
        yaxis_title='Real',
        height=CHART_CONFIG['default_height'],
        template=PLOTLY_TEMPLATE
    )
    
    return fig

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> go.Figure:
    """
    Cria curva ROC.
    
    Args:
        fpr (np.ndarray): Taxa de falsos positivos
        tpr (np.ndarray): Taxa de verdadeiros positivos
        roc_auc (float): Área sob a curva ROC
        
    Returns:
        go.Figure: Figura do Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(color=COLOR_PALETTES['main'][0])
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='grey')
    ))
    
    fig.update_layout(
        title='Curva ROC',
        xaxis_title='Taxa de Falsos Positivos',
        yaxis_title='Taxa de Verdadeiros Positivos',
        height=CHART_CONFIG['default_height'],
        template=PLOTLY_TEMPLATE
    )
    
    return fig

def plot_learning_curves(train_scores: list, val_scores: list, train_sizes: list) -> go.Figure:
    """
    Cria gráfico de curvas de aprendizado.
    
    Args:
        train_scores (list): Scores de treino
        val_scores (list): Scores de validação
        train_sizes (list): Tamanhos de treino
        
    Returns:
        go.Figure: Figura do Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=np.mean(train_scores, axis=1),
        mode='lines+markers',
        name='Train',
        line=dict(color=COLOR_PALETTES['main'][0])
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=np.mean(val_scores, axis=1),
        mode='lines+markers',
        name='Validation',
        line=dict(color=COLOR_PALETTES['main'][1])
    ))
    
    fig.update_layout(
        title='Curvas de Aprendizado',
        xaxis_title='Tamanho do Conjunto de Treino',
        yaxis_title='Score',
        height=CHART_CONFIG['default_height'],
        template=PLOTLY_TEMPLATE
    )
    
    return fig