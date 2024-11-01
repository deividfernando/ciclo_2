import streamlit as st
import pandas as pd
from typing import Optional, Union, Dict, List
from src.config.styles import COMPONENT_STYLES
from datetime import datetime

class MetricCard:
    """Classe para criar cards de métricas personalizados"""
    
    def __init__(self, title: str, value: Union[str, float], 
                 delta: Optional[str] = None,
                 description: Optional[str] = None,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 color: Optional[str] = None):
        """
        Inicializa um card de métrica.
        
        Args:
            title (str): Título da métrica
            value (Union[str, float]): Valor da métrica
            delta (str, optional): Variação da métrica
            description (str, optional): Descrição adicional
            prefix (str, optional): Prefixo para o valor
            suffix (str, optional): Sufixo para o valor
            color (str, optional): Cor personalizada
        """
        self.title = title
        self.value = value
        self.delta = delta
        self.description = description
        self.prefix = prefix or ""
        self.suffix = suffix or ""
        self.color = color or "#1f77b4"
        self.style = COMPONENT_STYLES['metric_card']

    def format_value(self) -> str:
        """Formata o valor da métrica com prefixo e sufixo"""
        if isinstance(self.value, float):
            formatted_value = f"{self.value:.2f}"
        else:
            formatted_value = str(self.value)
        return f"{self.prefix}{formatted_value}{self.suffix}"

    def render(self):
        """Renderiza o card da métrica no Streamlit"""
        if self.delta:
            st.markdown(
                f"""
                <div style="
                    background: {self.style['background']};
                    border-radius: {self.style['border_radius']};
                    padding: {self.style['padding']};
                    box-shadow: {self.style['shadow']};
                    border-left: 4px solid {self.color};
                ">
                    <p style="color: #666; margin-bottom: 4px; font-size: 14px;">
                        {self.title}
                    </p>
                    <h3 style="color: {self.color}; margin: 0; font-size: 24px;">{self.format_value()}</h3>
                    {f'<p style="color: {"green" if float(self.delta.strip("%")) > 0 else "red"}; margin: 4px 0; margin-top: -19px">{self.delta}</p>'}
                    {f'<p style="color: #666; margin: 4px 0; font-size: 12px; margin-top: -7px">{self.description}</p>' if self.description else ''}
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background: {self.style['background']};
                    border-radius: {self.style['border_radius']};
                    padding: {self.style['padding']};
                    box-shadow: {self.style['shadow']};
                    border-left: 4px solid {self.color};
                ">
                    <p style="color: #666; margin-bottom: 4px; font-size: 14px;">
                        {self.title}
                    </p>
                    <h3 style="color: {self.color}; margin: 0; font-size: 24px;">{self.format_value()}</h3>
                    {f'<p style="color: #666; margin: 4px 0; font-size: 12px;">{self.description}</p>' if self.description else ''}
                </div>
                """, 
                unsafe_allow_html=True
            )

class MetricGrid:
    """Classe para criar grids de métricas"""
    
    def __init__(self, num_columns: int = 3):
        """
        Inicializa um grid de métricas.
        
        Args:
            num_columns (int): Número de colunas no grid
        """
        self.num_columns = num_columns
        self.metrics: List[MetricCard] = []

    def add_metric(self, metric: MetricCard):
        """
        Adiciona uma métrica ao grid.
        
        Args:
            metric (MetricCard): Objeto MetricCard para adicionar
        """
        self.metrics.append(metric)

    def render(self):
        """Renderiza o grid de métricas"""
        cols = st.columns(self.num_columns)
        for idx, metric in enumerate(self.metrics):
            with cols[idx % self.num_columns]:
                metric.render()

def create_model_metrics_summary(results: pd.DataFrame) -> MetricGrid:
    """
    Cria um resumo das métricas dos modelos.
    
    Args:
        results (pd.DataFrame): DataFrame com resultados dos modelos
        
    Returns:
        MetricGrid: Grid com métricas resumidas
    """
    grid = MetricGrid(num_columns=4)
    
    # Melhor acurácia
    best_acc = results['Acurácia'].max()
    best_acc_model = results.loc[results['Acurácia'].idxmax(), 'modelo']
    grid.add_metric(MetricCard(
        title="Melhor Acurácia",
        value=best_acc,
        description=f"Modelo: {best_acc_model}",
        color="#1f77b4"
    ))
    
    # Melhor F1-Score
    best_f1 = results['F1-Score'].max()
    best_f1_model = results.loc[results['F1-Score'].idxmax(), 'modelo']
    grid.add_metric(MetricCard(
        title="Melhor F1-Score",
        value=best_f1,
        description=f"Modelo: {best_f1_model}",
        color="#2ca02c"
    ))
    
    # Tempo médio de treinamento
    avg_time = results['tempo_treinamento'].mean()
    grid.add_metric(MetricCard(
        title="Tempo Médio de Treinamento",
        value=avg_time,
        suffix="s",
        color="#ff7f0e"
    ))
    
    # Melhor AUC-ROC
    best_auc = results['AUC-ROC'].max()
    best_auc_model = results.loc[results['AUC-ROC'].idxmax(), 'modelo']
    grid.add_metric(MetricCard(
        title="Melhor AUC-ROC",
        value=best_auc,
        description=f"Modelo: {best_auc_model}",
        color="#d62728"
    ))
    
    return grid

def create_data_quality_metrics(df: pd.DataFrame) -> MetricGrid:
    """
    Cria métricas de qualidade dos dados.
    
    Args:
        df (pd.DataFrame): DataFrame para análise
        
    Returns:
        MetricGrid: Grid com métricas de qualidade
    """
    grid = MetricGrid(num_columns=3)
    
    # Total de registros
    grid.add_metric(MetricCard(
        title="Total de Registros",
        value=len(df),
        color="#1f77b4"
    ))
    
    # Percentual de dados faltantes
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    grid.add_metric(MetricCard(
        title="Dados Faltantes",
        value=missing_pct,
        suffix="%",
        color="#ff7f0e"
    ))
    
    # Balanceamento das classes (se aplicável)
    if 'y' in df.columns:
        majority_class_pct = (df['y'].value_counts().max() / len(df)) * 100
        grid.add_metric(MetricCard(
            title="Classe Majoritária",
            value=majority_class_pct,
            suffix="%",
            color="#2ca02c"
        ))
    
    return grid

def create_performance_comparison(baseline_metrics: Dict[str, float], 
                                current_metrics: Dict[str, float]) -> MetricGrid:
    """
    Cria comparação de performance com baseline.
    
    Args:
        baseline_metrics (Dict[str, float]): Métricas do baseline
        current_metrics (Dict[str, float]): Métricas atuais
        
    Returns:
        MetricGrid: Grid com comparação de métricas
    """
    grid = MetricGrid(num_columns=3)
    
    for metric_name in ['Acurácia', 'F1-Score', 'AUC-ROC']:
        baseline_value = baseline_metrics.get(metric_name, 0)
        current_value = current_metrics.get(metric_name, 0)
        
        if baseline_value > 0:
            improvement = ((current_value - baseline_value) / baseline_value) * 100
            delta = f"{improvement:+.1f}%"
        else:
            delta = None
        
        grid.add_metric(MetricCard(
            title=metric_name,
            value=current_value,
            delta=delta,
            color="#1f77b4"
        ))
    
    return grid

def create_timestamp_info():
    """Cria informação de timestamp"""
    st.markdown(f"""
    <div style="
        font-size: 12px;
        color: #666;
        text-align: right;
        padding: 5px;
        margin-top: 20px;
    ">
        Última atualização: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

def create_error_metric(message: str):
    """
    Cria métrica de erro.
    
    Args:
        message (str): Mensagem de erro
    """
    st.markdown(f"""
    <div style="
        background-color: #ffe5e5;
        border-left: 4px solid #ff0000;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    ">
        <p style="color: #cc0000; margin: 0;">
            ⚠️ {message}
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_success_metric(message: str):
    """
    Cria métrica de sucesso.
    
    Args:
        message (str): Mensagem de sucesso
    """
    st.markdown(f"""
    <div style="
        background-color: #e5ffe5;
        border-left: 4px solid #00cc00;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    ">
        <p style="color: #006600; margin: 0;">
            ✅ {message}
        </p>
    </div>
    """, unsafe_allow_html=True)