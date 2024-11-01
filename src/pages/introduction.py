import streamlit as st
from src.components.metrics import MetricCard
from src.components.charts import ChartCard
from datetime import datetime

def show_introduction():
    """Renderiza a página de introdução do projeto"""
    
    # Cabeçalho
    st.title("🎯 Introdução ao Projeto")
    
    # Contexto do Projeto
    st.markdown("### 📋 Contexto do Projeto")
    st.markdown("""
    Este projeto foi desenvolvido como parte do desafio final do Ciclo 2 do curso de Python na AlphaEdtech. 
    O objetivo principal foi aplicar técnicas avançadas de machine learning em um conjunto de dados anônimo, 
    focando especialmente em modelos ensemble como Random Forest e Gradient Boosting.
    """)
    
    # Métricas do Projeto em Grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        MetricCard(
            title="Modelos Implementados",
            value=3,
            description="Random Forest, XGBoost, LightGBM",
            color="#1f77b4"
        ).render()
    
    with col2:
        MetricCard(
            title="Features Analisadas",
            value="100+",
            description="Características numéricas",
            color="#2ca02c"
        ).render()
    
    with col3:
        MetricCard(
            title="Técnicas Utilizadas",
            value="5+",
            description="Métodos de pré-processamento e análise",
            color="#ff7f0e"
        ).render()

    # Ferramentas e Tecnologias
    st.markdown("### ⚙️ Ferramentas e Tecnologias")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Principais Bibliotecas")
        st.markdown("""
        - **scikit-learn**: Implementação dos modelos base
        - **XGBoost**: Gradient boosting otimizado
        - **LightGBM**: Implementação eficiente de gradient boosting
        - **Pandas & NumPy**: Manipulação e análise de dados
        - **Plotly**: Visualização interativa de dados
        """)
    
    with col2:
        st.markdown("#### Ambiente de Desenvolvimento")
        st.markdown("""
        - **Python 3.x**: Linguagem principal
        - **GitHub**: Controle de versão
        - **VS Code**: IDE principal
        - **Google Colab**: Experimentação
        - **Streamlit**: Deploy do dashboard
        """)

    # Objetivos do Projeto
    st.markdown("### 🎯 Objetivos do Projeto")
    
    with st.expander("Ver objetivos detalhados", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Objetivos Principais")
            st.markdown("""
            1. Desenvolver modelos de classificação robustos
            2. Comparar diferentes estratégias de tratamento de dados
            3. Otimizar hiperparâmetros dos modelos
            4. Avaliar impacto de feature engineering
            """)
        
        with col2:
            st.markdown("#### Desafios Específicos")
            st.markdown("""
            1. Lidar com dados desbalanceados
            2. Tratar valores ausentes eficientemente
            3. Reduzir dimensionalidade mantendo performance
            4. Criar visualizações interpretáveis
            """)

    # Metodologia
    st.markdown("### 📊 Metodologia")
    
    methodology_col1, methodology_col2, methodology_col3 = st.columns(3)
    
    with methodology_col1:
        with st.container():
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <h3>🔍 Pré-processamento de Dados</h3>
                <ul>
                    <li>Análise exploratória detalhada</li>
                    <li>Tratamento de valores ausentes</li>
                    <li>Feature engineering e seleção</li>
                    <li>Balanceamento de classes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with methodology_col2:
        with st.container():
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <h3>🤖 Modelagem</h3>
                <ul>
                    <li>Implementação de múltiplos modelos ensemble</li>
                    <li>Otimização de hiperparâmetros</li>
                    <li>Validação cruzada</li>
                    <li>Análise de importância de features</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with methodology_col3:
        with st.container():
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <h3>📈 Avaliação</h3>
                <ul>
                    <li>Métricas de performance</li>
                    <li>Análise de resultados</li>
                    <li>Comparação entre modelos</li>
                    <li>Interpretação dos resultados</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Dataset
    st.markdown("### 💾 Dataset")
    st.info("""
    O conjunto de dados utilizado é anônimo e apresenta diversos desafios comuns em problemas reais 
    de machine learning, incluindo:
    - Múltiplas features numéricas
    - Presença de valores ausentes
    - Classes desbalanceadas
    - Necessidade de feature engineering
    """)

    # Navegação do Dashboard
    st.markdown("### 🧭 Navegação do Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Seções Principais")
        st.markdown("""
        - **Dashboard**: Visão geral dos resultados
        - **Análise dos Dados**: Exploração detalhada do dataset
        - **Modelos**: Configurações e comparações
        - **Resultados**: Métricas e avaliações
        """)
    
    with col2:
        st.markdown("#### Recursos Interativos")
        st.markdown("""
        - Filtros personalizáveis
        - Gráficos interativos
        - Comparações dinâmicas
        - Download de resultados
        """)

    # Timestamp
    st.markdown("""
    <div style='text-align: right; color: #666; padding: 20px;'>
        <small>
            Última atualização: {}
        </small>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
    unsafe_allow_html=True)

if __name__ == "__main__":
    show_introduction()