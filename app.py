import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Configuração da página
st.set_page_config(
    page_title="ML Ensemble Analysis Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo CSS personalizado
st.markdown(
    """
    <style>
        .main {
            padding: 0rem 1rem;
        }
        .stAlert {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        h1 {
            color: #1E88E5;
        }
        h2 {
            color: #424242;
        }
        .stProgress .st-bo {
            background-color: #1E88E5;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# Função para carregar dados
@st.cache_data
def load_data():
    # Substitua isso pelo seu próprio carregamento de dados
    return pd.read_csv("predicoes_finais.csv")


# Sidebar com navegação
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Home",
            "ML Theory",
            "Data Analysis",
            "Model Training",
            "Results",
            "About",
        ],
        icons=["house", "book", "bar-chart", "gear", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Página inicial
if selected == "Home":
    st.title("🤖 Machine Learning Ensemble Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        ## Sobre o Projeto
        Este dashboard apresenta uma análise completa do desafio de Machine Learning utilizando
        modelos ensemble. Aqui você encontrará:
        
        - 📊 Análise exploratória dos dados
        - 🔍 Comparação entre diferentes modelos
        - 📈 Métricas de performance
        - 🎯 Resultados e insights
        """
        )

    with col2:
        st.info(
            "👈 Use o menu lateral para navegar entre as diferentes seções do dashboard"
        )

# Teoria do ML
elif selected == "ML Theory":
    st.title("📚 Fundamentos Teóricos dos Modelos Ensemble")

    st.markdown(
        """
    ## 🔄 Métodos Ensemble
    Os métodos ensemble são técnicas avançadas de machine learning que combinam múltiplos modelos base 
    para produzir um modelo preditivo aprimorado. A principal vantagem dessa abordagem é a redução da 
    variância e do viés, resultando em previsões mais robustas e confiáveis.

    ### 🌳 Random Forest
    O Random Forest é um algoritmo de ensemble learning baseado em árvores de decisão que opera através 
    do princípio de bagging (Bootstrap Aggregating). Características principais:
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Funcionamento do Random Forest**\n\n"
            "- Criação de múltiplas árvores de decisão\n"
            "- Amostragem aleatória com reposição\n"
            "- Seleção aleatória de features\n"
            "- Votação majoritária para classificação"
        )

    with col2:
        st.info(
            "**Vantagens**\n\n"
            "- Redução do overfitting\n"
            "- Robusto a outliers\n"
            "- Paralelizável\n"
            "- Importância das features"
        )

    st.markdown(
        """
    ### 📈 Gradient Boosting
    O Gradient Boosting é uma técnica de ensemble que constrói modelos de forma sequencial, 
    onde cada novo modelo tenta corrigir os erros dos modelos anteriores. As implementações 
    modernas incluem:
    """
    )

    col3, col4 = st.columns(2)

    with col3:
        st.warning(
            "**XGBoost**\n\n"
            "- Implementação otimizada\n"
            "- Regularização integrada\n"
            "- Tratamento eficiente de dados esparsos\n"
            "- Alta performance"
        )

    with col4:
        st.warning(
            "**LightGBM**\n\n"
            "- Crescimento de árvore baseado em folhas\n"
            "- Menor uso de memória\n"
            "- Treinamento mais rápido\n"
            "- Suporte a dados categóricos"
        )

    # Adicionar comparação técnica
    st.subheader("🔍 Comparação Técnica dos Modelos")

    comparison_df = pd.DataFrame(
        {
            "Característica": [
                "Método Base",
                "Construção",
                "Complexidade",
                "Paralelização",
                "Uso de Memória",
            ],
            "Random Forest": [
                "Bagging",
                "Paralela",
                "O(n_trees * n_samples * log(n_samples))",
                "Alta",
                "Moderado",
            ],
            "Gradient Boosting": [
                "Boosting",
                "Sequencial",
                "O(n_trees * n_samples * log(n_samples))",
                "Limitada",
                "Baixo",
            ],
        }
    )

    st.table(comparison_df)

    # Nota técnica final
    st.info(
        "📝 **Observação Técnica**\n\n"
        "A escolha entre Random Forest e Gradient Boosting frequentemente envolve um "
        "trade-off entre velocidade de treinamento, capacidade de paralelização e "
        "performance do modelo. Random Forest tende a ser mais robusto e fácil de "
        "ajustar, enquanto Gradient Boosting geralmente oferece melhor performance "
        "com ajuste adequado dos hiperparâmetros."
    )

# Página de análise de dados
elif selected == "Data Analysis":
    st.title("📊 Análise Exploratória dos Dados")

    tab1, tab2, tab3 = st.tabs(["Distribuição", "Correlações", "Missing Values"])

    with tab1:
        st.header("Distribuição das Features")
        # Adicione seus gráficos de distribuição aqui

    with tab2:
        st.header("Matriz de Correlação")
        # Adicione seu heatmap de correlação aqui

    with tab3:
        st.header("Análise de Dados Faltantes")
        # Adicione sua análise de dados faltantes aqui

# Página de treinamento
elif selected == "Model Training":
    st.title("⚙️ Treinamento dos Modelos")

    st.header("Configuração dos Modelos")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest")
        rf_estimators = st.slider("Número de Estimadores (RF)", 100, 500, 200)
        rf_depth = st.slider("Profundidade Máxima (RF)", 10, 100, 30)

    with col2:
        st.subheader("Gradient Boosting")
        gb_estimators = st.slider("Número de Estimadores (GB)", 100, 500, 200)
        gb_depth = st.slider("Profundidade Máxima (GB)", 3, 10, 5)

    if st.button("Iniciar Treinamento"):
        with st.spinner("Treinando modelos..."):
            # Adicione seu código de treinamento aqui
            st.success("Treinamento concluído!")

# Página de resultados
elif selected == "Results":
    st.title("📈 Resultados e Comparações")

    tab1, tab2 = st.tabs(["Métricas", "Importância das Features"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Acurácia RF", "0.85", "+0.03")
        with col2:
            st.metric("Acurácia XGBoost", "0.87", "+0.05")
        with col3:
            st.metric("Acurácia LightGBM", "0.86", "+0.04")

        # Adicione mais visualizações de métricas aqui

    with tab2:
        st.header("Importância das Features")
        # Adicione seu gráfico de importância das features aqui

# Página sobre
else:
    st.title("ℹ️ Sobre")

    st.markdown(
        """
    ## Desafio de Fim de Ciclo 2 - AlphaEdtech
    
    Este projeto faz parte do desafio final do ciclo 2 do curso de Python na AlphaEdtech, 
    onde o objetivo é realizar análises de Machine Learning utilizando modelos baseados em ensembles.
    
    ### 🎯 Objetivos
    1. Avaliar o impacto do número de árvores e profundidade no desempenho dos modelos
    2. Comparar o tempo de treinamento entre Random Forest e Gradient Boosting
    3. Analisar as variáveis mais importantes para o Random Forest
    4. Otimizar hiperparâmetros dos modelos
    
    ### 🛠️ Ferramentas Utilizadas
    - Python
    - Scikit-learn
    - XGBoost
    - LightGBM
    - Streamlit
    - Plotly
    """
    )

    st.balloons()


def main():
    # Seu código principal aqui
    pass


if __name__ == "__main__":
    main()
