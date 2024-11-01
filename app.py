import streamlit as st
from streamlit_option_menu import option_menu
import plotly.io as pio

# Importações das páginas
from src.pages.introduction import show_introduction
from src.pages.dashboard import show_dashboard
from src.pages.data_analysis import show_analysis
from src.pages.best_model import show_best_model
from src.pages.models import show_models
from src.pages.results import show_results
from src.pages.team import show_team

# Importações de utilidades
from src.utils.data_loader import load_data, load_results
from src.config.styles import CUSTOM_CSS
from src.config.plotly_config import configure_plotly_theme

def initialize_app():
    """Inicializa as configurações básicas do aplicativo"""
    st.set_page_config(
        page_title="ML Ensemble Analysis Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Configurar tema do Plotly
    configure_plotly_theme()
    
    # Injetar CSS customizado
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def create_sidebar():
    """Cria e retorna a seleção do menu lateral"""
    with st.sidebar:
        return option_menu(
            menu_title="Menu",
            options=["Introdução", "Dashboard", "Modelos", "Análise dos Dados", "Melhor Modelo", "Resultados", "Nosso Time"],
            icons=["house", "speedometer2", "gear", "bar-chart", "trophy", "graph-up", "people"],
            menu_icon="cast",
            default_index=0,
            styles={
                "nav-link-selected": {"background-color": "#1f77b4"}
            }
        )

def main():
    """Função principal do aplicativo"""
    # Inicializar configurações
    initialize_app()
    
    # Criar menu lateral
    selected_page = create_sidebar()
    
    # Carregar dados
    df = load_data()
    results = load_results()
    
    # Verificar se os dados foram carregados corretamente
    if df is None or results is None:
        st.error("Erro ao carregar os dados. Por favor, verifique os arquivos de dados.")
        return
    
    # Renderizar página selecionada
    try:
        if selected_page == "Introdução":
            show_introduction()
        elif selected_page == "Dashboard":
            show_dashboard(df, results)
        elif selected_page == "Melhor Modelo":
            show_best_model()
        elif selected_page == "Análise dos Dados":
            show_analysis(df)
        elif selected_page == "Modelos":
            show_models()
        elif selected_page == "Resultados":
            show_results(results)
        elif selected_page == "Nosso Time":
            show_team()
    except Exception as e:
        st.error(f"Erro ao renderizar a página {selected_page}: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
