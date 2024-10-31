import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
from src.config.constants import TEAM_MEMBERS, LOTTIE_URLS
from src.utils.data_loader import load_lottie_url

def create_team_card(member: dict):
    """
    Cria um card para um membro da equipe.
    
    Args:
        member (dict): Dicionário com informações do membro
    """
    st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <img src="https://github.com/{member['github_username']}.png" 
             style="width: 150px; height: 150px; border-radius: 50%; margin-bottom: 15px;">
        <h3 style="margin-bottom: 10px;">{member['name']}</h3>
        <div style="display: flex; justify-content: center; gap: 10px;">
            <a href="{member['github_url']}" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
            </a>
            <a href="{member['linkedin_url']}" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white">
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_team():
    """Renderiza a página da equipe"""
    
    # Cabeçalho
    st.title("👥 Nossa Equipe")
    
    # Introdução
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <p>
            Conheça os profissionais responsáveis pelo desenvolvimento deste projeto:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Grid de membros da equipe
    for i in range(0, len(TEAM_MEMBERS), 3):
        cols = st.columns(3)
        row_members = TEAM_MEMBERS[i:i+3]
        
        for col, member in zip(cols, row_members):
            with col:
                create_team_card(member)
    
    # Seção de Agradecimentos
    st.markdown("### 🙏 Agradecimentos")
    
    st.markdown("""
    <div style="
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    ">
        <p>
            Agradecemos à <strong>AlphaEdtech</strong> pela oportunidade de aprendizado
            e desenvolvimento proporcionada durante o curso. Este projeto representa
            nossa jornada de aprendizado em Machine Learning e o poder do trabalho em equipe.
        </p>
        <p>
            Também agradecemos a todos os instrutores e mentores que nos guiaram
            durante esta jornada.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        border-top: 1px solid #eee;
    ">
        <p style="color: #666;">
            Desenvolvido com ❤️ pela equipe<br>
            {data}
        </p>
    </div>
    """.format(data=pd.Timestamp.now().strftime("%Y")), unsafe_allow_html=True)
    
    # Easter egg
    if st.button("🎉", help="Um pequeno presente"):
        st.balloons()
        st.markdown("""
        <div style="text-align: center;">
            <h3>Obrigado por conhecer nossa equipe! 🚀</h3>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_team()