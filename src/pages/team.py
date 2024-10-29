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
        <p style="color: #666; margin-bottom: 15px;">{member['role']}</p>
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
    
    # Animação Lottie
    team_animation = load_lottie_url(LOTTIE_URLS["team"])
    if team_animation:
        st_lottie(team_animation, height=200)
    
    # Introdução
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <p>
            Conheça os profissionais responsáveis pelo desenvolvimento deste projeto.
            Nossa equipe multidisciplinar combina experiência em Data Science, 
            Engenharia de Dados e Análise de Dados.
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
    
    # Seção de Skills
    st.markdown("### 🛠️ Nossas Habilidades")
    
    skills = {
        "Data Science": [
            "Machine Learning",
            "Deep Learning",
            "Statistical Analysis",
            "Model Optimization"
        ],
        "Data Engineering": [
            "ETL Pipelines",
            "Data Warehousing",
            "Big Data Technologies",
            "Cloud Computing"
        ],
        "Data Analysis": [
            "Data Visualization",
            "Business Intelligence",
            "Exploratory Analysis",
            "Dashboard Development"
        ]
    }
    
    cols = st.columns(len(skills))
    
    for col, (area, skill_list) in zip(cols, skills.items()):
        with col:
            st.markdown(f"""
            <div style="
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            ">
                <h4>{area}</h4>
                <ul style="list-style-type: none; padding: 0;">
                    {''.join(f'<li style="margin: 10px 0;">{skill}</li>' for skill in skill_list)}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Seção de Contribuições
    st.markdown("### 🎯 Contribuições no Projeto")
    
    contributions = {
        "Cleverson Guandalin": "Desenvolvimento do pipeline de dados e implementação do XGBoost",
        "Deivid Fernando": "Arquitetura do projeto e desenvolvimento do dashboard",
        "Diego Alvarenga": "Análise exploratória e feature engineering",
        "Fernando Moreno": "Implementação do Random Forest e otimização de hiperparâmetros",
        "Renan Pinto": "Análise estatística e validação de modelos",
        "Yasmim Ferreira": "Visualização de dados e documentação"
    }
    
    for name, contribution in contributions.items():
        st.markdown(f"""
        <div style="
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #1f77b4;
        ">
            <strong>{name}</strong>
            <p style="margin: 5px 0 0 0;">{contribution}</p>
        </div>
        """, unsafe_allow_html=True)
    
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