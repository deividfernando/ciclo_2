import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def show_best_model():
    """
    Renderiza a página do melhor modelo.
    """
    # Cabeçalho
    st.title("🏆 Melhor Modelo - LightGBM")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Acurácia",
            value="86.51%",
            delta="Melhor performance"
        )
    
    with col2:
        st.metric(
            label="F1-Score",
            value="88%",
            delta="Classe majoritária"
        )
    
    with col3:
        st.metric(
            label="Precisão Média",
            value="74%",
            delta="Macro avg"
        )
    
    with col4:
        st.metric(
            label="AUC-ROC",
            value="79%",
            delta="Boa separação"
        )

    # Detalhes do modelo
    st.markdown("""
    ### 📊 Detalhes do Modelo
    
    **Modelo**: LightGBM com estratégia de análise de colunas
    
    **Pontuação média**: 0.7742
    
    **Métricas por classe**:
    - Classe 0 (Minoritária):
        - Precisão: 57.47%
        - Recall: 21%
    - Classe 1 (Majoritária):
        - Precisão: 88.28%
        - Recall: 97%
    """)

    # Matriz de Confusão e Curva ROC
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Matrix de Confusão")
        # Carregar e exibir a imagem da curva ROC
        confusion_matrix = Image.open('assets/confusion_matrix.png')
        st.image(confusion_matrix, caption='Matrix de Confusão')

    with col2:
        st.markdown("### Curva ROC")
        # Carregar e exibir a imagem da curva ROC
        roc_curve = Image.open('assets/roc_curve.png')
        st.image(roc_curve, caption='Curva ROC - AUC: 0.79')

        # Feature Importance
    st.markdown("""
    ### 🎯 Features Mais Importantes
    
    As 6 features que mais impactaram nas decisões do modelo:
    
    1. **Feature 0**: Principal indicador com maior peso nas decisões
    2. **Feature 22**: Segundo indicador mais relevante
    3. **Feature 19**: Terceiro fator mais importante
    4. **Feature 8**: Quarto indicador em relevância
    5. **Feature 34**: Quinta feature mais impactante
    6. **Feature 49**: Sexta feature em importância
    
    > Estas features foram identificadas através da análise SHAP (SHapley Additive exPlanations), que mede a contribuição de cada variável para as previsões do modelo.
    """)

    # Visualização das Features
    feature_importance = pd.DataFrame({
        'Feature': ['0', '22', '19', '8', '34', '49'],
        'Importância': [100, 85, 78, 72, 65, 58]  # Valores normalizados para visualização
    })

    # Criar gráfico de barras horizontal
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(feature_importance['Feature'], feature_importance['Importância'], 
                  color='skyblue')
    
    # Personalizar o gráfico
    ax.set_xlabel('Importância Relativa (%)')
    ax.set_title('Top 6 Features Mais Importantes')
    
    # Adicionar valores nas barras
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}%', 
                ha='left', va='center', fontweight='bold')
    
    st.pyplot(fig)

    # Resultados gerais
    st.markdown("""
    ### 📈 Resultados Gerais
    
    **Total de amostras**: 10,746
    - Acertos: 8,621 (80.23%)
    - Erros: 2,125 (19.77%)
    
    ### 💡 Insights Principais
    
    1. **Performance Geral**:
       - O modelo alcançou uma acurácia de 86.51%
       - AUC-ROC de 0.79 indica boa capacidade de discriminação
    
    2. **Desbalanceamento de Classes**:
       - Melhor performance na classe majoritária (1)
       - Oportunidade de melhoria na classe minoritária (0)
    
    3. **Trade-offs**:
       - Alto recall na classe majoritária (97%)
       - Precisão moderada na classe minoritária (57.47%)
    
    ### 🔍 Recomendações
    
    1. **Otimização**:
       - Investigar técnicas adicionais de balanceamento
       - Ajuste fino de hiperparâmetros focando na classe minoritária
    
    2. **Monitoramento**:
       - Acompanhar especialmente os falsos negativos
       - Validar performance em diferentes cenários
    
    3. **Próximos Passos**:
       - Avaliar custo-benefício entre precisão e recall
       - Considerar ensemble com outros modelos
    """)

    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <small>
            Análise realizada pela equipe AlphaEdTech<br>
            Última atualização: Outubro 2024
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_best_model()