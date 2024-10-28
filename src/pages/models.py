import streamlit as st
import pandas as pd
from src.components.metrics import MetricCard
from src.components.charts import MLMetricsCharts
from src.config.constants import MODELS_INFO
from src.config.styles import COLOR_PALETTES

def show_models():
    """Renderiza a página de modelos e suas configurações"""
    
    # Cabeçalho
    st.title("⚙️ Modelos de Machine Learning")
    
    # Introdução
    st.markdown("""
    Esta seção apresenta os detalhes dos modelos de ensemble utilizados no projeto.
    Cada modelo possui suas características específicas e diferentes configurações de hiperparâmetros.
    """)
    
    # Tabs para diferentes aspectos dos modelos
    tabs = st.tabs([
        "Visão Geral",
        "Configurações",
        "Hiperparâmetros",
        "Documentação"
    ])
    
    # Tab de Visão Geral
    with tabs[0]:
        st.markdown("### 📋 Modelos Implementados")
        
        for name, info in MODELS_INFO.items():
            with st.expander(name, expanded=True):
                st.markdown(f"""
                <div style="
                    border-left: 4px solid {info['color']};
                    padding-left: 20px;
                ">
                    <p>{info['desc']}</p>
                    <h4>Principais Características:</h4>
                    <ul>
                        {''.join(f'<li>{param}</li>' for param in info['params'])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Comparativo rápido
        st.markdown("### 🔄 Comparativo Rápido")
        
        comparison_data = {
            "Random Forest": {
                "Velocidade": "⭐⭐⭐",
                "Interpretabilidade": "⭐⭐⭐⭐",
                "Tunabilidade": "⭐⭐⭐",
                "Performance": "⭐⭐⭐⭐"
            },
            "XGBoost": {
                "Velocidade": "⭐⭐⭐⭐",
                "Interpretabilidade": "⭐⭐⭐",
                "Tunabilidade": "⭐⭐⭐⭐",
                "Performance": "⭐⭐⭐⭐⭐"
            },
            "LightGBM": {
                "Velocidade": "⭐⭐⭐⭐⭐",
                "Interpretabilidade": "⭐⭐⭐",
                "Tunabilidade": "⭐⭐⭐⭐",
                "Performance": "⭐⭐⭐⭐⭐"
            }
        }
        
        comparison_df = pd.DataFrame(comparison_data).T
        st.dataframe(comparison_df)
    
    # Tab de Configurações
    with tabs[1]:
        st.markdown("### ⚙️ Configurações dos Modelos")
        
        # Seleção do modelo
        selected_model = st.selectbox(
            "Selecione o modelo para ver as configurações",
            list(MODELS_INFO.keys())
        )
        
        if selected_model:
            # Parâmetros padrão
            st.markdown("#### Parâmetros Padrão")
            if selected_model == "Random Forest":
                st.code("""
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42
                )
                """)
            elif selected_model == "XGBoost":
                st.code("""
                XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.3,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    random_state=42
                )
                """)
            elif selected_model == "LightGBM":
                st.code("""
                LGBMClassifier(
                    n_estimators=100,
                    num_leaves=31,
                    max_depth=-1,
                    learning_rate=0.1,
                    random_state=42
                )
                """)
            
            # Recomendações de uso
            st.markdown("#### 💡 Recomendações de Uso")
            recommendations = {
                "Random Forest": [
                    "Ideal para datasets de tamanho médio",
                    "Quando a interpretabilidade é importante",
                    "Para casos com muitas features categóricas",
                    "Quando o overfitting é uma preocupação"
                ],
                "XGBoost": [
                    "Para datasets grandes",
                    "Quando a performance é crítica",
                    "Em competições de ML",
                    "Para casos com features numéricas"
                ],
                "LightGBM": [
                    "Para datasets muito grandes",
                    "Quando a velocidade é crucial",
                    "Em ambientes com recursos limitados",
                    "Para treinamento distribuído"
                ]
            }
            
            for rec in recommendations[selected_model]:
                st.markdown(f"- {rec}")
    
    # Tab de Hiperparâmetros
    with tabs[2]:
        st.markdown("### 🎛️ Otimização de Hiperparâmetros")
        
        # Seleção do modelo
        model_for_tuning = st.selectbox(
            "Selecione o modelo para ver os hiperparâmetros",
            list(MODELS_INFO.keys()),
            key="tuning"
        )
        
        # Grid de busca
        if model_for_tuning:
            st.markdown("#### 🔍 Grid de Busca")
            
            if model_for_tuning == "Random Forest":
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_for_tuning == "XGBoost":
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 1.0]
                }
            elif model_for_tuning == "LightGBM":
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'num_leaves': [31, 50, 100],
                    'max_depth': [-1, 5, 10],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            
            st.json(param_grid)
            
            # Dicas de tunagem
            st.markdown("#### 💡 Dicas de Tunagem")
            
            tuning_tips = {
                "Random Forest": [
                    "Comece com `n_estimators` mais alto",
                    "Ajuste `max_depth` para controlar overfitting",
                    "Balance `min_samples_split` e `min_samples_leaf`"
                ],
                "XGBoost": [
                    "Use `learning_rate` menor com mais `n_estimators`",
                    "Ajuste `max_depth` para controlar complexidade",
                    "Use `subsample` para reduzir overfitting"
                ],
                "LightGBM": [
                    "Prefira ajustar `num_leaves` a `max_depth`",
                    "Mantenha `learning_rate` pequeno inicialmente",
                    "Use `early_stopping` durante o treinamento"
                ]
            }
            
            for tip in tuning_tips[model_for_tuning]:
                st.markdown(f"- {tip}")
    
    # Tab de Documentação
    with tabs[3]:
        st.markdown("### 📚 Documentação")
        
        # Links para documentação
        st.markdown("""
        #### 🔗 Links Úteis
        
        **Random Forest**:
        - [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
        - [Random Forest User Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest)
        
        **XGBoost**:
        - [XGBoost Documentation](https://xgboost.readthedocs.io/)
        - [XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
        
        **LightGBM**:
        - [LightGBM Documentation](https://lightgbm.readthedocs.io/)
        - [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
        """)
        
        # Exemplo de código
        st.markdown("#### 💻 Exemplo de Implementação")
        
        st.code("""
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import RandomizedSearchCV

        # Definir modelo e grid de parâmetros
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Realizar busca randomizada
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=10,
            cv=5,
            random_state=42,
            n_jobs=-1
        )

        # Treinar modelo
        random_search.fit(X_train, y_train)

        # Obter melhores parâmetros
        best_params = random_search.best_params_
        """)
        
        # Notas importantes
        st.markdown("#### ⚠️ Notas Importantes")
        st.info("""
        - Sempre use validação cruzada para avaliar os modelos
        - Monitore o overfitting durante o treinamento
        - Considere o trade-off entre performance e tempo de treinamento
        - Mantenha um conjunto de teste separado para avaliação final
        """)

    # Footer com informações adicionais
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <small>
            Para mais informações sobre implementação e configuração dos modelos,<br>
            consulte a documentação oficial de cada biblioteca.
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_models()