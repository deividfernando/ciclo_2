# URLs das animações Lottie
LOTTIE_URLS = {
    "rocket": "https://assets8.lottiefiles.com/packages/lf20_zw0djhar.json",
    "data": "https://assets9.lottiefiles.com/packages/lf20_qp1q7mct.json",
    "machine_learning": "https://assets6.lottiefiles.com/packages/lf20_xqbbxz1x.json",
    "results": "https://assets5.lottiefiles.com/packages/lf20_ln8zdxqj.json",
    "team": "https://assets6.lottiefiles.com/packages/lf20_syqnfe7c.json"
}

# Informações dos membros da equipe
TEAM_MEMBERS = [
    {
        "name": "Cleverson Guandalin",
        "github_username": "CleverGnd", 
        "role": "Data Engineer",
        "github_url": "https://github.com/CleverGnd", 
        "linkedin_url": "https://www.linkedin.com/in/cleversonguandalin/"
    },
    {
        "name": "Deivid Fernando",
        "github_username": "deividfernando",
        "role": "Data Engineer",
        "github_url": "https://github.com/deividfernando",
        "linkedin_url": "https://www.linkedin.com/in/deividfsilva/"
    },
    {
        "name": "Diego Alvarenga",
        "github_username": "diegoalvarengarodrigues",
        "role": "Data Scientist",
        "github_url": "https://github.com/diegoalvarengarodrigues",
        "linkedin_url": "https://www.linkedin.com/in/diego-alvarenga-rodrigues-02b9b31a4/"
    },
    {
        "name": "Fernando Moreno",
        "github_username": "F-moreno",
        "role": "Data Engineer",
        "github_url": "https://github.com/F-moreno",
        "linkedin_url": "https://www.linkedin.com/in/moreno-fernando/"
    },
    {
        "name": "Renam R. C. Pinto",
        "github_username": "RenanRCPinto", 
        "role": "Data Scientist",
        "github_url": "https://github.com/RenanRCPinto", 
        "linkedin_url": "https://www.linkedin.com/in/renan-pinto-96b06b156/"
    },
    {
        "name": "Yasmim Ferreira",
        "github_username": "ysmmfe",  
        "role": "Data Analyst",
        "github_url": "https://github.com/ysmmfe",  
        "linkedin_url": "https://www.linkedin.com/in/ysmmfe/" 
    }
]

# Configurações dos modelos
MODELS_INFO = {
    "Random Forest": {
        "desc": "Ensemble de árvores de decisão com amostragem aleatória",
        "params": ["n_estimators", "max_depth", "min_samples_split"],
        "color": "#1f77b4"
    },
    "XGBoost": {
        "desc": "Gradient boosting otimizado com regularização",
        "params": ["n_estimators", "max_depth", "learning_rate"],
        "color": "#ff7f0e"
    },
    "LightGBM": {
        "desc": "Implementação rápida de gradient boosting",
        "params": ["n_estimators", "num_leaves", "learning_rate"],
        "color": "#2ca02c"
    }
}

# Configurações dos gráficos
CHART_CONFIG = {
    "default_height": 500,
    "default_width": None,  # None para usar largura do container
    "color_palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "background_color": "rgba(0, 0, 0, 0)",
    "font_color": "#2c3e50"
}

# Métricas padrão para avaliação
DEFAULT_METRICS = [
    "Acurácia",
    "Precisão",
    "Recall",
    "F1-Score",
    "AUC-ROC"
]

# Configurações de arquivo
FILE_PATHS = {
    "train_data": "data/train.parquet",
    "test_data": "data/test.parquet",
    "results": "resultados_modelos.parquet"
}

# Configurações da página
PAGE_CONFIG = {
    "title": "ML Ensemble Analysis Dashboard",
    "icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Mensagens de erro
ERROR_MESSAGES = {
    "data_load": "Erro ao carregar os dados. Por favor, verifique os arquivos de dados.",
    "model_load": "Erro ao carregar os modelos. Por favor, verifique a instalação das dependências.",
    "visualization": "Erro ao gerar visualização. Por favor, tente novamente.",
    "page_render": "Erro ao renderizar a página. Por favor, atualize a página."
}

# Configurações de cache
CACHE_CONFIG = {
    "ttl": 3600,  # Tempo de vida do cache em segundos
    "max_entries": 100  # Número máximo de entradas no cache
}