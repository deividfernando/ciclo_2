# Estilo CSS principal do aplicativo
CUSTOM_CSS = """
<style>
    /* Animações */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    /* Cards e Containers */
    .stcard {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .stcard:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        animation: slideIn 0.5s ease-out;
    }
    
    .chart-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 15px 0;
        animation: fadeIn 0.8s ease-out;
    }

    /* Textos e Métricas */
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .time-info {
        font-size: 12px;
        color: #666;
        text-align: right;
        padding: 5px;
    }

    /* Headers e Títulos */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
        animation: fadeIn 0.5s ease-out;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        font-size: 2rem;
        margin: 1.5rem 0;
    }
    
    h3 {
        font-size: 1.5rem;
        margin: 1rem 0;
    }

    /* Team Cards */
    .team-card {
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .team-card:hover {
        transform: translateY(-5px);
    }
    
    .team-card img {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        margin-bottom: 15px;
        border: 3px solid #f8f9fa;
    }
    
    .team-card h3 {
        margin: 10px 0;
        color: #2c3e50;
    }
    
    .team-card p {
        color: #666;
        margin: 5px 0;
    }

    /* Links e Botões */
    .social-links {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 15px;
    }
    
    .social-links a {
        text-decoration: none;
        transition: transform 0.3s ease;
    }
    
    .social-links a:hover {
        transform: translateY(-2px);
    }

    /* Dashboard Específicos */
    .dashboard-metric {
        border-left: 4px solid #1f77b4;
        padding-left: 15px;
    }
    
    .dashboard-chart {
        margin: 20px 0;
    }

    /* Responsividade */
    @media (max-width: 768px) {
        .metric-container {
            margin-bottom: 15px;
        }
        
        .chart-container {
            margin: 10px 0;
        }
        
        h1 {
            font-size: 2rem;
        }
        
        h2 {
            font-size: 1.75rem;
        }
        
        h3 {
            font-size: 1.25rem;
        }
    }

    /* Tabelas */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 0.9em;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
    }
    
    .styled-table thead tr {
        background-color: #1f77b4;
        color: white;
        text-align: left;
    }
    
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f8f9fa;
    }
    
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #1f77b4;
    }

    /* Loading States */
    .loading {
        text-align: center;
        padding: 20px;
        animation: pulse 1.5s infinite;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #2c3e50;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Streamlit Específicos */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #145d8d;
        transform: translateY(-2px);
    }
    
    .stSelectbox {
        color: #2c3e50;
    }
    
    .stTextInput>div>div>input {
        color: #2c3e50;
    }
    
    /* Download Buttons */
    .download-button {
        display: inline-block;
        padding: 10px 20px;
        background: linear-gradient(135deg, #1f77b4 0%, #145d8d 100%);
        color: white;
        border-radius: 5px;
        text-decoration: none;
        transition: all 0.3s ease;
        text-align: center;
        margin: 5px;
    }
    
    .download-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
</style>
"""

# Estilos específicos para gráficos Plotly
PLOTLY_TEMPLATE = {
    'layout': {
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {
            'color': '#2c3e50',
            'family': 'Arial, sans-serif'
        },
        'title': {
            'font': {
                'size': 24,
                'color': '#2c3e50'
            }
        },
        'xaxis': {
            'gridcolor': '#f8f9fa',
            'zerolinecolor': '#f8f9fa'
        },
        'yaxis': {
            'gridcolor': '#f8f9fa',
            'zerolinecolor': '#f8f9fa'
        }
    }
}

# Paletas de cores personalizadas
COLOR_PALETTES = {
    'main': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'blues': ['#08519c', '#3182bd', '#6baed6', '#bdd7e7', '#eff3ff'],
    'success': ['#28a745', '#34ce57', '#40d968', '#4ce97a', '#58f88b'],
    'warning': ['#ffc107', '#ffca2c', '#ffd34d', '#ffdc6f', '#ffe491'],
    'error': ['#dc3545', '#e4606d', '#eb8c95', '#f3b7bd', '#fae3e5']
}

# Configurações de tema para componentes específicos
COMPONENT_STYLES = {
    'metric_card': {
        'background': 'linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%)',
        'border_radius': '10px',
        'padding': '20px',
        'shadow': '0 4px 6px rgba(0, 0, 0, 0.05)'
    },
    'chart_card': {
        'background': 'white',
        'border_radius': '15px',
        'padding': '20px',
        'shadow': '0 2px 4px rgba(0, 0, 0, 0.05)'
    },
    'team_card': {
        'background': 'white',
        'border_radius': '15px',
        'padding': '20px',
        'shadow': '0 4px 6px rgba(0, 0, 0, 0.05)'
    }
}