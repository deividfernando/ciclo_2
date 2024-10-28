import plotly.io as pio
from src.config.styles import PLOTLY_TEMPLATE, COLOR_PALETTES

def configure_plotly_theme():
    """
    Configura o tema padrão do Plotly para todo o aplicativo.
    Define cores, fontes e estilos consistentes para os gráficos.
    """
    # Criar template personalizado baseado no plotly white
    custom_template = pio.templates["plotly_white"].to_plotly_json()
    
    # Atualizar com configurações personalizadas
    custom_template.update({
        'layout': {
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'font': {
                'family': 'Arial, sans-serif',
                'color': '#2c3e50',
                'size': 12
            },
            'title': {
                'font': {
                    'family': 'Arial, sans-serif',
                    'color': '#2c3e50',
                    'size': 24
                },
                'x': 0.5,
                'xanchor': 'center'
            },
            'legend': {
                'bgcolor': 'rgba(255, 255, 255, 0.8)',
                'bordercolor': 'rgba(0, 0, 0, 0.1)',
                'borderwidth': 1,
                'font': {'size': 10}
            },
            'xaxis': {
                'gridcolor': '#f8f9fa',
                'zerolinecolor': '#f8f9fa',
                'title': {
                    'font': {
                        'family': 'Arial, sans-serif',
                        'size': 14,
                        'color': '#2c3e50'
                    }
                },
                'tickfont': {
                    'family': 'Arial, sans-serif',
                    'size': 10,
                    'color': '#2c3e50'
                }
            },
            'yaxis': {
                'gridcolor': '#f8f9fa',
                'zerolinecolor': '#f8f9fa',
                'title': {
                    'font': {
                        'family': 'Arial, sans-serif',
                        'size': 14,
                        'color': '#2c3e50'
                    }
                },
                'tickfont': {
                    'family': 'Arial, sans-serif',
                    'size': 10,
                    'color': '#2c3e50'
                }
            },
            'colorway': COLOR_PALETTES['main'],
            'hoverlabel': {
                'font': {
                    'family': 'Arial, sans-serif',
                    'size': 12
                },
                'bgcolor': 'white',
                'bordercolor': '#f8f9fa'
            },
            'margin': {
                'l': 50,
                'r': 50,
                't': 50,
                'b': 50
            },
            'updatemenus': [{
                'bgcolor': 'white',
                'bordercolor': '#f8f9fa',
                'font': {'size': 10}
            }],
            'sliders': [{
                'bgcolor': 'white',
                'bordercolor': '#f8f9fa',
                'font': {'size': 10}
            }]
        },
        'data': {
            'bar': [{
                'error_x': {'color': '#2c3e50'},
                'error_y': {'color': '#2c3e50'},
                'marker': {
                    'line': {'color': 'white', 'width': 0.5}
                },
                'opacity': 0.8
            }],
            'scatter': [{
                'marker': {
                    'line': {'color': 'white', 'width': 0.5}
                },
                'opacity': 0.8
            }],
            'heatmap': [{
                'colorscale': 'RdBu',
                'showscale': True
            }],
            'box': [{
                'fillcolor': 'white',
                'line': {'color': COLOR_PALETTES['main'][0]},
                'opacity': 0.8
            }]
        }
    })
    
    # Registrar o template personalizado
    pio.templates["custom"] = custom_template
    
    # Definir como template padrão
    pio.templates.default = "custom"

def get_figure_config():
    """
    Retorna configurações padrão para figuras Plotly.
    Inclui configurações de interatividade e responsividade.
    
    Returns:
        dict: Configurações para figuras Plotly
    """
    return {
        'displayModeBar': True,
        'responsive': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'pan2d',
            'lasso2d',
            'select2d',
            'autoScale2d',
            'hoverClosestCartesian',
            'hoverCompareCartesian'
        ],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'chart',
            'height': 500,
            'width': 700,
            'scale': 2
        }
    }

def get_layout_config(title: str = None, 
                     height: int = None,
                     showlegend: bool = True):
    """
    Retorna configurações de layout padrão para figuras Plotly.
    
    Args:
        title (str, optional): Título do gráfico
        height (int, optional): Altura do gráfico
        showlegend (bool): Se deve mostrar legenda
        
    Returns:
        dict: Configurações de layout
    """
    layout = {
        'showlegend': showlegend,
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'margin': {'t': 50, 'r': 50, 'l': 50, 'b': 50}
    }
    
    if title:
        layout['title'] = {
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {
                'size': 24,
                'color': '#2c3e50'
            }
        }
    
    if height:
        layout['height'] = height
    
    return layout