<p align = "center">
  <img alt = "GitHub language count" src="https://img.shields.io/github/languages/count/deividfernando/ciclo_2?color=%2304D361">

  <img alt = "Repository size" src="https://img.shields.io/github/repo-size/deividfernando/ciclo_2">
  
  <img alt = "GitHub last commit" src="https://img.shields.io/github/last-commit/deividfernando/ciclo_2">
  
  <img alt = "Status" src="https://img.shields.io/static/v1?label=Status&message=Em Acabamento&color=FFFF00&style=flat"/>

  <img alt="GitHub Issues or Pull Requests" src="https://img.shields.io/github/issues/deividfernando/ciclo_2">

  <img alt="GitHub repo file or directory count" src="https://img.shields.io/github/directory-file-count/deividfernando/ciclo_2">

  <img alt = "Top language" src="https://img.shields.io/github/languages/top/deividfernando/ciclo_2?style=social">

  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/deividfernando/ciclo_2">

  <img alt="GitHub forks" src="https://img.shields.io/github/forks/deividfernando/ciclo_2">

  <img alt = "Stargazers" src="https://img.shields.io/github/stars/deividfernando/ciclo_2?style=social">

</p>

# Desafio de Fim de Ciclo 2 - AlphaEdtech

<p align="center">
 <a href="#-descrição-do-projeto">Descrição do projeto</a> •
 <a href="#-objetivos-principais">Objetivos</a> • 
 <a href="#%EF%B8%8F-tecnologias">Tecnologias</a> • 
 <a href="#-estrutura-do-projeto">Estrutura</a> •
 <a href="#-instruções-de-instalação-e-execução">Instalação</a> • 
 <a href="#-modelos-e-análises">Modelos</a> • 
 <a href="#-features-e-análises-avançadas">Features</a>
 <a href="#-pré-processamento-de-dados">Pré-processamento</a> •
 <a href="#-resultados-principais">Resultados</a> • 
 <a href="#-contribuições">Contribuições</a> • 
 <a href="#-licença">Licença</a>
 <a href="#autores">Autores</a>
</p>

## 📋 Descrição do Projeto
Este projeto faz parte do desafio final do ciclo 2 do curso de Python na AlphaEdtech. O objetivo é realizar análises de Machine Learning (ML) utilizando modelos baseados em ensembles (Random Forest e Gradient Boosting). O dataset fornecido é anônimo, e a missão é aplicar técnicas avançadas de treinamento e otimização de modelos para avaliar o desempenho de algoritmos de aprendizado supervisionado.

## 🎯 Objetivos Principais
1. Avaliar o impacto do número de árvores e profundidade no desempenho dos modelos
2. Comparar o tempo de treinamento entre **Random Forest** e **Gradient Boosting**
3. Analisar as variáveis mais importantes para o **Random Forest** e sua interpretação
4. Otimizar hiperparâmetros dos modelos usando técnicas como **Grid Search** ou **Random Search**
5. Desenvolver modelos de classificação robustos usando técnicas de ensemble learning
6. Comparar diferentes estratégias de tratamento de dados nulos
7. Avaliar o impacto da redução de dimensionalidade (PCA) no desempenho dos modelos

## 🛠️ Tecnologias
- **Linguagem:** Python 3.x
- **Principais Bibliotecas:** 
  - scikit-learn
  - xgboost
  - lightgbm
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - imblearn
- **IDEs e Ferramentas:**
  - Visual Studio Code
  - GitHub
  - Google Colab (experimentação e treinamento em nuvem)

## 📦 Estrutura do Projeto
```
📂 .devcontainer/
│   └── 📄 devcontainer.json
📂 data/
│   ├── 📄 train.parquet.encrypted
│   └── 📄 test.parquet.encrypted
📂 assets/
│   ├── 📄 confusion_matrix.png
│   └── 📄 roc_curve.png
📂 src/
│   ├── 📂 components/
│   │   ├── 📄 charts.py
│   │   └── 📄 metrics.py
│   ├── 📂 config/
│   │   ├── 📄 constants.py
│   │   ├── 📄 plotly_config.py
│   │   └── 📄 styles.py
│   ├── 📂 pages/
│   │   ├── 📄 dashboard.py
│   │   ├── 📄 data_analysis.py
│   │   ├── 📄 models.py
│   │   ├── 📄 result.py
│   │   ├── 📄 introduction.py
│   │   ├── 📄 best_model.py
│   │   └── 📄 team.py
│   ├── 📂 utils/
│       ├── 📄 data_loader.py
│       └── 📄 plotting.py
📂 utils/
│   └── 📄 functions.py
📄 analise_modelos.ipynb
📄 analise_resultados.ipynb
📄 app.py
📄 info_modelo.json
📄 modelo_final.joblib
📄 predicoes_finais.csv
📄 requirements.txt
📄 resultados_modelos.parquet
📄 treinamento.ipynb

```

## 🚀 Instruções de Instalação e Execução

1. Clone o repositório:
```bash
git clone https://github.com/deividfernando/ciclo_2.git
cd ciclo_2
```

2. Configure o ambiente virtual:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Execute os notebooks na ordem:
   - `treinamento.ipynb`
   - `analise_modelos.ipynb`
   - `analise_resultados.ipynb`

5. Visualize os resultados no dashboard interativo:
```bash
streamlit run app.py
```
O dashboard será aberto automaticamente no seu navegador padrão (geralmente em http://localhost:8501) e oferece:
   - 🏠 Visão geral do projeto e navegação intuitiva
   - 📊 Análise exploratória dos dados (distribuições, correlações e dados faltantes)
   - ⚙️ Interface para configuração e treinamento dos modelos
   - 📈 Visualização das métricas de performance e resultados
   - ℹ️ Informações sobre objetivos e ferramentas utilizadas

**Nota**: Certifique-se de que todos os notebooks foram executados antes de iniciar o dashboard, pois ele depende dos arquivos de resultados gerados durante o treinamento e análise dos modelos.

## 🔍 Modelos e Análises
### Modelos Implementados
- Random Forest
- XGBoost
- LightGBM

### Estratégias de Pré-processamento
- Média
- Mediana
- Moda
- Constante
- Análise personalizada por coluna

### Métricas de Desempenho
- Acurácia
- Precisão, Recall e F1-Score
- Matriz de Confusão
- Curva ROC e AUC

### Técnicas de Otimização
- Redução de dimensionalidade via PCA
- Balanceamento de classes com SMOTE
- RandomizedSearchCV para otimização de hiperparâmetros

## 📈 Features e Análises Avançadas
- Análise estatística completa de distribuições
- Múltiplas estratégias de tratamento de dados nulos
- Otimização avançada de hiperparâmetros
- Análise de importância de features
- Visualizações detalhadas de performance

## 🔧 Pré-processamento de Dados
A qualidade do pré-processamento foi essencial para o sucesso do projeto:
- Correta divisão dos dados
- Tratamento de outliers
- Imputação de dados faltantes
- Análise do impacto da normalização no desempenho
- Técnicas de redução de dimensionalidade

## 📊 Resultados Principais
- **Melhor Modelo**: LightGBM com estratégia de análise de colunas
  - Acurácia: 86.51%
  - Precisão Classe 0: 57.47%
  - Precisão Classe 1: 88.28%
  - Pontuação média: 77.42%

- **Desempenho no Conjunto de Teste Final**:
  - Acurácia: 80.23%
  - Precisão: 81.21%
  - Recall: 97.02%
  - F1-Score: 88.42%

## 🤝 Contribuições
Contribuições são bem-vindas! Para contribuir:
1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

## 📝 Licença
Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Autores

| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/105059460?v=4" width=115><br><sub>Cleverson Guandalin</sub>](https://github.com/CleverGnd) |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/55933599?v=4" width=115><br><sub>Deivid Fernando</sub>](https://github.com/deividfernando) |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/126590830?v=4" width=115><br><sub>Diego Alvarenga</sub>](https://github.com/diegoalvarengarodrigues) |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/15037530?v=4" width=115><br><sub>Fernando Moreno</sub>](https://github.com/F-moreno) | [<img loading="lazy" src="https://avatars.githubusercontent.com/u/144630236?s=400&u=7d7e40d80d8d466f5478a8ac9f390af04f909718&v=4" width=115><br><sub>Renan Pinto</sub>](https://github.com/RenanRCPinto) |  [<img loading="lazy" src="https://avatars.githubusercontent.com/u/84413595?v=4" width=115><br><sub>Yasmim Ferreira</sub>](https://github.com/ysmmfe)
| :---: | :---: | :---: | :---: | :---: | :---: |
