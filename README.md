# Desafio de Fim de Ciclo 2 - AlphaEdtech

## 📋 Descrição do Projeto
Este projeto faz parte do desafio final do ciclo 2 do curso de Python na AlphaEdtech. O objetivo é realizar análises de Machine Learning (ML) utilizando modelos baseados em ensembles. O dataset fornecido é anônimo, e a missão é aplicar técnicas avançadas de treinamento e otimização de modelos para avaliar o desempenho de algoritmos de aprendizado supervisionado.

## 🎯 Objetivos Principais
1. Avaliar o impacto do número de árvores e profundidade no desempenho dos modelos
2. Comparar o tempo de treinamento entre **Random Forest** e **Gradient Boosting**
3. Analisar as variáveis mais importantes para o **Random Forest** e sua interpretação
4. Otimizar hiperparâmetros dos modelos usando técnicas como **Grid Search** ou **Random Search**
5. Desenvolver modelos de classificação robustos usando técnicas de ensemble learning
6. Comparar diferentes estratégias de tratamento de dados nulos
7. Avaliar o impacto da redução de dimensionalidade (PCA) no desempenho dos modelos

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

## 💡 Justificativa de Escolha dos Modelos
Cada modelo foi escolhido com base nas características específicas do problema:
- **Random Forest**: Escolhido por sua robustez contra overfitting e boa performance em dados de alta dimensionalidade
- **Gradient Boosting (XGBoost/LightGBM)**: Selecionado por sua alta performance em competições de ML e capacidade de lidar com dados complexos

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

## 🔧 Pré-processamento de Dados
A qualidade do pré-processamento foi essencial para o sucesso do projeto:
- Correta divisão dos dados
- Tratamento de outliers
- Imputação de dados faltantes
- Análise do impacto da normalização no desempenho
- Técnicas de redução de dimensionalidade

## 🛠️ Ambiente de Desenvolvimento
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
├── data/
│   ├── train.parquet
│   └── test.parquet
├── utils/
│   └── functions.py
├── notebooks/
│   ├── treinamento.ipynb
│   ├── analise_modelos.ipynb
│   └── analise_resultados.ipynb
├── README.md
└── requirements.txt
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

## 📈 Features e Análises Avançadas
- Análise estatística completa de distribuições
- Múltiplas estratégias de tratamento de dados nulos
- Otimização avançada de hiperparâmetros
- Análise de importância de features
- Visualizações detalhadas de performance

## 🤝 Contribuições
Contribuições são bem-vindas! Para contribuir:
1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

## 📝 Licença
Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
