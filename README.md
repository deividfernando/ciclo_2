# Desafio de Fim de Ciclo 2 - AlphaEdtech

Este projeto faz parte do desafio final do ciclo 2 do curso de Python na AlphaEdtech, onde o objetivo é realizar análises de Machine Learning (ML) utilizando modelos baseados em ensembles. O dataset fornecido é anônimo, e a missão é aplicar técnicas avançadas de treinamento e otimização de modelos para avaliar o desempenho de algoritmos de aprendizado supervisionado.

## 📚 Sobre o Projeto

O desafio consiste em treinar e comparar dois modelos de ML baseados em árvores de decisão: **Random Forest** e **Gradient Boosting** (utilizando XGBoost ou LightGBM). O foco é entender como a combinação de várias árvores de decisão (ensembles) pode melhorar a performance dos modelos e reduzir o overfitting.

## 🎯 Objetivos

1. Avaliar o impacto do número de árvores e profundidade no desempenho dos modelos.
2. Comparar o tempo de treinamento entre **Random Forest** e **Gradient Boosting**.
3. Analisar as variáveis mais importantes para o **Random Forest** e sua interpretação.
4. Otimizar hiperparâmetros dos modelos usando técnicas como **Grid Search** ou **Random Search**.

## 🔍 Análises e Métricas

Durante o treinamento e avaliação dos modelos, monitoramos as seguintes métricas de desempenho:

- **Acurácia**
- **Precisão**, **Recall** e **F1-Score**
- **Matriz de Confusão**
- **Curva ROC e AUC**

### Interpretação dos Modelos

Analisamos as variáveis mais importantes que impactam cada modelo, verificamos o overfitting através de **cross-validation** e discutimos como as escolhas influenciam o problema abordado.

### Ajuste de Hiperparâmetros

Para otimizar o desempenho, aplicamos técnicas de ajuste de hiperparâmetros, como:

- **Grid Search**
- **Random Search**

## 🔧 Pré-processamento de Dados

A qualidade do pré-processamento foi essencial para o sucesso do projeto. A etapa incluiu:

- Correta **divisão dos dados**
- **Tratamento de outliers**
- **Imputação** de dados faltantes
- Análise de como a **normalização** afeta o desempenho dos modelos

## 💡 Justificativa de Escolha dos Modelos

Cada modelo foi escolhido com base nas características do problema abordado e sua capacidade de lidar com dados complexos, como o **Random Forest**, que é robusto contra overfitting, e o **Gradient Boosting**, conhecido por sua alta performance em competições de ML.

## 📊 Resultados

Os resultados dos modelos foram avaliados utilizando métricas como **AUC-ROC**, **F1-Score**, entre outras, e as variáveis mais importantes foram identificadas e interpretadas para justificar as decisões tomadas durante o processo.

## 🛠 Ferramentas Utilizadas

- **Linguagem:** Python
- **Bibliotecas:** Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn
- **Ambiente de Desenvolvimento:** Visual Studio Code, GitHub
- **Outras Ferramentas:** Google Colab para experimentação e treinamento em nuvem

## 🚀 Instruções de Instalação e Execução

Siga os passos abaixo para configurar e executar o projeto localmente:


**1. Clone o repositório:**
```bash
  git clone https://github.com/deividfernando/ciclo_2.git

```

**2. Navegue até o diretório do projeto:**
```bash
   cd ciclo_2
```

**3. Crie um ambiente virtual (opcional, mas recomendado):**
```bash
   - No Windows:
     python -m venv venv
     venv\Scripts\activate
   - No macOS/Linux:
     python3 -m venv venv
     source venv/bin/activate
```
**4. Instale as dependências:**
```bash
   pip install -r requirements.txt
```

