import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from scipy import stats
from scipy import stats
from scipy.stats import shapiro, anderson


def analisar_e_tratar_nulos(df, limite_nulos_linha=0.5, alpha=0.05):
    df_limpo = df.dropna(thresh=int(df.shape[1] * (1 - limite_nulos_linha)))
    
    tratamentos = {}
    
    for coluna in df_limpo.columns:
        if df_limpo[coluna].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_limpo[coluna]):
                dados_nao_nulos = df_limpo[coluna].dropna()
                
                # Testes de normalidade
                _, p_valor_shapiro = shapiro(dados_nao_nulos)
                resultado_anderson = anderson(dados_nao_nulos)
                estatistica_anderson = resultado_anderson.statistic
                valor_critico_anderson = resultado_anderson.critical_values[2]
                
                normal_shapiro = p_valor_shapiro > alpha
                normal_anderson = estatistica_anderson < valor_critico_anderson
                
                print(f"Coluna: {coluna}")
                print(f"Shapiro p-valor: {p_valor_shapiro}")
                print(f"Anderson estatística: {estatistica_anderson}, valor crítico: {valor_critico_anderson}")
                
                # Verificar assimetria e curtose
                skewness = stats.skew(dados_nao_nulos)
                kurtosis = stats.kurtosis(dados_nao_nulos)
                
                if normal_shapiro or normal_anderson or (abs(skewness) < 1 and abs(kurtosis) < 3):
                    tratamento = 'media'
                    valor = df_limpo[coluna].mean()
                elif dados_nao_nulos.mode().iloc[0] / dados_nao_nulos.count() > 0.3:
                    tratamento = 'moda'
                    valor = dados_nao_nulos.mode().iloc[0]
                else:
                    # Tentar transformações
                    transformacoes = [
                        ('original', dados_nao_nulos),
                        ('log', np.log1p(dados_nao_nulos - dados_nao_nulos.min() + 1)),
                        ('sqrt', np.sqrt(dados_nao_nulos - dados_nao_nulos.min())),
                        ('box-cox', stats.boxcox(dados_nao_nulos - dados_nao_nulos.min() + 1)[0])
                    ]
                    
                    melhor_transformacao = max(transformacoes, key=lambda x: shapiro(x[1])[1])
                    
                    if shapiro(melhor_transformacao[1])[1] > alpha:
                        tratamento = f'media_{melhor_transformacao[0]}'
                        valor = np.mean(melhor_transformacao[1])
                        if melhor_transformacao[0] != 'original':
                            valor = np.exp(valor) - 1 + dados_nao_nulos.min() - 1 if melhor_transformacao[0] == 'log' else valor ** 2 + dados_nao_nulos.min()
                    else:
                        tratamento = 'mediana'
                        valor = df_limpo[coluna].median()
            else:
                tratamento = 'moda'
                valor = df_limpo[coluna].mode().iloc[0]
            
            df_limpo[coluna].fillna(valor, inplace=True)
            tratamentos[coluna] = {
                'metodo': tratamento, 
                'valor': valor,
                'shapiro_normal': normal_shapiro if 'normal_shapiro' in locals() else None,
                'anderson_normal': normal_anderson if 'normal_anderson' in locals() else None
            }
    
    return df_limpo, tratamentos


def criar_amostra(df, frac=0.1):
    return df.groupby('y').apply(lambda x: x.sample(frac=frac)).reset_index(drop=True)


def tratar_nulos(df, estrategia='media'):
    imputer = SimpleImputer(strategy='mean' if estrategia == 'media' else
                            'median' if estrategia == 'mediana' else
                            'most_frequent' if estrategia == 'moda' else
                            'constant')
    
    if estrategia == 'constante':
        imputer.set_params(fill_value=0)
    
    colunas = df.columns.drop('y')
    df_tratado = df.copy()
    df_tratado[colunas] = imputer.fit_transform(df[colunas])
    return df_tratado


def preparar_dados(df, aplicar_pca=False):
    X = df.drop('y', axis=1)
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if aplicar_pca:
        pca = PCA(n_components=0.95, random_state=42)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        print(f"Número de componentes PCA: {pca.n_components_}")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test


def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    
    metricas = {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba)
    }
    
    # Adicionando classificação por classe
    report = classification_report(y_test, y_pred, output_dict=True)
    for classe in report.keys():
        if classe not in ['accuracy', 'macro avg', 'weighted avg']:
            metricas[f"Precisão_Classe_{classe}"] = report[classe]['precision']
            metricas[f"Recall_Classe_{classe}"] = report[classe]['recall']
            metricas[f"F1-Score_Classe_{classe}"] = report[classe]['f1-score']
    
    for nome, valor in metricas.items():
        print(f"{nome}: {valor:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Matriz de Confusão')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Previsto')
    plt.show()
    
    return metricas


def treinar_e_avaliar(X_train, X_test, y_train, y_test, estrategia):
    modelos = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42)
    }
    
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.7, 0.8, 0.9]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'max_depth': [-1, 5, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 50, 100]
        }
    }
    
    resultados = []
    for nome, modelo in modelos.items():
        print(f"\nOtimizando e treinando {nome}...")
        
        inicio = time.time()
        random_search = RandomizedSearchCV(modelo, param_distributions=param_grids[nome], 
                                            n_iter=10, cv=3, random_state=42, n_jobs=-1)
        random_search.fit(X_train, y_train)
        fim = time.time()
        
        tempo_treinamento = fim - inicio
        
        melhor_modelo = random_search.best_estimator_
        print(f"Melhores parâmetros para {nome}: {random_search.best_params_}")
        print(f"Tempo de treinamento: {tempo_treinamento:.2f} segundos")
        
        print(f"Avaliação do {nome} (Estratégia de preparação: {estrategia}):")
        metricas = avaliar_modelo(melhor_modelo, X_test, y_test)
        
        resultado = {
            'modelo': nome,
            'estrategia': estrategia,
            'tempo_treinamento': tempo_treinamento,
            **random_search.best_params_,
            **metricas
        }
        
        resultados.append(resultado)
    
    return resultados, {nome: random_search.best_estimator_ for nome, _ in modelos.items()}


def selecionar_melhor_modelo(todos_resultados, dados_preparados):
    melhor_pontuacao = 0
    melhor_modelo = None
    melhor_estrategia = None
    melhor_nome = None

    for resultado in todos_resultados:
        estrategia = resultado['estrategia']
        nome = resultado['modelo']
        
        # Calcular a pontuação média
        acuracia = resultado['Acurácia']
        precisao_0 = resultado['Precisão_Classe_0']
        precisao_1 = resultado['Precisão_Classe_1']
        pontuacao_media = (acuracia + precisao_0 + precisao_1) / 3
        
        if pontuacao_media > melhor_pontuacao:
            melhor_pontuacao = pontuacao_media
            melhor_estrategia = estrategia
            melhor_nome = nome
            # Recuperar o modelo treinado dos dados_preparados
            X_train, _, y_train, _ = dados_preparados[estrategia]
            melhor_modelo = treinar_modelo(nome, X_train, y_train)

    # Calcular as métricas individuais do melhor modelo
    melhor_acuracia = next(r['Acurácia'] for r in todos_resultados if r['estrategia'] == melhor_estrategia and r['modelo'] == melhor_nome)
    melhor_precisao_0 = next(r['Precisão_Classe_0'] for r in todos_resultados if r['estrategia'] == melhor_estrategia and r['modelo'] == melhor_nome)
    melhor_precisao_1 = next(r['Precisão_Classe_1'] for r in todos_resultados if r['estrategia'] == melhor_estrategia and r['modelo'] == melhor_nome)

    return melhor_estrategia, melhor_nome, melhor_modelo, melhor_pontuacao, melhor_acuracia, melhor_precisao_0, melhor_precisao_1



def treinar_modelo(nome_modelo, X_train, y_train):
    if nome_modelo == 'Random Forest':
        modelo = RandomForestClassifier(random_state=42)
    elif nome_modelo == 'XGBoost':
        modelo = XGBClassifier(random_state=42)
    elif nome_modelo == 'LightGBM':
        modelo = LGBMClassifier(random_state=42)
    else:
        raise ValueError(f"Modelo desconhecido: {nome_modelo}")
    
    modelo.fit(X_train, y_train)
    return modelo


def treinar_e_prever_modelo_final(df, test, melhor_modelo, melhor_estrategia, tratar_nulos):
    print("\nTreinando modelo final com todos os dados...")
    df_final = tratar_nulos(df, melhor_estrategia).reset_index(drop=True)
    X_final = df_final.drop('y', axis=1)
    y_final = df_final['y']

    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)

    smote_final = SMOTE(random_state=42)
    X_final_balanced, y_final_balanced = smote_final.fit_resample(X_final_scaled, y_final)

    modelo_final = melhor_modelo.__class__(random_state=42)
    modelo_final.fit(X_final_balanced, y_final_balanced)

    test_tratado = tratar_nulos(test, melhor_estrategia)
    X_test_final = test_tratado.drop('y', axis=1) if 'y' in test_tratado.columns else test_tratado

    colunas_faltantes = set(X_final.columns) - set(X_test_final.columns)
    for coluna in colunas_faltantes:
        X_test_final[coluna] = 0

    X_test_final = X_test_final[X_final.columns]

    X_test_final_scaled = scaler_final.transform(X_test_final)
    y_pred_final = modelo_final.predict(X_test_final_scaled)

    predicoes = pd.DataFrame({'y_pred': y_pred_final})

    if 'id' not in test.columns:
        print("Aviso: A coluna 'id' não está presente no DataFrame de teste.")
        print("Salvando as previsões com o índice como identificador...")
        predicoes.to_csv('predicoes_finais.csv', index=True)
        print("Previsões finais salvas em 'predicoes_finais.csv' com o índice como identificador.")
    else:
        predicoes['id'] = test['id']
        predicoes[['id', 'y_pred']].to_csv('predicoes_finais.csv', index=False)
        print("Previsões finais salvas em 'predicoes_finais.csv'")

    return predicoes, modelo_final, y_pred_final


def treinar_e_prever_modelo_final2(df, test, melhor_modelo):
    print("\nTreinando modelo final com todos os dados...")
    df_final = analisar_e_tratar_nulos(df).reset_index(drop=True)
    X_final = df_final.drop('y', axis=1)
    y_final = df_final['y']

    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)

    smote_final = SMOTE(random_state=42)
    X_final_balanced, y_final_balanced = smote_final.fit_resample(X_final_scaled, y_final)

    modelo_final = melhor_modelo.__class__(random_state=42)
    modelo_final.fit(X_final_balanced, y_final_balanced)

    test_tratado = analisar_e_tratar_nulos(test)
    X_test_final = test_tratado.drop('y', axis=1) if 'y' in test_tratado.columns else test_tratado

    colunas_faltantes = set(X_final.columns) - set(X_test_final.columns)
    for coluna in colunas_faltantes:
        X_test_final[coluna] = 0

    X_test_final = X_test_final[X_final.columns]

    X_test_final_scaled = scaler_final.transform(X_test_final)
    y_pred_final = modelo_final.predict(X_test_final_scaled)

    predicoes = pd.DataFrame({'y_pred': y_pred_final})

    if 'id' not in test.columns:
        print("Aviso: A coluna 'id' não está presente no DataFrame de teste.")
        print("Salvando as previsões com o índice como identificador...")
        predicoes.to_csv('predicoes_finais.csv', index=True)
        print("Previsões finais salvas em 'predicoes_finais.csv' com o índice como identificador.")
    else:
        predicoes['id'] = test['id']
        predicoes[['id', 'y_pred']].to_csv('predicoes_finais.csv', index=False)
        print("Previsões finais salvas em 'predicoes_finais.csv'")

    return predicoes, modelo_final, y_pred_final


def avaliar_modelo_final(y_pred_final, test):
    print("\nTestando o modelo final com o conjunto de teste...")

    if 'y' in test.columns:
        y_test_real = test['y']
        
        acuracia = accuracy_score(y_test_real, y_pred_final)
        precisao = precision_score(y_test_real, y_pred_final)
        recall = recall_score(y_test_real, y_pred_final)
        f1 = f1_score(y_test_real, y_pred_final)
        
        print(f"Acurácia: {acuracia:.4f}")
        print(f"Precisão: {precisao:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        cm = confusion_matrix(y_test_real, y_pred_final)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusão')
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.show()
        
        print("\nRelatório de Classificação:")
        print(classification_report(y_test_real, y_pred_final))
        
        acertos = sum(y_test_real == y_pred_final)
        erros = sum(y_test_real != y_pred_final)
        total = len(y_test_real)
        
        print(f"\nTotal de amostras: {total}")
        print(f"Acertos: {acertos} ({acertos/total:.2%})")
        print(f"Erros: {erros} ({erros/total:.2%})")
    else:
        print("Aviso: Não foi possível calcular as métricas de desempenho porque o conjunto de teste não contém a coluna 'y' com os valores reais.")
        print("As previsões foram feitas, mas não podemos avaliar a precisão sem os dados reais.")
