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
        random_search = RandomizedSearchCV(modelo, param_distributions=param_grids[nome], 
                                            n_iter=10, cv=3, random_state=42, n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        melhor_modelo = random_search.best_estimator_
        print(f"Melhores parâmetros para {nome}: {random_search.best_params_}")
        
        print(f"Avaliação do {nome} (Estratégia de preparação: {estrategia}):")
        metricas = avaliar_modelo(melhor_modelo, X_test, y_test)
        
        resultado = {
            'modelo': nome,
            'estrategia': estrategia,
            **random_search.best_params_,
            **metricas
        }
        
        resultados.append(resultado)
    
    pd.DataFrame(resultados).to_parquet('resultados_modelos.parquet')
    return resultados, {nome: random_search.best_estimator_ for nome, _ in modelos.items()}


def selecionar_melhor_modelo(resultados_modelos, dados_preparados):
    melhor_modelo = None
    melhor_f1 = 0
    melhor_estrategia = None
    melhor_nome = None

    for estrategia, resultados in resultados_modelos.items():
        for resultado in resultados:
            nome = resultado['modelo']
            f1 = resultado['F1-Score']
            if f1 > melhor_f1:
                melhor_f1 = f1
                melhor_estrategia = estrategia
                melhor_nome = nome

    X_train = dados_preparados[melhor_estrategia][0]
    y_train = dados_preparados[melhor_estrategia][2]
    
    if melhor_nome == 'Random Forest':
        melhor_modelo = RandomForestClassifier(random_state=42)
    elif melhor_nome == 'XGBoost':
        melhor_modelo = XGBClassifier(random_state=42)
    elif melhor_nome == 'LightGBM':
        melhor_modelo = LGBMClassifier(random_state=42)
    
    melhores_params = next(resultado for resultado in resultados_modelos[melhor_estrategia] if resultado['modelo'] == melhor_nome)
    params = {k: v for k, v in melhores_params.items() if k not in ['modelo', 'estrategia', 'Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']}
    
    melhor_modelo.set_params(**params)
    melhor_modelo.fit(X_train, y_train)

    return melhor_estrategia, melhor_nome, melhor_modelo, melhor_f1


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
