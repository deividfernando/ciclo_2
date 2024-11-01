import pandas as pd
import streamlit as st
import requests
from src.config.constants import LOTTIE_URLS, FILE_PATHS, ERROR_MESSAGES
from datetime import datetime
import os
from cryptography.fernet import Fernet
import io


@st.cache_data
def load_lottie_url(url: str) -> dict:
    """
    Carrega uma animação Lottie de uma URL.

    Args:
        url (str): URL da animação Lottie

    Returns:
        dict: Dados da animação ou None se houver erro
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erro ao carregar animação: {str(e)}")
        return None


def get_crypto_key():
    """
    Obtém a chave de criptografia do ambiente.

    Returns:
        bytes: Chave de criptografia em bytes
    """
    try:
        # Carregar variáveis de ambiente
        from dotenv import load_dotenv

        load_dotenv()

        # Obter chave do ambiente
        key = os.getenv("CRYPTO_KEY")
        if not key:
            raise ValueError("CRYPTO_KEY não encontrada nas variáveis de ambiente")

        # Remover possíveis aspas e espaços
        key = key.strip().strip('"').strip("'")

        # Converter para bytes e retornar
        return key.encode()

    except Exception as e:
        st.error(f"Erro ao carregar chave de criptografia: {str(e)}")
        return None


@st.cache_data
def load_data():
    """
    Carrega os dados de treino criptografados.
    """
    try:
        # Obter chave de criptografia
        key = get_crypto_key()
        if key is None:
            raise ValueError("Não foi possível obter a chave de criptografia")

        f = Fernet(key)

        # Carregar e descriptografar dados
        with open(FILE_PATHS["train_data"], "rb") as file:
            encrypted_data = file.read()
            decrypted_data = f.decrypt(encrypted_data)
            df = pd.read_parquet(io.BytesIO(decrypted_data))

        return df

    except Exception as e:
        st.error(f"{ERROR_MESSAGES['data_load']} Erro: {str(e)}")
        return None


@st.cache_data
def load_test_data():
    """
    Carrega os dados de teste criptografados.
    """
    try:
        # Obter chave de criptografia
        key = get_crypto_key()
        if key is None:
            raise ValueError("Não foi possível obter a chave de criptografia")

        f = Fernet(key)

        # Carregar e descriptografar dados
        with open(FILE_PATHS["test_data"], "rb") as file:
            encrypted_data = file.read()
            decrypted_data = f.decrypt(encrypted_data)
            test = pd.read_parquet(io.BytesIO(decrypted_data))

        return test

    except Exception as e:
        st.error(f"{ERROR_MESSAGES['data_load']} Erro: {str(e)}")
        return None


@st.cache_data
def load_results() -> pd.DataFrame:
    """
    Carrega e valida os resultados dos modelos.

    Returns:
        pd.DataFrame: DataFrame com os resultados ou None se houver erro
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(FILE_PATHS["results"]):
            raise FileNotFoundError(f"Arquivo não encontrado: {FILE_PATHS['results']}")

        # Carregar resultados
        results = pd.read_parquet(FILE_PATHS["results"])

        # Validar estrutura básica dos resultados
        required_columns = [
            "modelo",
            "estrategia",
            "Acurácia",
            "Precisão",
            "Recall",
            "F1-Score",
        ]
        missing_columns = [
            col for col in required_columns if col not in results.columns
        ]

        if missing_columns:
            raise ValueError(f"Colunas ausentes nos resultados: {missing_columns}")

        return results

    except Exception as e:
        st.error(f"Erro ao carregar resultados: {str(e)}")
        return None


@st.cache_data
def load_all_animations() -> dict:
    """
    Carrega todas as animações Lottie necessárias.

    Returns:
        dict: Dicionário com as animações carregadas
    """
    animations = {}
    for key, url in LOTTIE_URLS.items():
        animations[key] = load_lottie_url(url)
    return animations


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Coleta informações básicas sobre o DataFrame.

    Args:
        df (pd.DataFrame): DataFrame para análise

    Returns:
        dict: Dicionário com informações sobre os dados
    """
    if df is None:
        return None

    try:
        info = {
            "total_registros": len(df),
            "total_colunas": len(df.columns),
            "memoria_uso": f"{df.memory_usage().sum() / 1024**2:.2f} MB",
            "valores_nulos": df.isnull().sum().to_dict(),
            "tipos_dados": df.dtypes.to_dict(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if "y" in df.columns:
            info["distribuicao_classes"] = df["y"].value_counts().to_dict()

        return info

    except Exception as e:
        st.error(f"Erro ao coletar informações dos dados: {str(e)}")
        return None


def validate_data_consistency(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """
    Valida a consistência entre os dados de treino e teste.

    Args:
        train_df (pd.DataFrame): DataFrame de treino
        test_df (pd.DataFrame): DataFrame de teste

    Returns:
        bool: True se os dados são consistentes, False caso contrário
    """
    try:
        # Verificar se os DataFrames existem
        if train_df is None or test_df is None:
            return False

        # Comparar colunas (exceto target 'y')
        train_cols = set(train_df.columns) - {"y"}
        test_cols = set(test_df.columns) - {"y"}

        if train_cols != test_cols:
            st.warning("Diferença nas colunas entre treino e teste")
            return False

        # Comparar tipos de dados
        for col in train_cols:
            if train_df[col].dtype != test_df[col].dtype:
                st.warning(f"Tipo de dado diferente para coluna {col}")
                return False

        return True

    except Exception as e:
        st.error(f"Erro ao validar consistência dos dados: {str(e)}")
        return False


def create_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria um resumo estatístico dos dados.

    Args:
        df (pd.DataFrame): DataFrame para análise

    Returns:
        pd.DataFrame: DataFrame com o resumo estatístico
    """
    try:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        summary = pd.DataFrame(
            {
                "tipo_dado": df.dtypes,
                "valores_nulos": df.isnull().sum(),
                "valores_unicos": df.nunique(),
                "media": df[numeric_cols].mean(),
                "mediana": df[numeric_cols].median(),
                "desvio_padrao": df[numeric_cols].std(),
                "minimo": df[numeric_cols].min(),
                "maximo": df[numeric_cols].max(),
            }
        )

        return summary

    except Exception as e:
        st.error(f"Erro ao criar resumo dos dados: {str(e)}")
        return None
