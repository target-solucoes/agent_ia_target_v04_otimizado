"""
Módulo de normalização de texto para garantir consistência em consultas e dados.
Aplica transformações padrão para resolver problemas de capitalização e formatação.
"""

import re
import unicodedata
import pandas as pd
from typing import Union, List, Dict, Any


class TextNormalizer:
    """Classe para normalização consistente de texto em datasets e consultas."""
    
    def __init__(self):
        """Inicializa o normalizador com configurações padrão."""
        self.text_columns_cache = {}
    
    def normalize_text(self, text: Union[str, None]) -> str:
        """
        Normaliza uma string individual aplicando transformações consistentes.
        
        Args:
            text: String a ser normalizada
            
        Returns:
            String normalizada ou string vazia se entrada for None/NaN
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Converter para string se não for
        text = str(text)
        
        # Remover espaços extras no início e fim
        text = text.strip()
        
        # Normalizar caracteres Unicode (remover acentos)
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        # Converter para minúsculas
        text = text.lower()
        
        # Normalizar espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def normalize_column(self, series: pd.Series) -> pd.Series:
        """
        Normaliza uma coluna inteira do pandas DataFrame.
        
        Args:
            series: Serie do pandas a ser normalizada
            
        Returns:
            Serie normalizada
        """
        return series.apply(self.normalize_text)
    
    def identify_text_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identifica colunas que contêm texto e podem se beneficiar da normalização.
        
        Args:
            df: DataFrame para análise
            
        Returns:
            Lista de nomes de colunas que contêm texto
        """
        text_columns = []
        
        for col in df.columns:
            # Verificar se é coluna de tipo object ou category
            if df[col].dtype in ['object', 'category']:
                # Verificar se contém strings (não apenas números)
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    # Verificar se pelo menos alguns valores são strings não-numéricas
                    string_count = sum(
                        1 for val in sample_values 
                        if isinstance(val, str) and not val.strip().replace('.', '').replace(',', '').isdigit()
                    )
                    if string_count > len(sample_values) * 0.1:  # 10% threshold
                        text_columns.append(col)
        
        return text_columns
    
    def normalize_dataframe(self, df: pd.DataFrame, specific_columns: List[str] = None) -> pd.DataFrame:
        """
        Normaliza todas as colunas de texto de um DataFrame.
        
        Args:
            df: DataFrame a ser normalizado
            specific_columns: Lista específica de colunas para normalizar (opcional)
            
        Returns:
            DataFrame com colunas de texto normalizadas
        """
        df_normalized = df.copy()
        
        # Determinar quais colunas normalizar
        if specific_columns is not None:
            columns_to_normalize = specific_columns
        else:
            columns_to_normalize = self.identify_text_columns(df)
        
        # Aplicar normalização às colunas identificadas
        for col in columns_to_normalize:
            if col in df_normalized.columns:
                df_normalized[col] = self.normalize_column(df_normalized[col])
        
        return df_normalized
    
    def normalize_query_terms(self, query: str, alias_mapping: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        Normaliza termos de uma consulta do usuário e mapeia aliases.
        
        Args:
            query: Consulta do usuário
            alias_mapping: Dicionário de mapeamento de aliases (opcional)
            
        Returns:
            Dicionário com query normalizada e termos mapeados
        """
        normalized_query = self.normalize_text(query)
        
        result = {
            'original_query': query,
            'normalized_query': normalized_query,
            'mapped_terms': {}
        }
        
        # Se houver mapeamento de aliases, aplicar
        if alias_mapping:
            for column, aliases in alias_mapping.items():
                normalized_aliases = [self.normalize_text(alias) for alias in aliases]
                
                # Verificar se algum alias aparece na query normalizada
                for i, alias in enumerate(normalized_aliases):
                    if alias in normalized_query and alias.strip():
                        result['mapped_terms'][alias] = {
                            'original_alias': aliases[i],
                            'mapped_column': column
                        }
        
        return result
    
    def create_search_index(self, df: pd.DataFrame, text_columns: List[str] = None) -> Dict[str, Dict[str, List[int]]]:
        """
        Cria um índice de busca para facilitar consultas rápidas.
        
        Args:
            df: DataFrame para indexar
            text_columns: Colunas específicas para indexar (opcional)
            
        Returns:
            Dicionário com índice de busca por coluna e termo
        """
        if text_columns is None:
            text_columns = self.identify_text_columns(df)
        
        search_index = {}
        
        for col in text_columns:
            if col in df.columns:
                search_index[col] = {}
                
                for idx, value in df[col].items():
                    normalized_value = self.normalize_text(value)
                    if normalized_value:
                        if normalized_value not in search_index[col]:
                            search_index[col][normalized_value] = []
                        search_index[col][normalized_value].append(idx)
        
        return search_index


def load_alias_mapping(alias_file_path: str = None) -> Dict[str, List[str]]:
    """
    Carrega mapeamento de aliases de um arquivo JSON.
    
    Args:
        alias_file_path: Caminho para arquivo de aliases
        
    Returns:
        Dicionário com mapeamento de aliases
    """
    import json
    
    if alias_file_path is None:
        alias_file_path = "data/mappings/alias.json"
    
    try:
        with open(alias_file_path, 'r', encoding='utf-8') as f:
            alias_data = json.load(f)
        
        # Extrair apenas o mapeamento de colunas
        return alias_data.get('columns', {})
    
    except FileNotFoundError:
        print(f"Warning: Alias file not found at {alias_file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in alias file {alias_file_path}")
        return {}


# Instância global para uso conveniente
normalizer = TextNormalizer()