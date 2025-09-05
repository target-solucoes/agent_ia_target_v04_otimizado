"""
Módulo de normalização de texto para garantir consistência em consultas e dados.
Aplica transformações padrão para resolver problemas de capitalização e formatação.
"""

import re
import unicodedata
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
import yaml
import calendar
from datetime import datetime, timedelta

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
    
    def parse_temporal_entities(self, text: str) -> Dict[str, Any]:
        """
        Extrai e converte entidades temporais de texto natural para formatos estruturados.
        
        Exemplos:
        - "julho de 2015" → {"Data_>=": "2015-07-01", "Data_<": "2015-08-01"}
        - "janeiro 2020" → {"Data_>=": "2020-01-01", "Data_<": "2020-02-01"}
        - "dezembro de 2023" → {"Data_>=": "2023-12-01", "Data_<": "2024-01-01"}
        
        Args:
            text: Texto contendo possíveis referências temporais
            
        Returns:
            Dicionário com entidades temporais estruturadas
        """
        text_lower = text.lower().strip()
        
        # Mapeamento de meses em português
        month_mapping = {
            'janeiro': 1, 'jan': 1,
            'fevereiro': 2, 'fev': 2,
            'março': 3, 'mar': 3, 'marco': 3,
            'abril': 4, 'abr': 4,
            'maio': 5, 'mai': 5,
            'junho': 6, 'jun': 6,
            'julho': 7, 'jul': 7,
            'agosto': 8, 'ago': 8,
            'setembro': 9, 'set': 9, 'sep': 9,
            'outubro': 10, 'out': 10, 'oct': 10,
            'novembro': 11, 'nov': 11,
            'dezembro': 12, 'dez': 12, 'dec': 12
        }
        
        temporal_entities = {}
        
        # Padrão principal: "mês de ano" ou "mês ano"
        month_year_patterns = [
            r'\b(\w+)\s+de\s+(\d{4})\b',  # "julho de 2015"
            r'\b(\w+)\s+(\d{4})\b',       # "julho 2015"
            r'\bem\s+(\w+)\s+de\s+(\d{4})\b',  # "em julho de 2015"
            r'\bno\s+(\w+)\s+de\s+(\d{4})\b',  # "no julho de 2015"
            r'\bdurante\s+(\w+)\s+de\s+(\d{4})\b',  # "durante julho de 2015"
        ]
        
        for pattern in month_year_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                month_text = match.group(1).strip()
                year_text = match.group(2).strip()
                
                # Verificar se o mês é reconhecido
                if month_text in month_mapping:
                    month_num = month_mapping[month_text]
                    year_num = int(year_text)
                    
                    # Calcular primeiro e último dia do mês
                    start_date = f"{year_num:04d}-{month_num:02d}-01"
                    
                    # Calcular primeiro dia do mês seguinte
                    if month_num == 12:
                        next_month = 1
                        next_year = year_num + 1
                    else:
                        next_month = month_num + 1
                        next_year = year_num
                    
                    end_date = f"{next_year:04d}-{next_month:02d}-01"
                    
                    temporal_entities['Data_>='] = start_date
                    temporal_entities['Data_<'] = end_date
                    
                    # Adicionar metadados para debugging
                    temporal_entities['_temporal_metadata'] = {
                        'original_text': match.group(0),
                        'parsed_month': month_text,
                        'parsed_year': year_text,
                        'month_number': month_num,
                        'year_number': year_num
                    }
                    break  # Usar apenas a primeira ocorrência válida
        
        # Padrão para períodos entre meses: "entre junho e julho de 2015"
        between_pattern = r'\bentre\s+(\w+)\s+e\s+(\w+)\s+de\s+(\d{4})\b'
        between_match = re.search(between_pattern, text_lower)
        
        if between_match and not temporal_entities:
            start_month_text = between_match.group(1).strip()
            end_month_text = between_match.group(2).strip()
            year_text = between_match.group(3).strip()
            
            if start_month_text in month_mapping and end_month_text in month_mapping:
                start_month_num = month_mapping[start_month_text]
                end_month_num = month_mapping[end_month_text]
                year_num = int(year_text)
                
                # Data de início: primeiro dia do primeiro mês
                start_date = f"{year_num:04d}-{start_month_num:02d}-01"
                
                # Data de fim: primeiro dia do mês após o último mês
                if end_month_num == 12:
                    next_month = 1
                    next_year = year_num + 1
                else:
                    next_month = end_month_num + 1
                    next_year = year_num
                
                end_date = f"{next_year:04d}-{next_month:02d}-01"
                
                temporal_entities['Data_>='] = start_date
                temporal_entities['Data_<'] = end_date
                
                temporal_entities['_temporal_metadata'] = {
                    'original_text': between_match.group(0),
                    'type': 'period_between_months',
                    'start_month': start_month_text,
                    'end_month': end_month_text,
                    'parsed_year': year_text
                }
        
        # Padrão para anos individuais: "em 2015", "no ano de 2015"
        year_patterns = [
            r'\bem\s+(\d{4})\b',
            r'\bno\s+ano\s+de\s+(\d{4})\b',
            r'\bdurante\s+(\d{4})\b',
        ]
        
        if not temporal_entities:  # Só aplicar se não encontrou padrão mês/ano
            for pattern in year_patterns:
                year_match = re.search(pattern, text_lower)
                if year_match:
                    year_num = int(year_match.group(1))
                    
                    temporal_entities['Data_>='] = f"{year_num:04d}-01-01"
                    temporal_entities['Data_<'] = f"{year_num + 1:04d}-01-01"
                    
                    temporal_entities['_temporal_metadata'] = {
                        'original_text': year_match.group(0),
                        'type': 'full_year',
                        'parsed_year': year_num
                    }
                    break
        
        return temporal_entities
    
    def format_temporal_filter(self, temporal_data: Dict[str, Any]) -> str:
        """
        Converte entidades temporais em filtro SQL adequado.
        
        Args:
            temporal_data: Dados temporais extraídos de parse_temporal_entities()
            
        Returns:
            String com filtro SQL para cláusula WHERE
        """
        if not temporal_data or ('Data_>=' not in temporal_data or 'Data_<' not in temporal_data):
            return ""
        
        start_date = temporal_data['Data_>=']
        end_date = temporal_data['Data_<']
        
        # Retornar filtro SQL
        return f"Data >= '{start_date}' AND Data < '{end_date}'"
    
    def extract_and_format_temporal(self, text: str) -> Optional[Tuple[Dict[str, str], str]]:
        """
        Método conveniente que extrai entidades temporais e retorna tanto o contexto
        estruturado quanto o filtro SQL formatado.
        
        Args:
            text: Texto para processamento
            
        Returns:
            Tupla (contexto_estruturado, filtro_sql) ou None se nenhuma entidade encontrada
        """
        temporal_entities = self.parse_temporal_entities(text)
        
        if not temporal_entities:
            return None
        
        # Criar contexto estruturado (removendo metadados)
        context = {k: v for k, v in temporal_entities.items() if not k.startswith('_')}
        
        # Criar filtro SQL
        sql_filter = self.format_temporal_filter(temporal_entities)
        
        return context, sql_filter


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
        alias_file_path = "data/mappings/alias.yaml"
    
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
