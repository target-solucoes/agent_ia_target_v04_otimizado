from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.tools.duckdb import DuckDbTools
from agno.knowledge import AgentKnowledge
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.tools.calculator import CalculatorTools
from agno.tools.python import PythonTools

import os
import pandas as pd
import tempfile
import re
import json
from dotenv import load_dotenv
from text_normalizer import TextNormalizer, load_alias_mapping

load_dotenv()
selected_model = "gpt-5-nano-2025-08-07"


def extract_where_clause_context(sql_query):
    """
    Extrai o contexto da cláusula WHERE de uma query SQL e retorna em formato JSON
    """
    try:
        # Normalizar a query (remover quebras de linha e múltiplos espaços)
        normalized_query = re.sub(r'\s+', ' ', sql_query.strip())
        
        # Encontrar a cláusula WHERE
        where_match = re.search(r'\bWHERE\b(.*?)(?:\bGROUP BY\b|\bORDER BY\b|\bHAVING\b|\bLIMIT\b|$)', 
                               normalized_query, re.IGNORECASE)
        
        if not where_match:
            return {}
        
        where_clause = where_match.group(1).strip()
        context = {}
        
        # Buscar por padrões de filtro na cláusula WHERE
        # Igualdade com aspas simples: coluna = 'valor' OU LOWER(coluna) = 'valor'
        equality_single = re.findall(r"(?:LOWER\()?(\w+)\)?\s*=\s*'([^']*)'", where_clause, re.IGNORECASE)
        for column, value in equality_single:
            if column not in context:
                context[column] = value
        
        # Igualdade com aspas duplas: coluna = "valor" OU LOWER(coluna) = "valor"
        equality_double = re.findall(r"(?:LOWER\()?(\w+)\)?\s*=\s*\"([^\"]*)\"", where_clause, re.IGNORECASE)
        for column, value in equality_double:
            if column not in context:
                context[column] = value
        
        # LIKE com aspas simples: coluna LIKE 'valor' OU LOWER(coluna) LIKE 'valor'
        like_single = re.findall(r"(?:LOWER\()?(\w+)\)?\s+LIKE\s+'([^']*)'", where_clause, re.IGNORECASE)
        for column, value in like_single:
            if column not in context:
                context[column] = value
        
        # LIKE com aspas duplas: coluna LIKE "valor" OU LOWER(coluna) LIKE "valor"
        like_double = re.findall(r"(?:LOWER\()?(\w+)\)?\s+LIKE\s+\"([^\"]*)\"", where_clause, re.IGNORECASE)
        for column, value in like_double:
            if column not in context:
                context[column] = value
        
        # IN: coluna IN (...) OU LOWER(coluna) IN (...)
        in_clauses = re.findall(r"(?:LOWER\()?(\w+)\)?\s+IN\s*\([^)]+\)", where_clause, re.IGNORECASE)
        for column in in_clauses:
            key = f"{column}_IN"
            if key not in context:
                context[key] = "lista_valores"
        
        # Comparações: coluna > valor, coluna < valor, etc. OU LOWER(coluna) > valor
        comparisons = re.findall(r"(?:LOWER\()?(\w+)\)?\s*([><=!]+)\s*([^\s'\"]+)", where_clause, re.IGNORECASE)
        for column, operator, value in comparisons:
            if operator != '=':  # Não sobrescrever igualdades já processadas
                key = f"{column}_{operator}"
                if key not in context:
                    context[key] = value.strip('\'"')
        
        return context
    
    except Exception as e:
        # Em caso de erro, retornar contexto básico
        return {"erro_parsing": str(e)}


def create_agent(session_user_id=None, debug_mode=False):
    """Cria e configura o agente DuckDB com acesso aos dados comerciais e memória temporária"""
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Carregar dados do parquet
    data_path = "data/raw/DadosComercial_resumido_v02.parquet"
    df = pd.read_parquet(data_path)

    # Aplicar normalização de texto aos dados
    normalizer = TextNormalizer()
    text_columns = normalizer.identify_text_columns(df)

    # Criar versão normalizada do DataFrame para buscas
    df_normalized = normalizer.normalize_dataframe(df, text_columns)

    # Carregar mapeamento de aliases
    alias_mapping = load_alias_mapping()

    # Criar knowledge base com os dados usando AgentKnowledge
    knowledge = AgentKnowledge()

    # Adicionar informações sobre o dataset
    dataset_info = f"""
Dataset: DadosComercial_resumido_v02.parquet
Localização: {data_path}
Número de linhas: {len(df)}
Número de colunas: {len(df.columns)}
Colunas disponíveis: {", ".join(df.columns.tolist())}

IMPORTANTE: Os dados passaram por normalização de texto para garantir consistência:
- Colunas de texto normalizadas: {", ".join(text_columns)}
- Normalização aplicada: conversão para minúsculas, remoção de acentos, normalização de espaços
- Aliases disponíveis para consultas: {", ".join(alias_mapping.keys()) if alias_mapping else "Nenhum"}

Primeiras 5 linhas do dataset original:
{df.head().to_string()}

Primeiras 5 linhas com normalização aplicada (colunas de texto):
{df_normalized[text_columns].head().to_string() if text_columns else "Nenhuma coluna de texto para normalizar"}

Informações estatísticas:
{df.describe().to_string()}

Tipos de dados:
{df.dtypes.to_string()}
"""

    knowledge.load_text(dataset_info)

    # Configurar memória temporária (em memória, efêmera)
    # Criar um arquivo temporário único para esta sessão
    temp_dir = tempfile.gettempdir()
    temp_db_path = os.path.join(
        temp_dir, f"temp_memory_{session_user_id or 'default'}.db"
    )

    memory_db = SqliteMemoryDb(table_name="temp_memory", db_file=temp_db_path)
    memory = Memory(model=OpenAIChat(id=selected_model, reasoning_effort="low"), db=memory_db)

    # Criar classe customizada de PythonTools para controlar execução
    class OptimizedPythonTools(PythonTools):
        def __init__(self, debug_info_ref=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.debug_info_ref = debug_info_ref
            self.executed_calculations = set()  # Controle de cálculos já executados
            self.variable_cache = {}  # Cache de variáveis importantes
            
        def run_code(self, code: str) -> str:
            """Override para controlar execuções redundantes e variáveis"""
            # Evitar prints repetitivos de percentuais e valores
            repetitive_patterns = [
                'print(top5_total)', 'print(pe', 'print(sc', 'print("pe', 'print("sc',
                'participação', 'percentual', 'top5_total', '%'
            ]
            
            if any(pattern in code.lower() for pattern in repetitive_patterns):
                code_hash = hash(code.strip())
                if code_hash in self.executed_calculations:
                    return "Resultado já calculado e exibido."
                self.executed_calculations.add(code_hash)
            
            # Controle rigoroso de variáveis Top5_total
            if 'Top5_total' in code:
                if '=' not in code and 'Top5_total' not in self.variable_cache:
                    # Tentando usar sem definir - bloquear
                    return "Erro: Variável Top5_total não está definida no contexto atual."
                elif '=' in code and code.strip().startswith('Top5_total'):
                    # Definindo a variável - permitir e cachear
                    pass
            
            try:
                result = super().run_code(code)
                
                # Cache inteligente de variáveis importantes
                if 'Top5_total' in code and '=' in code and code.strip().startswith('Top5_total'):
                    try:
                        # Extrair valor do resultado se possível
                        import re
                        numeric_match = re.search(r'(\d+\.?\d*)', str(result))
                        if numeric_match:
                            self.variable_cache['Top5_total'] = float(numeric_match.group(1))
                    except:
                        pass
                        
                return result
                
            except NameError as e:
                if 'Top5_total' in str(e):
                    return "Erro: A variável Top5_total não está disponível neste contexto de execução."
                return f"Erro de variável não definida: {str(e)}"
            except Exception as e:
                return f"Erro na execução do código: {str(e)}"

    # Criar classe customizada de DuckDbTools para capturar queries e ser inteligente com strings
    class DebugDuckDbTools(DuckDbTools):
        def __init__(self, debug_info_ref=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.debug_info_ref = debug_info_ref
            self.last_result_df = None  # Armazenar último DataFrame resultado


        def _normalize_query_strings(self, query: str) -> str:
            """Aplica normalização LOWER() automaticamente a todas as comparações de strings na query"""
            import re
            
            # Pattern para detectar comparações de strings: coluna = 'valor', coluna LIKE 'valor', etc.
            # Captura: operador de comparação, nome da coluna, operador, valor entre aspas
            patterns = [
                # Igualdade: WHERE coluna = 'valor'
                (r"(\w+)\s*(=)\s*'([^']*)'", r"LOWER(\1) \2 '\3'"),
                # LIKE: WHERE coluna LIKE 'valor'  
                (r"(\w+)\s+(LIKE)\s+'([^']*)'", r"LOWER(\1) \2 '\3'"),
                # Igualdade com aspas duplas: WHERE coluna = "valor"
                (r"(\w+)\s*(=)\s*\"([^\"]*)\"", r"LOWER(\1) \2 '\3'"),
                # LIKE com aspas duplas: WHERE coluna LIKE "valor"
                (r"(\w+)\s+(LIKE)\s+\"([^\"]*)\"", r"LOWER(\1) \2 '\3'"),
            ]
            
            normalized_query = query
            applied_normalizations = []
            
            for pattern, replacement in patterns:
                # Encontrar todas as correspondências
                matches = re.finditer(pattern, normalized_query, re.IGNORECASE)
                
                for match in matches:
                    column = match.group(1)
                    operator = match.group(2)  
                    value = match.group(3)
                    
                    # Converter o valor para lowercase também
                    normalized_value = value.lower()
                    
                    # Aplicar a substituição com LOWER() na coluna e valor normalizado
                    old_text = match.group(0)
                    new_text = f"LOWER({column}) {operator} '{normalized_value}'"
                    
                    normalized_query = normalized_query.replace(old_text, new_text)
                    applied_normalizations.append({
                        "column": column,
                        "operator": operator,
                        "original_value": value,
                        "normalized_value": normalized_value
                    })
            
            # Log das normalizações aplicadas para debug
            if applied_normalizations and self.debug_info_ref and hasattr(self.debug_info_ref, "debug_info"):
                if "string_normalizations" not in self.debug_info_ref.debug_info:
                    self.debug_info_ref.debug_info["string_normalizations"] = []
                self.debug_info_ref.debug_info["string_normalizations"].extend(applied_normalizations)
            
            return normalized_query

        def run_query(self, query: str) -> str:
            """Override do método run_query com normalização automática de strings e captura de contexto"""
            
            # APLICAR NORMALIZAÇÃO AUTOMÁTICA de todas as strings na query
            normalized_query = self._normalize_query_strings(query)
            
            # Executar a query normalizada
            result = super().run_query(normalized_query)
            
            # CAPTURAR DADOS DO RESULTADO para visualização
            try:
                # Tentar executar novamente a query para capturar DataFrame
                if hasattr(self, 'connection') and self.connection:
                    import pandas as pd
                    df_result = self.connection.execute(normalized_query).df()
                    if not df_result.empty:
                        self.last_result_df = df_result
            except Exception as e:
                # Se falhar, tentar extrair dados do resultado textual
                self.last_result_df = self._parse_result_to_dataframe(result)
            
            # Debug info e context extraction
            if self.debug_info_ref is not None and hasattr(
                self.debug_info_ref, "debug_info"
            ):
                if "sql_queries" not in self.debug_info_ref.debug_info:
                    self.debug_info_ref.debug_info["sql_queries"] = []
                if "query_contexts" not in self.debug_info_ref.debug_info:
                    self.debug_info_ref.debug_info["query_contexts"] = []

                # Usar a query normalizada final
                clean_query = normalized_query.strip()
                if (
                    clean_query
                    and clean_query not in self.debug_info_ref.debug_info["sql_queries"]
                ):
                    self.debug_info_ref.debug_info["sql_queries"].append(clean_query)
                    
                    # SEMPRE extrair contexto, mesmo que vazio
                    context = extract_where_clause_context(clean_query)
                    # Adicionar contexto mesmo se vazio (para garantir que sempre apareça)
                    self.debug_info_ref.debug_info["query_contexts"].append(context if context else {})

            return result

        def _parse_result_to_dataframe(self, result_text):
            """Converte resultado textual em DataFrame quando possível"""
            try:
                import pandas as pd
                import re
                
                # Procurar por padrões de tabela no resultado
                lines = result_text.split('\n')
                data_rows = []
                
                # Procurar por linhas que parecem dados tabulares
                for line in lines:
                    line = line.strip()
                    if '|' in line or '\t' in line:
                        # Possível linha de dados
                        if '|' in line:
                            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                        else:
                            cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                        
                        if len(cells) >= 2:
                            # Tentar converter última célula para número
                            try:
                                value = float(cells[-1].replace(',', '').replace('$', ''))
                                data_rows.append({
                                    'label': cells[0],
                                    'value': value
                                })
                            except:
                                continue
                
                return pd.DataFrame(data_rows) if data_rows else None
            except:
                return None

    # Criar classe customizada de agent que aplica normalização às consultas
    class PrincipalAgent(Agent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.normalizer = normalizer
            self.alias_mapping = alias_mapping
            self.df_normalized = df_normalized
            self.text_columns = text_columns
            self.memory = memory
            self.session_user_id = session_user_id or "default_user"
            self.debug_info = {}  # Para armazenar informações de debug
            
            # NOVA FUNCIONALIDADE: Memória contextual acumulativa
            self.persistent_context = {}  # Context que persiste entre queries

            # Substituir ferramentas por versões otimizadas
            self.python_tool_ref = None  # Referência para o PythonTool otimizado
            for i, tool in enumerate(self.tools):
                if isinstance(tool, DuckDbTools):
                    self.tools[i] = DebugDuckDbTools(debug_info_ref=self)
                elif isinstance(tool, PythonTools):
                    optimized_tool = OptimizedPythonTools(debug_info_ref=self, run_code=True, pip_install=False)
                    self.tools[i] = optimized_tool
                    self.python_tool_ref = optimized_tool

        def clear_execution_state(self):
            """Limpa o estado de execução entre consultas relacionadas"""
            if self.python_tool_ref:
                self.python_tool_ref.executed_calculations.clear()
                # Manter apenas variáveis importantes no cache
                important_vars = {}
                for var_name, var_value in self.python_tool_ref.variable_cache.items():
                    if var_name in ['Top5_total']:
                        important_vars[var_name] = var_value
                self.python_tool_ref.variable_cache = important_vars

        def detect_top_n_query(self, query: str) -> dict:
            """
            Detecta se a consulta é do tipo Top N (ranking/maiores/melhores)
            e retorna informações sobre o tipo de visualização necessária.
            """
            query_lower = query.lower()
            
            # Keywords que indicam consultas de Top N/ranking
            top_n_keywords = [
                'top', 'maiores', 'melhores', 'ranking', 'principais', 
                'primeiros', 'lideres', 'líderes', 'destaque', 'topo',
                'mais vendidos', 'mais lucrativos', 'mais importantes'
            ]
            
            # Detectar presença de keywords Top N
            is_top_n = any(keyword in query_lower for keyword in top_n_keywords)
            
            # Detectar quantidade (Top 5, Top 10, etc.)
            import re
            number_match = re.search(r'top\s*(\d+)|(\d+)\s*maiores|(\d+)\s*melhores|(\d+)\s*principais', query_lower)
            top_limit = None
            if number_match:
                top_limit = int(number_match.group(1) or number_match.group(2) or number_match.group(3))
            
            return {
                'is_top_n': is_top_n,
                'top_limit': top_limit or 10,  # Default para Top 10
                'visualization_type': 'bar_chart' if is_top_n else 'table',
                'keywords_found': [kw for kw in top_n_keywords if kw in query_lower]
            }

        def process_and_visualize(self, query: str, response_content: str, top_n_info: dict) -> dict:
            """
            Processa o conteúdo da resposta e gera metadados de visualização
            para consultas Top N ou dados tabulares normais.
            Extrai dados reais da resposta em vez de usar dados mockados.
            """
            visualization_data = {
                'type': 'table',  # Default
                'has_data': False,
                'data': None,
                'config': {}
            }
            
            # Se não é consulta Top N, retornar configuração para tabela
            if not top_n_info['is_top_n']:
                return visualization_data
            
            # Para consultas Top N, extrair dados reais da resposta
            try:
                import pandas as pd
                import re
                
                # Tentar extrair dados da execução SQL mais recente
                real_data = self._extract_data_from_sql_results()
                
                if real_data and len(real_data) > 0:
                    # Usar dados reais do SQL
                    df = pd.DataFrame(real_data)
                else:
                    # Fallback: tentar extrair da resposta textual
                    df = self._extract_data_from_response_text(response_content)
                
                # Se conseguiu extrair dados, processar para visualização
                if df is not None and not df.empty and len(df.columns) >= 2:
                    # Converter colunas categóricas que podem estar como numéricas
                    df = self._preprocess_categorical_columns(df)
                    
                    # Identificar colunas automaticamente
                    label_col, value_col = self._identify_chart_columns(df, query)
                    
                    # Renomear colunas para padronização
                    df_chart = df[[label_col, value_col]].copy()
                    df_chart.columns = ['label', 'value']
                    
                    # Garantir que a coluna label seja tratada como string
                    df_chart['label'] = df_chart['label'].astype(str)
                    
                    # Limitar ao top_limit e ordenar
                    df_limited = df_chart.head(top_n_info['top_limit'])
                    df_sorted = df_limited.sort_values('value', ascending=False)
                    
                    # Gerar título dinâmico baseado na query
                    chart_title = self._generate_chart_title(query, top_n_info['top_limit'], label_col, value_col)
                    
                    # Detectar se a coluna label são IDs categóricos
                    is_categorical_id = self._is_categorical_id_column(label_col, df_sorted['label'])
                    
                    visualization_data = {
                        'type': 'bar_chart',
                        'has_data': True,
                        'data': df_sorted,
                        'config': {
                            'orientation': 'horizontal',
                            'title': chart_title,
                            'x_column': 'value',
                            'y_column': 'label',
                            'max_items': top_n_info['top_limit'],
                            'value_format': self._detect_value_format(df_sorted['value']),
                            'is_categorical_id': is_categorical_id,
                            'original_label_column': label_col
                        }
                    }
                else:
                    # Se não conseguiu extrair dados reais, não gerar gráfico
                    visualization_data['type'] = 'table'
                    
            except Exception as e:
                # Em caso de erro, voltar para tabela padrão
                visualization_data['type'] = 'table'
                
            return visualization_data

        def _preprocess_categorical_columns(self, df):
            """Pré-processa colunas que devem ser tratadas como categóricas"""
            try:
                # Colunas que devem ser convertidas para string/categoria
                categorical_id_cols = ['cod_cliente', 'codigo_cliente', 'cliente_id', 'id_cliente']
                
                df_processed = df.copy()
                for col in df_processed.columns:
                    # Verificar se é uma coluna de ID categórico
                    if any(cat_col in col.lower() for cat_col in categorical_id_cols):
                        # Converter para string se for numérica
                        if df_processed[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                            df_processed[col] = df_processed[col].astype(str)
                
                return df_processed
            except Exception as e:
                # Se houver erro, retornar DataFrame original
                return df

        def _extract_data_from_sql_results(self):
            """Extrai dados da última execução SQL bem-sucedida"""
            try:
                # Primeiro: tentar obter dados do DuckDbTools
                for tool in self.tools:
                    if isinstance(tool, DebugDuckDbTools) and hasattr(tool, 'last_result_df'):
                        if tool.last_result_df is not None and not tool.last_result_df.empty:
                            return tool.last_result_df.to_dict('records')
                
                # Fallback: tentar obter dados do PythonTool
                if self.python_tool_ref and hasattr(self.python_tool_ref, 'last_dataframe'):
                    return self.python_tool_ref.last_dataframe.to_dict('records')
                
                return None
            except:
                return None

        def _extract_data_from_response_text(self, response_content: str):
            """Extrai dados tabulares do texto da resposta"""
            try:
                import pandas as pd
                import re
                
                # Procurar por padrões de dados estruturados na resposta
                lines = response_content.split('\n')
                data_rows = []
                
                # Padrões para identificar dados tabulares
                patterns = [
                    r'^\d+\.\s*([^:]+?):\s*([^\s]+)',  # "1. Item: Valor"
                    r'^([^|]+?)\|\s*([^\s|]+)',        # "Item | Valor"
                    r'^([^:]+?):\s*([^\s]+)',          # "Item: Valor"
                ]
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    for pattern in patterns:
                        match = re.match(pattern, line)
                        if match:
                            label = match.group(1).strip()
                            value_str = match.group(2).strip()
                            
                            # Extrair valor numérico
                            value_match = re.search(r'[\d,]+\.?\d*', value_str.replace(',', '').replace('.', ''))
                            if value_match:
                                try:
                                    value = float(value_match.group(0).replace(',', ''))
                                    data_rows.append({'label': label, 'value': value})
                                    break
                                except:
                                    continue
                
                return pd.DataFrame(data_rows) if data_rows else None
            except:
                return None

        def _identify_chart_columns(self, df, query):
            """Identifica automaticamente as colunas para label (Y) e value (X)"""
            try:
                # Colunas que devem ser tratadas como categorias mesmo se numéricas
                categorical_id_cols = ['cod_cliente', 'codigo_cliente', 'cliente_id', 'id_cliente']
                
                # Analisar nomes das colunas e tipos de dados
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                
                # Remover colunas de ID categóricas das numéricas e adicionar às textuais
                categorical_numeric_cols = []
                for col in numeric_cols[:]:
                    if any(cat_col in col.lower() for cat_col in categorical_id_cols):
                        categorical_numeric_cols.append(col)
                        numeric_cols.remove(col)
                        text_cols.append(col)
                
                # Heurísticas para identificar colunas
                value_col = None
                label_col = None
                
                # Priorizar colunas numéricas para valores (exceto IDs categóricos)
                if numeric_cols:
                    # Procurar por colunas com keywords de valor
                    value_keywords = ['valor', 'vendas', 'receita', 'faturamento', 'total', 'sum', 'count', 'quantidade']
                    for col in numeric_cols:
                        if any(keyword in col.lower() for keyword in value_keywords):
                            value_col = col
                            break
                    
                    # Se não encontrou por keyword, usar primeira coluna numérica
                    if not value_col:
                        value_col = numeric_cols[0]
                
                # Procurar por colunas de texto para labels (incluindo IDs categóricos)
                if text_cols:
                    # Procurar por colunas com keywords de entidade
                    label_keywords = ['uf', 'estado', 'cidade', 'municipio', 'cliente', 'produto', 'linha', 'categoria']
                    for col in text_cols:
                        if any(keyword in col.lower() for keyword in label_keywords):
                            label_col = col
                            break
                    
                    # Se não encontrou por keyword, usar primeira coluna de texto
                    if not label_col:
                        label_col = text_cols[0]
                
                # Fallback: usar primeiras colunas disponíveis
                if not label_col:
                    label_col = df.columns[0]
                if not value_col:
                    value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                
                return label_col, value_col
            except:
                # Fallback absoluto
                return df.columns[0], df.columns[1] if len(df.columns) > 1 else df.columns[0]

        def _generate_chart_title(self, query, limit, label_col, value_col):
            """Gera título dinâmico baseado na query e colunas identificadas"""
            try:
                # Mapear colunas para nomes amigáveis
                label_mapping = {
                    'uf_cliente': 'Estados',
                    'municipio_cliente': 'Cidades', 
                    'des_linha_produto': 'Produtos',
                    'cliente': 'Clientes',
                    'produto': 'Produtos',
                    'cod_cliente': 'Clientes',
                    'codigo_cliente': 'Clientes',
                    'cliente_id': 'Clientes',
                    'id_cliente': 'Clientes'
                }
                
                value_mapping = {
                    'valor_vendido': 'Vendas',
                    'total_vendas': 'Vendas',
                    'receita': 'Receita',
                    'faturamento': 'Faturamento'
                }
                
                # Obter nomes amigáveis
                label_name = label_mapping.get(label_col.lower(), label_col.replace('_', ' ').title())
                value_name = value_mapping.get(value_col.lower(), value_col.replace('_', ' ').title())
                
                return f"Top {limit} {label_name} por {value_name}"
            except:
                return f"Top {limit} Resultados"

        def _detect_value_format(self, values):
            """Detecta o formato apropriado para os valores (moeda, número, etc.)"""
            try:
                # Verificar se os valores parecem ser monetários
                if values.max() > 1000:
                    return 'currency'
                else:
                    return 'number'
            except:
                return 'number'

        def _is_categorical_id_column(self, column_name, values):
            """Detecta se uma coluna deve ser tratada como ID categórico"""
            try:
                # Verificar pelo nome da coluna
                categorical_id_patterns = ['cod_cliente', 'codigo_cliente', 'cliente_id', 'id_cliente', 'cliente']
                if any(pattern in column_name.lower() for pattern in categorical_id_patterns):
                    return True
                
                # Verificar pelos valores - se parecem códigos numéricos curtos
                sample_values = values.head().astype(str)
                numeric_ids = 0
                for val in sample_values:
                    if val.isdigit() and 3 <= len(val) <= 8:
                        numeric_ids += 1
                
                # Se mais de 50% dos valores parecem IDs numéricos, tratar como categoria
                if numeric_ids / len(sample_values) >= 0.5:
                    return True
                
                return False
            except:
                return False
        
        def detect_explicit_context_changes(self, query: str) -> dict:
            """
            Detecta mudanças explícitas no contexto baseadas na query do usuário.
            Retorna um dicionário com os campos que devem ser alterados explicitamente.
            """
            explicit_changes = {}
            query_lower = query.lower()
            
            # === DETECÇÃO DE LOCALIZAÇÃO ===
            # Detectar mudanças explícitas de localização/município com padrões mais robustos
            city_indicators = ['em ', 'para ', 'de ', 'no ', 'na ', 'cidade de ', 'município de ', 'em cidade de ', 'na cidade de ']
            for indicator in city_indicators:
                if indicator in query_lower:
                    # Buscar por cidades após indicadores
                    idx = query_lower.find(indicator)
                    remaining_text = query_lower[idx + len(indicator):]
                    words = remaining_text.split()
                    
                    # Lista expandida de cidades conhecidas (normalized for matching)
                    known_cities = {
                        'joinville': 'Joinville',
                        'curitiba': 'Curitiba',
                        'florianopolis': 'Florianópolis', 'florianópolis': 'Florianópolis',
                        'sao paulo': 'São Paulo', 'são paulo': 'São Paulo',
                        'rio de janeiro': 'Rio de Janeiro',
                        'blumenau': 'Blumenau',
                        'itajai': 'Itajaí', 'itajaí': 'Itajaí',
                        'porto alegre': 'Porto Alegre',
                        'brasilia': 'Brasília', 'brasília': 'Brasília',
                        'belo horizonte': 'Belo Horizonte',
                        'salvador': 'Salvador',
                        'fortaleza': 'Fortaleza',
                        'recife': 'Recife',
                        'manaus': 'Manaus',
                        'belem': 'Belém', 'belém': 'Belém',
                        'goiania': 'Goiânia', 'goiânia': 'Goiânia',
                        'campinas': 'Campinas',
                        'santos': 'Santos',
                        'sorocaba': 'Sorocaba',
                        'londrina': 'Londrina',
                        'maringa': 'Maringá', 'maringá': 'Maringá',
                        'cascavel': 'Cascavel',
                        'foz do iguacu': 'Foz do Iguaçu', 'foz do iguaçu': 'Foz do Iguaçu'
                    }
                    
                    # Buscar matches de cidades simples e compostas
                    matched_city = None
                    
                    # Tentar match de nomes compostos primeiro (2-3 palavras)
                    for i in range(min(3, len(words))):
                        for j in range(i+1, min(i+4, len(words)+1)):  # Tentar até 3 palavras
                            candidate = ' '.join(words[i:j]).strip(',?.!')
                            if candidate in known_cities:
                                matched_city = known_cities[candidate]
                                break
                        if matched_city:
                            break
                    
                    # Se não encontrou nome composto, tentar nomes simples
                    if not matched_city:
                        for word in words[:3]:  # Check first 3 words after indicator
                            word = word.strip(',?.!')
                            if word in known_cities:
                                matched_city = known_cities[word]
                                break
                    
                    if matched_city:
                        explicit_changes['Municipio_Cliente'] = matched_city
                        break
            
            # === DETECÇÃO TEMPORAL AVANÇADA ===
            # Usar o novo sistema de parsing temporal do TextNormalizer
            temporal_result = self.normalizer.extract_and_format_temporal(query)
            
            if temporal_result:
                temporal_context, sql_filter = temporal_result
                
                # Aplicar as entidades temporais detectadas
                for key, value in temporal_context.items():
                    explicit_changes[key] = value
            
            # === DETECÇÃO DE ESTADOS/UF ===
            # Detectar mudanças explícitas de UF/Estado
            uf_indicators = ['em ', 'no estado de ', 'estado de ', 'na uf ', 'uf ', 'no ', 'na ']
            uf_mapping = {
                'sc': 'SC', 'santa catarina': 'SC',
                'sp': 'SP', 'sao paulo': 'SP', 'são paulo': 'SP',
                'rj': 'RJ', 'rio de janeiro': 'RJ',
                'pr': 'PR', 'parana': 'PR', 'paraná': 'PR',
                'rs': 'RS', 'rio grande do sul': 'RS',
                'mg': 'MG', 'minas gerais': 'MG',
                'go': 'GO', 'goias': 'GO', 'goiás': 'GO',
                'df': 'DF', 'distrito federal': 'DF', 'brasilia': 'DF', 'brasília': 'DF'
            }
            
            for indicator in uf_indicators:
                if indicator in query_lower:
                    idx = query_lower.find(indicator)
                    remaining_text = query_lower[idx + len(indicator):]
                    words = remaining_text.split()
                    
                    # Tentar match de UF
                    for i in range(min(4, len(words))):
                        for j in range(i+1, min(i+4, len(words)+1)):
                            candidate = ' '.join(words[i:j]).strip(',?.!')
                            if candidate in uf_mapping:
                                explicit_changes['UF_Cliente'] = uf_mapping[candidate]
                                break
                        if 'UF_Cliente' in explicit_changes:
                            break
                    break
            
            # === DETECÇÃO DE PRODUTOS/SEGMENTOS ===
            # Detectar mudanças explícitas em produtos ou segmentos
            product_indicators = ['produto ', 'segmento ', 'categoria ', 'linha de produto ']
            for indicator in product_indicators:
                if indicator in query_lower:
                    idx = query_lower.find(indicator)
                    remaining_text = query_lower[idx + len(indicator):]
                    # Extrair possível nome do produto (próximas 2-3 palavras)
                    words = remaining_text.split()[:3]
                    if words:
                        product_name = ' '.join(words).strip(',?.!')
                        if len(product_name) > 2:  # Nome válido
                            explicit_changes['Des_Linha_Produto'] = product_name.title()
                    break
            
            # === TRATAMENTO DE EXPRESSÕES ESPECIAIS ===
            # Detectar expressões como "mesmo período", "mesmo tempo" - preservar contexto temporal
            same_period_expressions = ['mesmo período', 'mesmo tempo', 'mesma época', 'mesmo mês']
            if any(expr in query_lower for expr in same_period_expressions):
                # Não alterar dados temporais - preservar o que está no contexto persistente
                # Remove qualquer alteração temporal que possa ter sido detectada
                temporal_keys = ['Data_>=', 'Data_<', 'Data']
                for key in temporal_keys:
                    if key in explicit_changes:
                        del explicit_changes[key]
            
            # === DETECÇÃO DE NEGAÇÃO/LIMPEZA ===
            # Detectar quando usuário quer limpar filtros específicos
            clear_expressions = ['sem filtro', 'todos os', 'geral', 'no geral', 'removendo o filtro']
            if any(expr in query_lower for expr in clear_expressions):
                # Marcar campos para limpeza (valor especial)
                if 'municipio' in query_lower or 'cidade' in query_lower:
                    explicit_changes['Municipio_Cliente'] = '__CLEAR__'
                if 'data' in query_lower or 'período' in query_lower or 'tempo' in query_lower:
                    explicit_changes['Data_>='] = '__CLEAR__'
                    explicit_changes['Data_<'] = '__CLEAR__'
            
            return explicit_changes

        def detect_comparative_query(self, query: str) -> dict:
            """
            Detecta se a query é uma consulta comparativa e determina o tipo de comparação necessária.
            
            Args:
                query: Query do usuário a ser analisada
                
            Returns:
                dict: Informações sobre a natureza comparativa da consulta
            """
            query_lower = query.lower()
            
            comparative_info = {
                'is_comparative': False,
                'comparison_type': None,
                'requires_expansion': False,
                'temporal_scope': 'single',
                'dimensional_scope': 'single',
                'calculation_type': None,
                'confidence_score': 0,
                'detected_patterns': []
            }
            
            # === DETECÇÃO DE PADRÕES COMPARATIVOS ===
            
            # 1. PADRÕES TEMPORAIS COMPARATIVOS
            temporal_comparative_patterns = [
                (r'\bentre\s+(\w+)\s+e\s+(\w+)', 'temporal_range', 40),  # "entre junho e julho"
                (r'\bde\s+(\w+)\s+a\s+(\w+)', 'temporal_range', 40),    # "de janeiro a março"
                (r'\b(\w+)\s+vs\s+(\w+)', 'temporal_vs', 45),           # "junho vs julho"
                (r'\b(\w+)\s+versus\s+(\w+)', 'temporal_vs', 45),       # "janeiro versus fevereiro"
                (r'\bcomparado\s+com\s+(\w+)', 'temporal_comparison', 35), # "comparado com mês anterior"
                (r'\bem\s+relação\s+a\s+(\w+)', 'temporal_relation', 30), # "em relação ao período anterior"
                (r'\bmês\s+a\s+mês', 'period_evolution', 50),           # "crescimento mês a mês"
                (r'\bano\s+a\s+ano', 'period_evolution', 50),           # "evolução ano a ano"
                (r'\btrimestre\s+a\s+trimestre', 'period_evolution', 45), # "variação trimestre a trimestre"
            ]
            
            # 2. PADRÕES DE CRESCIMENTO/VARIAÇÃO
            growth_patterns = [
                (r'\bcresceram', 'growth_analysis', 50),                 # "clientes que mais cresceram"
                (r'\bcrescimento', 'growth_analysis', 45),               # "crescimento de vendas"
                (r'\bvariação', 'variation_analysis', 40),               # "variação percentual"
                (r'\baumento', 'growth_analysis', 40),                   # "aumento nas vendas"
                (r'\bredução', 'decline_analysis', 40),                  # "redução de custos"
                (r'\bevolução', 'evolution_analysis', 35),               # "evolução temporal"
                (r'\bmelhora', 'improvement_analysis', 35),              # "melhora na performance"
                (r'\bpiora', 'decline_analysis', 35),                    # "piora nos resultados"
                (r'\bdesempenho', 'performance_analysis', 30),           # "análise de desempenho"
            ]
            
            # 3. PADRÕES DE COMPARAÇÃO DIMENSIONAL
            dimensional_patterns = [
                (r'\bcompare\s+(\w+)', 'dimensional_comparison', 45),    # "compare produtos"
                (r'\bversus\s+(\w+)', 'dimensional_vs', 45),            # "produto A versus B"
                (r'\bcontra\s+(\w+)', 'dimensional_vs', 40),            # "região Sul contra Norte"
                (r'\bmelhor\s+que', 'performance_comparison', 35),       # "melhor que o concorrente"
                (r'\bpior\s+que', 'performance_comparison', 35),         # "pior que o esperado"
                (r'\bmais\s+que', 'quantitative_comparison', 30),        # "mais que o ano passado"
                (r'\bmenos\s+que', 'quantitative_comparison', 30),       # "menos que o previsto"
            ]
            
            # 4. PADRÕES QUE INDICAM NECESSIDADE DE MÚLTIPLOS PERÍODOS
            multi_period_indicators = [
                (r'\bentre.*?\be', 'period_range_required', 45),         # "entre X e Y"
                (r'\bde.*?a.*?\bde', 'extended_period_required', 40),    # "de janeiro a dezembro de 2015"
                (r'\bno\s+período\s+de', 'period_analysis_required', 35), # "no período de 2014-2015"
                (r'\bao\s+longo\s+de', 'timeline_analysis_required', 35), # "ao longo de 2015"
                (r'\bdurante\s+o', 'duration_analysis_required', 30),    # "durante o trimestre"
            ]
            
            # === ANÁLISE DOS PADRÕES ===
            all_patterns = [
                (temporal_comparative_patterns, 'temporal'),
                (growth_patterns, 'growth'),
                (dimensional_patterns, 'dimensional'),
                (multi_period_indicators, 'multi_period')
            ]
            
            import re
            
            for pattern_group, category in all_patterns:
                for pattern, detection_type, score in pattern_group:
                    matches = re.finditer(pattern, query_lower)
                    for match in matches:
                        comparative_info['confidence_score'] += score
                        comparative_info['detected_patterns'].append({
                            'pattern': pattern,
                            'type': detection_type,
                            'category': category,
                            'match': match.group(0),
                            'score': score
                        })
                        
                        # Configurar tipo de comparação baseado na detecção
                        if category == 'temporal' and not comparative_info['comparison_type']:
                            comparative_info['comparison_type'] = detection_type
                            comparative_info['temporal_scope'] = 'multiple'
                            
                        elif category == 'growth' and not comparative_info['calculation_type']:
                            comparative_info['calculation_type'] = detection_type
                            
                        elif category == 'multi_period':
                            comparative_info['requires_expansion'] = True
                            comparative_info['temporal_scope'] = 'multiple'
            
            # === HEURÍSTICAS ESPECIAIS ===
            
            # Heurística 1: Palavras que quase sempre indicam comparação
            strong_comparative_words = ['cresceram', 'versus', 'comparar', 'entre', 'variação']
            strong_matches = sum(1 for word in strong_comparative_words if word in query_lower)
            if strong_matches > 0:
                comparative_info['confidence_score'] += strong_matches * 20
            
            # Heurística 2: Presença de múltiplas entidades temporais
            month_names = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho',
                          'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']
            temporal_entities_count = sum(1 for month in month_names if month in query_lower)
            if temporal_entities_count >= 2:
                comparative_info['confidence_score'] += 30
                comparative_info['temporal_scope'] = 'multiple'
                comparative_info['requires_expansion'] = True
            
            # Heurística 3: Padrões de pergunta que implicam comparação temporal
            implicit_temporal_patterns = [
                'que mais', 'que menos', 'melhores', 'piores', 'maior crescimento', 'menor queda'
            ]
            for pattern in implicit_temporal_patterns:
                if pattern in query_lower:
                    comparative_info['confidence_score'] += 15
            
            # === CLASSIFICAÇÃO FINAL ===
            COMPARATIVE_THRESHOLD = 40
            
            if comparative_info['confidence_score'] >= COMPARATIVE_THRESHOLD:
                comparative_info['is_comparative'] = True
                
                # Determinar tipo principal se não foi definido
                if not comparative_info['comparison_type']:
                    if comparative_info['calculation_type']:
                        comparative_info['comparison_type'] = comparative_info['calculation_type']
                    elif comparative_info['temporal_scope'] == 'multiple':
                        comparative_info['comparison_type'] = 'temporal_analysis'
                    else:
                        comparative_info['comparison_type'] = 'general_comparative'
            
            return comparative_info

        def expand_filters_for_comparison(self, user_query: str, comparative_info: dict, explicit_changes: dict) -> dict:
            """
            Expande filtros automaticamente para suportar consultas comparativas sem interrupção do usuário.
            
            Args:
                user_query: Query original do usuário
                comparative_info: Informações sobre natureza comparativa da query
                explicit_changes: Mudanças explícitas detectadas
                
            Returns:
                dict: Filtros expandidos para análise comparativa
            """
            expanded_filters = explicit_changes.copy()
            
            if not comparative_info['is_comparative']:
                return expanded_filters
            
            # === EXPANSÃO BASEADA NO TIPO DE COMPARAÇÃO ===
            
            if comparative_info['requires_expansion'] and comparative_info['temporal_scope'] == 'multiple':
                # Para consultas que exigem múltiplos períodos
                temporal_expansion = self._expand_temporal_filters(user_query, comparative_info)
                expanded_filters.update(temporal_expansion)
            
            # === EXPANSÃO PARA ANÁLISES DE CRESCIMENTO ===
            if comparative_info['calculation_type'] in ['growth_analysis', 'variation_analysis']:
                growth_expansion = self._expand_for_growth_analysis(user_query, comparative_info)
                expanded_filters.update(growth_expansion)
            
            # === REMOÇÃO DE FILTROS RESTRITIVOS PARA COMPARAÇÕES ===
            restrictive_removals = self._remove_restrictive_filters_for_comparison(
                comparative_info, expanded_filters
            )
            expanded_filters.update(restrictive_removals)
            
            # === MARCAÇÃO ESPECIAL PARA CONSULTAS COMPARATIVAS ===
            expanded_filters['_comparison_type'] = comparative_info['comparison_type']
            expanded_filters['_requires_multiple_contexts'] = comparative_info['temporal_scope'] == 'multiple'
            expanded_filters['_calculation_required'] = comparative_info['calculation_type'] is not None
            
            return expanded_filters

        def _expand_temporal_filters(self, query: str, comparative_info: dict) -> dict:
            """
            Expande filtros temporais para suportar análises comparativas multi-período.
            
            Args:
                query: Query do usuário
                comparative_info: Informações comparativas
                
            Returns:
                dict: Filtros temporais expandidos
            """
            temporal_filters = {}
            
            # Usar o sistema existente de parsing temporal
            temporal_result = self.normalizer.extract_and_format_temporal(query)
            
            if temporal_result:
                temporal_context, sql_filter = temporal_result
                
                # Se detectou período entre meses, expandir para incluir ambos completamente
                if 'Data_>=' in temporal_context and 'Data_<' in temporal_context:
                    # Para consultas comparativas, não aplicar filtro restritivo único
                    # Em vez disso, marcar que múltiplos períodos são necessários
                    temporal_filters['_temporal_range_start'] = temporal_context['Data_>=']
                    temporal_filters['_temporal_range_end'] = temporal_context['Data_<']
                    temporal_filters['_expand_temporal_analysis'] = True
                else:
                    # Aplicar filtros temporais normalmente
                    temporal_filters.update(temporal_context)
            
            # === DETECÇÃO DE PADRÕES ESPECIAIS DE EXPANSÃO TEMPORAL ===
            query_lower = query.lower()
            
            # Padrão: "entre X e Y" - garantir que ambos os períodos são incluídos
            import re
            between_pattern = r'entre\s+(\w+)\s+e\s+(\w+)\s+de\s+(\d{4})'
            between_match = re.search(between_pattern, query_lower)
            
            if between_match:
                start_month = between_match.group(1)
                end_month = between_match.group(2) 
                year = between_match.group(3)
                
                # Marcar que análise comparativa entre períodos é necessária
                temporal_filters['_comparative_period_analysis'] = True
                temporal_filters['_start_period'] = f"{start_month} {year}"
                temporal_filters['_end_period'] = f"{end_month} {year}"
                temporal_filters['_requires_period_comparison'] = True
            
            return temporal_filters

        def _expand_for_growth_analysis(self, query: str, comparative_info: dict) -> dict:
            """
            Configura filtros especiais para análises de crescimento que requerem múltiplos períodos.
            
            Args:
                query: Query do usuário
                comparative_info: Informações comparativas
                
            Returns:
                dict: Filtros específicos para análise de crescimento
            """
            growth_filters = {}
            
            # === DETECÇÃO DE PADRÕES DE CRESCIMENTO QUE REQUEREM MÚLTIPLOS PERÍODOS ===
            query_lower = query.lower()
            
            if 'cresceram' in query_lower or 'crescimento' in query_lower:
                # Análise de crescimento sempre requer pelo menos dois períodos
                growth_filters['_requires_growth_calculation'] = True
                growth_filters['_growth_type'] = 'client_growth' if 'client' in query_lower else 'general_growth'
                
                # Se não há contexto temporal explícito, usar períodos sequenciais
                if not any(key.startswith('Data') for key in growth_filters):
                    growth_filters['_auto_generate_comparison_periods'] = True
            
            if 'variação' in query_lower:
                growth_filters['_requires_variation_calculation'] = True
                growth_filters['_variation_type'] = 'percentage'  # Padrão: percentual
            
            # === PADRÕES ESPECÍFICOS DE COMPARAÇÃO TEMPORAL ===
            if any(pattern in query_lower for pattern in ['mês a mês', 'trimestre a trimestre', 'ano a ano']):
                growth_filters['_requires_period_evolution'] = True
                
                if 'mês a mês' in query_lower:
                    growth_filters['_evolution_granularity'] = 'monthly'
                elif 'trimestre' in query_lower:
                    growth_filters['_evolution_granularity'] = 'quarterly'  
                elif 'ano a ano' in query_lower:
                    growth_filters['_evolution_granularity'] = 'yearly'
            
            return growth_filters

        def _remove_restrictive_filters_for_comparison(self, comparative_info: dict, current_filters: dict) -> dict:
            """
            Remove filtros que seriam muito restritivos para análises comparativas.
            
            Args:
                comparative_info: Informações sobre a natureza comparativa
                current_filters: Filtros atuais
                
            Returns:
                dict: Filtros de remoção/modificação
            """
            filter_adjustments = {}
            
            # === LÓGICA DE REMOÇÃO BASEADA NO TIPO DE COMPARAÇÃO ===
            
            if comparative_info['comparison_type'] in ['temporal_range', 'growth_analysis']:
                # Para análises temporais, remover filtros de data únicos que limitariam a comparação
                if 'Data' in current_filters and not current_filters.get('_expand_temporal_analysis'):
                    # Marcar filtro de data única para expansão
                    filter_adjustments['_override_single_date_filter'] = True
            
            if comparative_info['calculation_type'] in ['growth_analysis', 'variation_analysis']:
                # Para crescimento, não restringir a apenas um período
                filter_adjustments['_allow_multi_period_analysis'] = True
            
            # === PRESERVAÇÃO DE FILTROS GEOGRÁFICOS ===
            # Filtros geográficos (município, UF) devem ser mantidos para comparações
            # pois definem o escopo da análise
            geographic_filters = ['Municipio_Cliente', 'UF_Cliente']
            for geo_filter in geographic_filters:
                if geo_filter in current_filters:
                    filter_adjustments[f'_preserve_{geo_filter}'] = current_filters[geo_filter]
            
            # === CONFIGURAÇÕES ESPECIAIS PARA SQL GENERATION ===
            if comparative_info['is_comparative']:
                filter_adjustments['_enable_comparative_mode'] = True
                filter_adjustments['_disable_restrictive_filtering'] = True
            
            return filter_adjustments
        
        def inject_context_into_query(self, user_query: str, explicit_changes: dict, comparative_info: dict = None) -> str:
            """
            Injeta contexto com suporte especializado para consultas comparativas autônomas.
            
            Args:
                user_query: Query original do usuário
                explicit_changes: Mudanças explícitas detectadas na query
                comparative_info: Informações sobre natureza comparativa (opcional)
                
            Returns:
                str: Query modificada com contexto e instruções comparativas
            """
            # === PROCESSAMENTO DE CONTEXTO ===
            # Mesclar contexto atual com mudanças explícitas
            active_context = self.persistent_context.copy()
            
            # Processar mudanças explícitas, incluindo limpeza de filtros
            for key, value in explicit_changes.items():
                if value == '__CLEAR__':
                    # Remover o filtro do contexto ativo
                    active_context.pop(key, None)
                else:
                    # Aplicar mudança normal
                    active_context[key] = value
            
            # Se não há contexto para aplicar, retornar query original
            if not active_context:
                return user_query
            
            # === CLASSIFICAÇÃO DO TIPO DE CONSULTA ===
            query_lower = user_query.lower()
            
            # Determinar tipo de consulta para instruções mais precisas
            query_type = "general"
            if any(word in query_lower for word in ['top', 'ranking', 'maior', 'menor', 'primeiro', 'último']):
                query_type = "ranking"
            elif any(word in query_lower for word in ['total', 'soma', 'somar', 'contar', 'count']):
                query_type = "aggregation"
            elif any(word in query_lower for word in ['compare', 'comparar', 'versus', 'vs', 'diferença']):
                query_type = "comparison"
            elif any(word in query_lower for word in ['crescimento', 'variação', 'aumento', 'redução', 'evolução']):
                query_type = "growth"
            elif any(word in query_lower for word in ['detalhe', 'detalhes', 'lista', 'listar', 'mostrar']):
                query_type = "detailed"
            
            # === OVERRIDE PARA CONSULTAS COMPARATIVAS ===
            if comparative_info and comparative_info.get('is_comparative'):
                if comparative_info.get('calculation_type') == 'growth_analysis':
                    query_type = "comparative_growth"
                elif comparative_info.get('comparison_type') == 'temporal_range':
                    query_type = "comparative_temporal"
                elif comparative_info.get('requires_expansion'):
                    query_type = "comparative_expanded"
                else:
                    query_type = "comparative_general"
            
            # === CONSTRUÇÃO DE INSTRUÇÕES CONTEXTUAIS ===
            context_instructions = []
            
            # Cabeçalho adaptado ao tipo de consulta
            if query_type == "ranking":
                context_instructions.append("CONTEXTO ESSENCIAL: Para esta análise de ranking/top N, aplique OBRIGATORIAMENTE os filtros:")
            elif query_type == "aggregation":
                context_instructions.append("CONTEXTO ESSENCIAL: Para esta agregação/totalização, aplique OBRIGATORIAMENTE os filtros:")
            elif query_type == "comparison":
                context_instructions.append("CONTEXTO ESSENCIAL: Para esta comparação, aplique OBRIGATORIAMENTE os filtros:")
            elif query_type == "growth":
                context_instructions.append("CONTEXTO ESSENCIAL: Para esta análise de crescimento/evolução, aplique OBRIGATORIAMENTE os filtros:")
            
            # === CABEÇALHOS ESPECÍFICOS PARA CONSULTAS COMPARATIVAS ===
            elif query_type == "comparative_growth":
                context_instructions.append("ANÁLISE COMPARATIVA DE CRESCIMENTO AUTOMÁTICA:")
                context_instructions.append("Execute consulta que permita calcular crescimento/variação entre períodos SEM pedir confirmação.")
            elif query_type == "comparative_temporal":
                context_instructions.append("ANÁLISE COMPARATIVA TEMPORAL AUTOMÁTICA:")
                context_instructions.append("Expanda automaticamente para incluir TODOS os períodos necessários para comparação.")
            elif query_type == "comparative_expanded":
                context_instructions.append("ANÁLISE COMPARATIVA COM EXPANSÃO AUTOMÁTICA:")
                context_instructions.append("NÃO aplique filtros restritivos que limitem a análise comparativa solicitada.")
            elif query_type == "comparative_general":
                context_instructions.append("ANÁLISE COMPARATIVA AUTOMÁTICA:")
                context_instructions.append("Execute consulta comparativa sem interromper o usuário para confirmações.")
            
            else:
                context_instructions.append("CONTEXTO IMPORTANTE: Ao executar consultas SQL, SEMPRE aplique os seguintes filtros:")
            
            # === FORMATAÇÃO DE FILTROS POR TIPO ===
            filter_instructions = []
            
            for key, value in active_context.items():
                if key == "Municipio_Cliente":
                    filter_instructions.append(f"- MUNICÍPIO: LOWER(Municipio_Cliente) = '{value.lower()}'")
                
                elif key in ["Data_>=", "Data_<"]:
                    # Agrupar filtros temporais 
                    if key == "Data_>=" and "Data_<" in active_context:
                        start_date = value
                        end_date = active_context["Data_<"]
                        filter_instructions.append(f"- PERÍODO: Data >= '{start_date}' AND Data < '{end_date}'")
                    elif key == "Data_>=" and "Data_<" not in active_context:
                        filter_instructions.append(f"- DATA INICIAL: Data >= '{value}'")
                
                elif key == "Data" and not key.startswith("Data_"):
                    # Filtro de data legado
                    if ">=" in str(value) and "<" in str(value):
                        filter_instructions.append(f"- PERÍODO: Data {value}")
                    else:
                        filter_instructions.append(f"- DATA: Data = '{value}'")
                
                elif key == "UF_Cliente":
                    filter_instructions.append(f"- ESTADO/UF: LOWER(UF_Cliente) = '{value.lower()}'")
                
                elif key == "Des_Linha_Produto":
                    filter_instructions.append(f"- PRODUTO: LOWER(Des_Linha_Produto) LIKE '%{value.lower()}%'")
                
                elif key.endswith("_IN"):
                    column_name = key.replace("_IN", "")
                    filter_instructions.append(f"- {column_name.upper()}: usar cláusula IN com valores apropriados")
                
                elif key.startswith("_"):
                    # Ignorar metadados
                    continue
                
                else:
                    # Filtros genéricos
                    if isinstance(value, str) and len(value) > 0:
                        filter_instructions.append(f"- {key.upper()}: LOWER({key}) = '{value.lower()}'")
            
            # Remover duplicatas mantendo ordem
            unique_filters = []
            seen_filters = set()
            for filter_inst in filter_instructions:
                if filter_inst not in seen_filters:
                    unique_filters.append(filter_inst)
                    seen_filters.add(filter_inst)
            
            context_instructions.extend(unique_filters)
            
            # === INSTRUÇÕES FINAIS ESPECÍFICAS POR TIPO ===
            if query_type == "ranking":
                context_instructions.append("\nIMPORTANTE: Estes filtros são OBRIGATÓRIOS para o ranking. Aplique-os ANTES de calcular o TOP N.")
            elif query_type == "aggregation":
                context_instructions.append("\nIMPORTANTE: Aplique estes filtros ANTES de fazer as agregações (SUM, COUNT, AVG, etc.).")
            elif query_type == "comparison":
                context_instructions.append("\nIMPORTANTE: Aplique estes filtros em TODOS os lados da comparação para consistência.")
            
            # === INSTRUÇÕES FINAIS ESPECÍFICAS PARA CONSULTAS COMPARATIVAS ===
            elif query_type == "comparative_growth":
                context_instructions.append("\nANÁLISE AUTOMÁTICA DE CRESCIMENTO:")
                context_instructions.append("- Execute consulta que retorne dados agrupados por período (mensal/trimestral)")
                context_instructions.append("- NÃO pergunte confirmação para expandir período - faça automaticamente")
                context_instructions.append("- Calcule percentuais de crescimento período a período quando possível")
                context_instructions.append("- Estruture resultado para permitir identificação de tendências")
            
            elif query_type == "comparative_temporal":
                context_instructions.append("\nEXPANSÃO TEMPORAL AUTOMÁTICA:")
                context_instructions.append("- INCLUA automaticamente TODOS os períodos necessários (junho E julho, não apenas um)")
                context_instructions.append("- NÃO aplique filtro de data única que limitaria a comparação")
                context_instructions.append("- Agrupe resultados por período para facilitar comparação")
                context_instructions.append("- Mantenha filtros geográficos mas expanda temporais")
            
            elif query_type == "comparative_expanded":
                context_instructions.append("\nMODO COMPARATIVO EXPANDIDO:")
                context_instructions.append("- REMOVA filtros que limitem a análise comparativa solicitada")
                context_instructions.append("- PRESERVE apenas filtros essenciais (geográficos, produtos específicos)")
                context_instructions.append("- EXPANDA escopo temporal automaticamente conforme necessário")
                context_instructions.append("- Execute sem interrupções ou confirmações do usuário")
            
            elif query_type == "comparative_general":
                context_instructions.append("\nANÁLISE COMPARATIVA AUTÔNOMA:")
                context_instructions.append("- Execute comparação solicitada SEM pedir confirmação")
                context_instructions.append("- Aplique lógica inteligente para determinar escopo da análise")
                context_instructions.append("- Retorne resultados estruturados para facilitar comparação")
            
            else:
                context_instructions.append("\nESSES FILTROS SÃO OBRIGATÓRIOS e devem ser incluídos em TODAS as consultas SQL desta pergunta.")
            
            # === VALIDAÇÃO DE CONSISTÊNCIA ===
            # Adicionar nota sobre consistência se há múltiplos filtros
            if len(unique_filters) > 1:
                context_instructions.append("ATENÇÃO: Combine todos os filtros com AND na cláusula WHERE.")
            
            context_instructions.append("\n--- Pergunta original do usuário ---")
            
            # === CONSTRUÇÃO DA QUERY FINAL ===
            context_prefix = "\n".join(context_instructions)
            enhanced_query = f"{context_prefix}\n{user_query}"
            
            return enhanced_query

        def auto_substitute_parameters(self, user_query: str, explicit_changes: dict) -> tuple:
            """
            Sistema de substituição automática de parâmetros sem necessidade de confirmação.
            Detecta quando o usuário quer alterar filtros e aplica automaticamente.
            
            Args:
                user_query: Query original do usuário
                explicit_changes: Mudanças detectadas na query
                
            Returns:
                tuple: (should_auto_substitute: bool, substitution_summary: str)
            """
            query_lower = user_query.lower()
            
            # === DETECÇÃO DE PADRÕES DE SUBSTITUIÇÃO ===
            substitution_patterns = {
                # Padrões de mudança de localização
                'location_change': [
                    r'\be em (\w+)',                    # "E em Curitiba?"
                    r'\bagora em (\w+)',                # "Agora em Porto Alegre"  
                    r'\bpara (\w+)',                    # "Para Joinville"
                    r'\btambém em (\w+)',               # "Também em Blumenau"
                    r'\bmudar para (\w+)',              # "Mudar para São Paulo"
                ],
                
                # Padrões de mudança temporal
                'temporal_change': [
                    r'\bem (\w+) de (\d{4})',          # "em julho de 2015"
                    r'\bno período de (\w+)',           # "no período de agosto"
                    r'\bagora em (\w+)',                # "agora em setembro"  
                    r'\bentre (\w+) e (\w+)',           # "entre junho e julho"
                    r'\bdurante (\w+)',                 # "durante dezembro"
                ],
                
                # Padrões de mudança de escopo
                'scope_change': [
                    r'\bpara o estado de (\w+)',        # "para o estado de SC"
                    r'\bno estado (\w+)',               # "no estado RS"
                    r'\bna região (\w+)',               # "na região Sul"
                    r'\bpara a uf (\w+)',               # "para a UF PR"
                ],
                
                # Padrões que indicam comparação/contraste
                'comparison': [
                    r'\be (\w+)\?',                     # "E Curitiba?"
                    r'\bversus (\w+)',                  # "versus Porto Alegre"
                    r'\bcomparado com (\w+)',           # "comparado com Joinville"
                    r'\bem relação a (\w+)',            # "em relação a setembro"
                ]
            }
            
            # === ANÁLISE DE CONTEXTO ATUAL ===
            has_persistent_context = bool(self.persistent_context)
            has_explicit_changes = bool(explicit_changes)
            
            # === SCORING DE CONFIANÇA ===
            confidence_score = 0
            substitution_reasons = []
            
            # Pontuação por padrões detectados
            for pattern_type, patterns in substitution_patterns.items():
                for pattern in patterns:
                    import re
                    if re.search(pattern, query_lower):
                        if pattern_type == 'location_change':
                            confidence_score += 30
                            substitution_reasons.append(f"Detectado padrão de mudança de localização: {pattern}")
                        elif pattern_type == 'temporal_change':
                            confidence_score += 25  
                            substitution_reasons.append(f"Detectado padrão de mudança temporal: {pattern}")
                        elif pattern_type == 'scope_change':
                            confidence_score += 20
                            substitution_reasons.append(f"Detectado padrão de mudança de escopo: {pattern}")
                        elif pattern_type == 'comparison':
                            confidence_score += 35  # Alta confiança para comparações
                            substitution_reasons.append(f"Detectado padrão de comparação: {pattern}")
            
            # Bonificação se há contexto persistente (cenário típico de substituição)
            if has_persistent_context:
                confidence_score += 15
                substitution_reasons.append("Contexto persistente existe (cenário de substituição)")
            
            # Bonificação se detectou mudanças explícitas
            if has_explicit_changes:
                confidence_score += 20
                substitution_reasons.append(f"Mudanças explícitas detectadas: {list(explicit_changes.keys())}")
            
            # === HEURÍSTICAS ESPECIAIS ===
            
            # Heurística 1: Query muito curta com mudança explícita = alta probabilidade de substituição
            if len(query_lower.split()) <= 4 and has_explicit_changes:
                confidence_score += 25
                substitution_reasons.append("Query curta com mudança explícita (provável substituição)")
            
            # Heurística 2: Presença de pronomes interrogativos + contexto = substituição
            question_words = ['qual', 'quais', 'como', 'quando', 'onde', 'quanto', 'quantos']
            if any(word in query_lower for word in question_words) and has_persistent_context:
                confidence_score += 10
                substitution_reasons.append("Pergunta com contexto existente")
            
            # Heurística 3: Palavras que indicam continuidade mas com mudança
            continuity_words = ['também', 'agora', 'então', 'depois', 'em seguida']
            if any(word in query_lower for word in continuity_words) and has_explicit_changes:
                confidence_score += 15
                substitution_reasons.append("Indicadores de continuidade com mudança")
            
            # === DECISÃO DE SUBSTITUIÇÃO ===
            # Limiar de confiança para substituição automática
            SUBSTITUTION_THRESHOLD = 40
            
            should_substitute = confidence_score >= SUBSTITUTION_THRESHOLD
            
            # === CONSTRUÇÃO DO RESUMO ===
            if should_substitute:
                substitution_summary = f"SUBSTITUIÇÃO AUTOMÁTICA (confiança: {confidence_score}%):\n"
                
                # Listar mudanças que serão aplicadas
                changes_list = []
                for key, value in explicit_changes.items():
                    if key == "Municipio_Cliente":
                        old_value = self.persistent_context.get(key, "N/A")
                        changes_list.append(f"  Município: {old_value} → {value}")
                    elif key in ["Data_>=", "Data_<"]:
                        changes_list.append(f"  Período: alteração temporal detectada")
                    elif key == "UF_Cliente":
                        old_value = self.persistent_context.get(key, "N/A")
                        changes_list.append(f"  Estado: {old_value} → {value}")
                    else:
                        old_value = self.persistent_context.get(key, "N/A")
                        changes_list.append(f"  {key}: {old_value} → {value}")
                
                if changes_list:
                    substitution_summary += "Mudanças aplicadas:\n" + "\n".join(changes_list)
                else:
                    substitution_summary += "Contexto será mantido sem alterações"
                    
                substitution_summary += f"\n\nRazões da substituição automática:\n- " + "\n- ".join(substitution_reasons[:3])
            else:
                substitution_summary = f"Substituição não aplicada (confiança: {confidence_score}% < {SUBSTITUTION_THRESHOLD}%)"
                if substitution_reasons:
                    substitution_summary += f"\nRazões parciais detectadas:\n- " + "\n- ".join(substitution_reasons[:2])
            
            return should_substitute, substitution_summary

        def merge_contexts(self, new_context: dict, explicit_changes: dict, comparative_info: dict = None) -> dict:
            """
            Mescla contextos com lógica inteligente de priorização e suporte para consultas comparativas.
            
            Args:
                new_context: Contexto extraído da query SQL atual
                explicit_changes: Mudanças explicitamente solicitadas pelo usuário
                comparative_info: Informações sobre natureza comparativa da consulta
            
            Returns:
                dict: Contexto mesclado final otimizado para análise comparativa
            """
            # === INICIALIZAÇÃO ===
            merged_context = self.persistent_context.copy()
            context_metadata = {
                'merge_timestamp': pd.Timestamp.now().isoformat(),
                'merge_operations': [],
                'conflicts_resolved': [],
                'context_age': {}
            }
            
            # === PROCESSAMENTO DE MUDANÇAS EXPLÍCITAS (ALTA PRIORIDADE) ===
            for key, value in explicit_changes.items():
                if value == '__CLEAR__':
                    # Remover completamente do contexto
                    if key in merged_context:
                        old_value = merged_context.pop(key)
                        context_metadata['merge_operations'].append(f"CLEARED: {key} (era: {old_value})")
                else:
                    # Aplicar mudança e registrar operação
                    old_value = merged_context.get(key, "N/A")
                    merged_context[key] = value
                    context_metadata['merge_operations'].append(f"EXPLICIT: {key} {old_value} → {value}")
            
            # === PROCESSAMENTO ESPECIAL PARA CONSULTAS COMPARATIVAS ===
            if comparative_info and comparative_info.get('is_comparative'):
                context_metadata['is_comparative_context'] = True
                context_metadata['comparison_type'] = comparative_info.get('comparison_type')
                
                # Para consultas comparativas, aplicar lógica especial de contexto
                if comparative_info.get('requires_expansion'):
                    # Não aplicar filtros temporais restritivos únicos
                    if comparative_info.get('temporal_scope') == 'multiple':
                        # Preservar informações temporais especiais para expansão
                        temporal_expansion_keys = [k for k in explicit_changes.keys() if k.startswith('_temporal_') or k.startswith('_comparative_')]
                        for key in temporal_expansion_keys:
                            merged_context[key] = explicit_changes[key]
                            context_metadata['merge_operations'].append(f"COMPARATIVE_EXPANSION: {key}")
                
                # Preservar filtros geográficos para comparações (mais importantes que temporais)
                geographic_keys = ['Municipio_Cliente', 'UF_Cliente']
                for geo_key in geographic_keys:
                    if geo_key in explicit_changes:
                        # Para consultas comparativas, filtros geográficos têm alta prioridade
                        merged_context[geo_key] = explicit_changes[geo_key]
                        context_metadata['merge_operations'].append(f"COMPARATIVE_GEO_PRIORITY: {geo_key}")
                
                # Marcar configurações especiais para SQL generation
                comparative_flags = [k for k in explicit_changes.keys() if k.startswith('_') and 'comparative' in k.lower()]
                for flag in comparative_flags:
                    merged_context[flag] = explicit_changes[flag]
                    context_metadata['merge_operations'].append(f"COMPARATIVE_FLAG: {flag}")
            
            # === VALIDAÇÃO DE CONSISTÊNCIA TEMPORAL ===
            # Verificar se datas fazem sentido logicamente
            if 'Data_>=' in merged_context and 'Data_<' in merged_context:
                try:
                    start_date = pd.to_datetime(merged_context['Data_>='])
                    end_date = pd.to_datetime(merged_context['Data_<'])
                    
                    if start_date >= end_date:
                        # Datas inconsistentes - manter apenas a mais recente (explícita)
                        if 'Data_>=' in explicit_changes or 'Data_<' in explicit_changes:
                            context_metadata['conflicts_resolved'].append("Corrigido conflito temporal - datas explícitas mantidas")
                        else:
                            # Remover datas problemáticas
                            merged_context.pop('Data_>=', None)
                            merged_context.pop('Data_<', None)
                            context_metadata['conflicts_resolved'].append("Removidas datas inconsistentes")
                            
                except Exception as e:
                    # Datas mal formadas - limpar
                    merged_context.pop('Data_>=', None)
                    merged_context.pop('Data_<', None)
                    context_metadata['conflicts_resolved'].append(f"Datas mal formadas removidas: {str(e)}")
            
            # === PROCESSAMENTO DO NOVO CONTEXTO (MÉDIA PRIORIDADE) ===
            for key, value in new_context.items():
                if key in explicit_changes:
                    # Já processado - pular
                    continue
                    
                if not value or str(value).strip() == "":
                    # Valor vazio - pular
                    continue
                    
                if key not in merged_context:
                    # Novo campo - adicionar
                    merged_context[key] = value
                    context_metadata['merge_operations'].append(f"ADDED: {key} = {value}")
                else:
                    # Campo existe - decidir se atualizar
                    old_value = merged_context[key]
                    
                    # === HEURÍSTICAS DE SUBSTITUIÇÃO ===
                    should_replace = False
                    
                    # Heurística 1: Valor novo é mais específico (mais longo)
                    if isinstance(value, str) and isinstance(old_value, str):
                        if len(value.strip()) > len(old_value.strip()):
                            should_replace = True
                            context_metadata['merge_operations'].append(f"REPLACED (mais específico): {key} {old_value} → {value}")
                    
                    # Heurística 2: Valor novo parece mais recente (contém ano maior)
                    import re
                    old_years = re.findall(r'\d{4}', str(old_value))
                    new_years = re.findall(r'\d{4}', str(value))
                    
                    if old_years and new_years:
                        max_old_year = max(int(y) for y in old_years)
                        max_new_year = max(int(y) for y in new_years)
                        
                        if max_new_year > max_old_year:
                            should_replace = True  
                            context_metadata['merge_operations'].append(f"REPLACED (mais recente): {key} {old_value} → {value}")
                    
                    # Heurística 3: Valor antigo é genérico demais
                    generic_values = ['DATE', 'N/A', 'undefined', 'null', '']
                    if str(old_value) in generic_values and str(value) not in generic_values:
                        should_replace = True
                        context_metadata['merge_operations'].append(f"REPLACED (específico vs genérico): {key} {old_value} → {value}")
                    
                    if should_replace:
                        merged_context[key] = value
            
            # === LIMPEZA E OTIMIZAÇÃO ===
            # Remover campos com valores inválidos
            invalid_keys = []
            for key, value in merged_context.items():
                if key.startswith('_'):
                    continue  # Preservar metadados
                    
                if not value or str(value).strip() in ['', 'N/A', 'null', 'undefined']:
                    invalid_keys.append(key)
            
            for key in invalid_keys:
                merged_context.pop(key, None)
                context_metadata['merge_operations'].append(f"CLEANED: removido {key} (valor inválido)")
            
            # === DECAY DE CONTEXTO ANTIGO ===
            # Implementar decay de contextos muito antigos (simulado por número de queries)
            if not hasattr(self, '_context_usage_count'):
                self._context_usage_count = {}
            
            for key in merged_context.keys():
                if not key.startswith('_'):
                    self._context_usage_count[key] = self._context_usage_count.get(key, 0) + 1
                    
                    # Se um contexto foi usado muitas vezes sem mudança, pode ser obsoleto
                    if self._context_usage_count[key] > 20:  # Threshold de decay
                        context_metadata['merge_operations'].append(f"WARNING: {key} usado {self._context_usage_count[key]} vezes (possível decay)")
            
            # === VALIDAÇÃO FINAL ===
            # Garantir que não há conflitos lógicos
            geo_fields = ['Municipio_Cliente', 'UF_Cliente']
            geo_values = {field: merged_context.get(field) for field in geo_fields if field in merged_context}
            
            if len(geo_values) > 1:
                # Verificar consistência geográfica básica
                municipio = geo_values.get('Municipio_Cliente', '').lower()
                uf = geo_values.get('UF_Cliente', '').lower()
                
                # Mapeamento básico município → UF (alguns casos conhecidos)
                municipio_uf_map = {
                    'joinville': 'sc', 'curitiba': 'pr', 'porto alegre': 'rs',
                    'sao paulo': 'sp', 'são paulo': 'sp', 'rio de janeiro': 'rj',
                    'florianopolis': 'sc', 'florianópolis': 'sc'
                }
                
                if municipio in municipio_uf_map:
                    expected_uf = municipio_uf_map[municipio]
                    if uf and uf != expected_uf:
                        # Conflito detectado - priorizar município (mais específico)
                        merged_context.pop('UF_Cliente', None)
                        context_metadata['conflicts_resolved'].append(f"Conflito geo: removido UF {uf} (inconsistente com município {municipio})")
            
            # === ATUALIZAÇÃO DO CONTEXTO PERSISTENTE ===
            # Atualizar contexto persistente para próximas queries
            self.persistent_context = merged_context.copy()
            
            # Adicionar metadados apenas em modo debug
            if hasattr(self, 'debug_info') and self.debug_info is not None:
                self.debug_info['context_merge_metadata'] = context_metadata
            
            return merged_context

        def run(self, query: str, debug_mode=False, **kwargs):
            # === FASE 1: DETECÇÃO DE CONSULTAS COMPARATIVAS ===
            comparative_info = self.detect_comparative_query(query)
            
            # === FASE 1.5: DETECÇÃO DE CONSULTAS TOP N ===
            top_n_info = self.detect_top_n_query(query)
            
            # === FASE 2: DETECÇÃO DE MUDANÇAS CONTEXTUAIS ===
            explicit_changes = self.detect_explicit_context_changes(query)
            
            # === FASE 3: EXPANSÃO DE FILTROS PARA COMPARAÇÃO ===
            if comparative_info['is_comparative']:
                expanded_filters = self.expand_filters_for_comparison(query, comparative_info, explicit_changes)
                # Usar filtros expandidos ao invés dos explícitos básicos
                explicit_changes = expanded_filters
            
            # === FASE 4: SISTEMA DE SUBSTITUIÇÃO AUTOMÁTICA ===
            should_substitute, substitution_summary = self.auto_substitute_parameters(query, explicit_changes)
            
            # === FASE 5: ANÁLISE DE RELEVÂNCIA DA CONSULTA ===
            unrelated_keywords = ['quantas', 'todos', 'mostre', 'liste', 'total', 'geral']
            is_unrelated_query = any(keyword in query.lower() for keyword in unrelated_keywords)
            
            # Para consultas comparativas, não limpar contexto mesmo se parecer não relacionada
            if is_unrelated_query and not explicit_changes and not should_substitute and not comparative_info['is_comparative']:
                self.persistent_context = {}
                self.clear_execution_state()
            
            # === FASE 6: APLICAÇÃO DO CONTEXTO COMPARATIVO ===
            # Para consultas comparativas, priorizar sistema de expansão
            if comparative_info['is_comparative']:
                context_to_apply = explicit_changes  # Já inclui filtros expandidos
            elif should_substitute:
                context_to_apply = explicit_changes
            else:
                context_to_apply = explicit_changes if explicit_changes else {}
            
            # SISTEMA COMPARATIVO: Injeção de contexto com suporte a comparações
            enhanced_query = self.inject_context_into_query(query, context_to_apply, comparative_info)
            
            # === FASE 7: INICIALIZAÇÃO DE DEBUG COMPARATIVO ===
            self.debug_info = {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "processed_query": "",
                "sql_queries": [],
                "query_contexts": [],
                "memory_context": "",
                "string_format_adjustments": [],
                "explicit_context_changes": explicit_changes,
                "persistent_context_before": self.persistent_context.copy(),
                "context_injected": len(self.persistent_context) > 0 or len(context_to_apply) > 0,
                
                # === INFORMAÇÕES DE SUBSTITUIÇÃO AUTOMÁTICA ===
                "auto_substitution_applied": should_substitute,
                "substitution_summary": substitution_summary,
                "context_applied": context_to_apply,
                "substitution_confidence": int(substitution_summary.split('(confiança: ')[1].split('%')[0]) if 'confiança:' in substitution_summary else 0,
                
                # === INFORMAÇÕES COMPARATIVAS ===
                "is_comparative_query": comparative_info['is_comparative'],
                "comparative_info": comparative_info,
                "filters_expanded": comparative_info['is_comparative'] and comparative_info.get('requires_expansion', False),
                "comparative_type": comparative_info.get('comparison_type'),
                "temporal_scope": comparative_info.get('temporal_scope'),
                "calculation_required": comparative_info.get('calculation_type') is not None,
            }

            # Normalizar a consulta do usuário (usar enhanced_query com contexto injetado)
            query_analysis = self.normalizer.normalize_query_terms(
                enhanced_query, self.alias_mapping
            )

            # Substituir aliases na query com contexto se necessário  
            processed_query = enhanced_query
            for alias, mapping_info in query_analysis["mapped_terms"].items():
                processed_query = processed_query.replace(
                    mapping_info["original_alias"], mapping_info["mapped_column"]
                )

            self.debug_info["processed_query"] = processed_query

            # Filtro inteligente de memória - só buscar se realmente relevante
            memory_keywords = ['top', 'percentual', 'anterior', 'primeiro', 'segundo', 'compare', 'relação', 'versus']
            should_use_memory = any(keyword in query.lower() for keyword in memory_keywords)
            
            relevant_memories = []
            if should_use_memory:
                try:
                    relevant_memories = self.memory.search_user_memories(
                        user_id=self.session_user_id, query=processed_query, limit=3  # Reduzido de 5 para 3
                    )
                    
                    # Filtrar memórias por relevância real - evitar contexto de Top5_total se não for sobre Top 5
                    if 'percentual' in query.lower() and 'segundo' in query.lower():
                        # Para perguntas sobre percentual do segundo estado, filtrar memórias específicas do Top 5
                        relevant_memories = [mem for mem in relevant_memories 
                                           if any(term in mem.memory.lower() for term in ['top 5', 'uf', 'estado', 'vendas'])]
                        relevant_memories = relevant_memories[:2]  # Máximo 2 memórias
                except:
                    relevant_memories = []

            # Adicionar contexto da memória de forma mais limpa
            if relevant_memories:
                # Criar contexto mais focado
                memory_summaries = []
                for mem in relevant_memories:
                    # Extrair apenas informações essenciais
                    if 'Top 5' in mem.memory and 'UF' in mem.memory:
                        memory_summaries.append("Contexto: Foi perguntado anteriormente sobre os Top 5 estados por vendas.")
                
                if memory_summaries:
                    memory_context = " ".join(memory_summaries[:1])  # Apenas 1 linha de contexto
                    processed_query = f"{memory_context}\n\nPergunta atual: {processed_query}"
                    self.debug_info["memory_context"] = memory_context

            # Executar a consulta processada - queries serão capturadas automaticamente pelo DebugDuckDbTools
            response = super().run(processed_query, **kwargs)
            
            # === FASE 5.5: PROCESSAMENTO DE VISUALIZAÇÃO ===
            # Processar dados de visualização baseado na detecção Top N
            visualization_data = self.process_and_visualize(query, response.content, top_n_info)
            
            # === FASE 6: PROCESSAMENTO CONTEXTUAL FINAL ===
            # Coletar todos os contextos das queries executadas
            all_query_contexts = self.debug_info.get("query_contexts", [])
            
            # Mesclar todos os contextos em um único contexto atual
            current_context = {}
            for ctx in all_query_contexts:
                if isinstance(ctx, dict):
                    current_context.update(ctx)
            
            # SISTEMA COMPARATIVO: Mesclar contextos com suporte comparativo
            final_context = self.merge_contexts(current_context, context_to_apply, comparative_info)
            
            # Atualizar debug info com informações contextuais finais
            self.debug_info["query_contexts"] = [final_context] if final_context else [{}]
            self.debug_info["final_merged_context"] = final_context
            self.debug_info["persistent_context_after"] = self.persistent_context.copy()
            
            # === ADICIONAR INFORMAÇÕES DE VISUALIZAÇÃO AO DEBUG INFO ===
            self.debug_info["top_n_info"] = top_n_info
            self.debug_info["visualization_data"] = visualization_data
            self.debug_info["should_visualize"] = visualization_data.get('type') == 'bar_chart' and visualization_data.get('has_data', False)
            
            # Garantir que sempre temos pelo menos um contexto
            if not self.debug_info.get("query_contexts"):
                self.debug_info["query_contexts"] = [{}]

            # Armazenar a interação na memória
            try:
                from agno.memory.v2.schema import UserMemory

                # Limpar caracteres problemáticos para encoding
                clean_query = query.encode("ascii", errors="ignore").decode("ascii")
                clean_response = response.content.encode(
                    "ascii", errors="ignore"
                ).decode("ascii")

                # Criar memória baseada na interação
                memory_content = f"Usuario: {clean_query}\nAssistente: {clean_response}"
                user_memory = UserMemory(
                    memory=memory_content, topics=["conversation", "interaction"]
                )

                self.memory.add_user_memory(
                    memory=user_memory, user_id=self.session_user_id
                )
            except Exception as e:
                # Se houver erro na memória, continuar sem falhar a resposta
                pass  # Silently fail to avoid encoding errors

            return response

    agent = PrincipalAgent(
        model=OpenAIChat(id=selected_model, reasoning_effort="low"),
        description="Você é um assistente especializado em análise de dados comerciais. Você tem acesso ao dataset DadosComercial_resumido_v02.parquet com normalização de texto aplicada e pode responder perguntas baseadas nesse conteúdo. Você também tem memória contextual para lembrar de conversas anteriores na mesma sessão.",
        tools=[
            ReasoningTools(add_instructions=True),
            CalculatorTools(
                add=True, subtract=True, multiply=True, divide=True, exponentiate=True
            ),
            PythonTools(run_code=True, pip_install=False),
            DuckDbTools(),
        ],
        knowledge=knowledge,
        enable_user_memories=True,
        instructions=f"""
# System Prompt - Target AI Agent Agno v2.0

## ⚡ OTIMIZAÇÃO DE PERFORMANCE
- EVITE re-execuções desnecessárias de cálculos já realizados
- PARE de imprimir o mesmo resultado múltiplas vezes 
- Execute cada cálculo UMA ÚNICA VEZ por pergunta
- Responda de forma DIRETA e CONCISA sem loops de raciocínio

## 🎯 IDENTIDADE E MISSÃO

Você é o **Agno**, um Analista Sênior de Business Intelligence especializado em transformar dados comerciais em insights estratégicos acionáveis. Sua missão é democratizar o acesso a análises complexas através de uma interface conversacional intuitiva, fornecendo respostas precisas, contextualizadas e de alto valor agregado.

### Competências Core:
- **Análise Estatística Avançada**: Domínio completo de métricas comerciais e financeiras
- **Storytelling com Dados**: Transformar números em narrativas compreensíveis
- **Consultoria Estratégica**: Identificar oportunidades e riscos nos dados
- **Comunicação Adaptativa**: Ajustar linguagem ao perfil do usuário

### Escopo de Atuação:
- ✅ **Foco principal**: Análises do dataset `DadosComercial_resumido_v02.parquet`
- ✅ **Temas relacionados**: Contexto de mercado, benchmarks, estratégias comerciais
- ⚠️ **Limitação**: Para temas completamente fora do escopo comercial, redirecione educadamente:
> "Essa questão está além da minha especialização em análise comercial. Posso ajudá-lo com insights sobre vendas, clientes, produtos e performance do seu negócio. Como posso apoiá-lo nessas áreas?"

---

## 🧠 FRAMEWORK DE PROCESSAMENTO (ReAct Enhanced)

### Fase 1: COMPREENSÃO
```python
# Processo interno - não visível ao usuário
1. Classificar tipo de consulta: [Exploratória | Específica | Comparativa | Temporal | Diagnóstica]
2. Identificar entidades: [Produtos | Clientes | Regiões | Períodos | Métricas]
3. Detectar nível técnico: [Executivo | Analista | Operacional]
4. Mapear dados necessários: [Colunas | Agregações | Filtros | Joins]
```

### Fase 2: PLANEJAMENTO
```python
# Estratégia de análise
1. Definir abordagem:
- Consulta simples → SQL direto
- Análise complexa → SQL + Python
- Insights profundos → Multi-step analysis
2. Priorizar insights por relevância
3. Planejar visualizações necessárias
```

### Fase 3: EXECUÇÃO
```python
# Ordem de uso das ferramentas
1. DuckDB → Extração e agregação de dados
2. Python/Calculator → Cálculos e transformações
3. Validação → Verificar coerência dos resultados
```

### Fase 4: SÍNTESE
```python
# Construção da resposta
1. Estruturar narrativa: Conclusão → Evidência → Contexto
2. Adicionar insights não solicitados mas relevantes
3. Sugerir próximos passos
```

---

## ⚙️ CONFIGURAÇÃO TÉCNICA

### 📊 Acesso aos Dados

```sql
-- Padrão obrigatório para todas as consultas
SELECT * FROM read_parquet('{data_path}')
WHERE condições
GROUP BY agrupamentos
ORDER BY ordenação
```

**Metadados do Dataset:**
- Arquivo: `{data_path}`
- Dimensões: `{len(df)}` registros × `{len(df.columns)}` colunas
- Colunas disponíveis: `{", ".join(df.columns.tolist())}`
- Colunas de texto normalizadas: `{", ".join(text_columns)}`

### 🔧 Ferramentas e Protocolos

#### DuckDB (SQL)
**Use para:**
- SELECT, WHERE, GROUP BY, ORDER BY
- Agregações: SUM, AVG, COUNT, MIN, MAX
- Window functions e CTEs
- **Nunca para:** Cálculos percentuais ou matemática complexa
- **Recurso inteligente:** O sistema automaticamente testa diferentes formatos de string (UPPERCASE, lowercase, Title Case) quando não encontra resultados

#### Python/Calculator
**Use para:**
- Cálculos percentuais e proporções
- Estatísticas avançadas
- Transformações complexas
- Validações matemáticas

#### Protocolo de Separação de Responsabilidades
```python
# CORRETO ✅
1. SQL: SELECT valor, quantidade FROM tabela
2. Python: percentual = (valor_a / valor_total) * 100

# INCORRETO ❌
1. SQL: SELECT (valor_a / valor_total) * 100 as percentual
```

### 🔍 Validação de Qualidade

**Checklist Obrigatório:**
- [ ] Valores dentro de ranges esperados
- [ ] Somas batem com totais
- [ ] Sem valores null inesperados
- [ ] Coerência temporal (datas válidas)
- [ ] Consistência de unidades (R$, unidades, %)

---

## 📝 ESTRUTURA DE RESPOSTA

### Template Master de Formatação

```markdown
## **[Título Contextualizado da Análise]** [Emoji Relevante]

[Parágrafo introdutório com resposta direta à pergunta - máximo 2 linhas]

### 📊 Dados e Evidências

| **Dimensão** | **Métrica 1** | **Métrica 2** |
|:---|---:|---:|
| Item A | R$ 100.000 | 1.500 un |
| Item B | R$ 85.000 | 1.200 un | 

### 💡 Principais Insights

**1. [Insight Mais Importante]**
- Explicação clara do achado
- Impacto nos negócios
- Recomendação específica

**2. [Segundo Insight]**
- Contextualização com mercado
- Comparação temporal se aplicável
- Ação sugerida

**3. [Oportunidade Identificada]**
- Potencial de crescimento
- Recursos necessários
- Timeline proposto

### 📈 Análise de Tendências
[Se aplicável, incluir análise temporal ou projeções]


### 🔍 Próximos Passos

Posso aprofundar esta análise em:
- **Detalhamento por [dimensão]**: Como cada [item] contribui?
- **Análise temporal**: Evolução mês a mês ou sazonalidade?
- **Benchmarking**: Como estamos versus o mercado?
- **Segmentação avançada**: Perfil detalhado de [categoria]?

*Qual aspecto você gostaria de explorar primeiro?*
```

### Adaptação por Tipo de Consulta

#### 🔹 Consulta Exploratória (ex: "fale sobre as vendas")
- Começar com visão macro (totais, médias)
- Top 5 em múltiplas dimensões
- Identificar padrões e anomalias
- Sugerir 3-4 análises específicas

#### 🔹 Consulta Específica (ex: "vendas de produto X em SP")
- Resposta direta e precisa
- Contextualização com totais
- Comparação com similares
- Evolução temporal se relevante

#### 🔹 Consulta Comparativa (ex: "compare Q1 vs Q2")
- Tabela comparativa clara
- Variações percentuais e absolutas
- Drivers de mudança
- Projeções baseadas em tendências

#### 🔹 Consulta Diagnóstica (ex: "por que vendas caíram?")
- Análise de causas raiz
- Decomposição por fatores
- Correlações identificadas
- Plano de ação corretivo

---

## 🎨 PRINCÍPIOS DE COMUNICAÇÃO

### Tom e Voz
- **Profissional mas acessível**: Evite jargões desnecessários
- **Confiante sem ser arrogante**: "Os dados indicam..." não "Obviamente..."
- **Proativo e consultivo**: Sempre adicione valor além do solicitado
- **Empático**: Reconheça desafios do negócio

### Formatação Visual
- ✅ **Use emojis estrategicamente**: Máximo 1 por seção
- ✅ **Destaque com negrito**: Apenas informações críticas
- ✅ **Tabelas para dados**: Sempre que > 3 itens
- ✅ **Tabelas Streamlit-compatíveis**: Use formatação markdown simples sem caracteres especiais desnecessários
- ✅ **Alinhamento de colunas**: Left (:---), Center (:---:), Right (---:)
- ❌ **Evite**: Excesso de itálico, CAPS LOCK, sublinhado, barras verticais extras em tabelas

### Formatação de Tabelas - REGRAS OBRIGATÓRIAS
```markdown
# FORMATO CORRETO para Streamlit:
| **Coluna 1** | **Coluna 2** | **Coluna 3** |
|:---|---:|:---:|
| Valor 1 | R$ 1.000 | 10% |
| Valor 2 | R$ 2.000 | 20% |

# NUNCA usar formatos que quebram o Streamlit:
- Colunas extras sem cabeçalho
- Caracteres especiais desnecessários
- Alinhamento inconsistente
```

### Storytelling com Dados
1. **Lead com a conclusão**: Responda primeiro, explique depois
2. **Contextualize sempre**: Compare com períodos, médias, benchmarks
3. **Seja específico**: "Aumento de 23%" não "aumento significativo"
4. **Actionable insights**: Toda análise deve sugerir uma ação

---

## 🚀 CASOS DE USO E EXEMPLOS

### Exemplo 1: Pergunta Vaga
**User**: "Como estão as vendas?"

**Agno**:
## **Panorama Geral de Vendas** 📊

As vendas totalizam R$ 809,6 milhões com forte concentração regional e oportunidades claras de expansão.

### 📊 Visão Executiva

| **Métrica** | **Valor** | **Destaque** |
|:---|---:|:---|
| **Faturamento Total** | R$ 809,6 mi | Acima da meta anual |
| **Volume Vendido** | 1,2 bi unidades | Recorde histórico |
| **Ticket Médio** | R$ 0,67 | Oportunidade de upsell |
| **Top Estado** | SC (13,5%) | R$ 109,4 milhões |
| **Top Produto** | [Categoria A] | 35% do faturamento |

### 💡 Principais Insights

**1. Concentração Regional Preocupante**
- Top 5 estados = 52% do faturamento
- 15 estados abaixo do potencial
- Recomendo expansão focada em SP e RJ

**2. Mix de Produtos Desbalanceado**
- 20% dos SKUs geram 80% da receita
- Produtos C e D com margem superior mas baixo volume
- Oportunidade: campanhas para produtos de alta margem

### 🔍 Gostaria de Explorar:

- **Evolução temporal**: Tendências mensais e sazonalidade?
- **Análise por cliente**: Quem são os top 20 compradores?
- **Performance regional**: Detalhar potencial por estado?
- **Mix de produtos**: Quais categorias crescem mais?

*Por onde começamos a análise detalhada?*

### Exemplo 2: Pergunta Específica com Cálculo
**User**: "Qual o percentual de crescimento de vendas mês a mês?"

**Agno**:
[Executa SQL para extrair vendas por mês]
[Usa Python para calcular percentuais]
[Formata resposta com tabela de evolução e gráfico de tendência]

---

## 🔄 APRENDIZADO CONTÍNUO

### Memória Contextual
- Mantenha histórico da conversa
- Referencie análises anteriores
- Evolua profundidade conforme interação
- Aprenda preferências do usuário

### Auto-Avaliação
Após cada resposta, valide mentalmente:
- [ ] Respondi completamente a pergunta?
- [ ] Adicionei insights não solicitados mas valiosos?
- [ ] A formatação está clara e profissional?
- [ ] Sugeri próximos passos relevantes?
- [ ] Usei as ferramentas corretamente?

---

## 🚨 TRATAMENTO DE EXCEÇÕES

### Dados Ausentes
```markdown
⚠️ **Nota sobre Dados**: 
Alguns registros apresentam valores ausentes em [campo]. 
A análise considera apenas os {{X}}% de dados completos, 
o que ainda representa uma amostra estatisticamente válida.
```

### Consultas Sem Resultados
```markdown
🔍 **Sem Resultados para os Critérios Especificados**

Não encontrei dados para [critério]. Isso pode indicar:
1. Produto/período ainda não cadastrado
2. Filtros muito restritivos

**Alternativas disponíveis:**
- [Sugestão similar 1]
- [Sugestão similar 2]

Gostaria de ajustar os parâmetros da busca?
```

### Erros Técnicos
```markdown
⚠️ **Ajuste Necessário**

Encontrei uma limitação técnica ao processar sua solicitação.
Estou reformulando a análise para contornar o problema.

[Tenta abordagem alternativa]
[Se persistir, explica limitação e sugere alternativa]
```

---

## 📚 REFERÊNCIA RÁPIDA

### Aliases de Colunas
```python
alias_mapping = {alias_mapping}
```

### Funções SQL Mais Usadas
```sql
-- Agregações com condicionais
SUM(CASE WHEN condição THEN valor ELSE 0 END)

-- Rankings
ROW_NUMBER() OVER (PARTITION BY grupo ORDER BY métrica DESC)

-- Períodos
DATE_TRUNC('month', data_coluna)

-- Filtros inteligentes
WHERE LOWER(coluna) LIKE '%termo%'
```

### Cálculos Python Padrão
```python
# Percentual
percentual = (parte / total) * 100

# Variação
variacao = ((valor_atual - valor_anterior) / valor_anterior) * 100

# Market Share
market_share = (vendas_empresa / vendas_mercado) * 100

# Taxa de Crescimento Composta
cagr = ((valor_final / valor_inicial) ** (1 / periodos)) - 1
```

---

## ✨ REGRA DE OURO

> **"Cada resposta deve deixar o usuário mais inteligente sobre seu negócio"**

Não apenas responda perguntas - eduque, inspire e capacite tomadas de decisão baseadas em dados. Seja o parceiro analítico que todo gestor gostaria de ter ao seu lado.
""",
        show_tool_calls=debug_mode,
        markdown=True,
    )

    # Fazer o DuckDB carregar automaticamente o arquivo parquet
    agent.run(
        f"CREATE TABLE dados_comerciais AS SELECT * FROM read_parquet('{data_path}');"
    )

    return agent, df


# Para compatibilidade com uso direto do arquivo
if __name__ == "__main__":
    agent, df = create_agent()

    # Teste simples para verificar funcionamento
    try:
        response = agent.run("Quantas linhas tem o dataset?")
        print("SUCCESS: Agent is working correctly!")
        print(f"Response type: {type(response)}")
    except Exception as e:
        print(f"ERROR: {e}")
