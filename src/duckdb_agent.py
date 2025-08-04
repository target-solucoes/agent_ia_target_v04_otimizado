from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.tools.duckdb import DuckDbTools
from agno.knowledge import AgentKnowledge
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.tools.calculator import CalculatorTools

import os
import pandas as pd
import tempfile
from dotenv import load_dotenv
from text_normalizer import TextNormalizer, load_alias_mapping

load_dotenv()
selected_model = "gpt-4.1-nano-2025-04-14"


def create_agent(session_user_id=None):
    """Cria e configura o agente DuckDB com acesso aos dados comerciais e memória temporária"""
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Carregar dados do parquet
    data_path = "data/raw/DadosComercial_limpo.parquet"
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
Dataset: DadosComercial_limpo.parquet
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
    memory = Memory(model=OpenAIChat(id=selected_model), db=memory_db)

    # Criar classe customizada de DuckDbTools para capturar queries
    class DebugDuckDbTools(DuckDbTools):
        def __init__(self, debug_info_ref=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.debug_info_ref = debug_info_ref

        def run_query(self, query: str) -> str:
            """Override do método run_query para capturar queries SQL executadas"""
            if self.debug_info_ref is not None and hasattr(
                self.debug_info_ref, "debug_info"
            ):
                if "sql_queries" not in self.debug_info_ref.debug_info:
                    self.debug_info_ref.debug_info["sql_queries"] = []

                # Limpar e formatar a query
                clean_query = query.strip()
                if (
                    clean_query
                    and clean_query not in self.debug_info_ref.debug_info["sql_queries"]
                ):
                    self.debug_info_ref.debug_info["sql_queries"].append(clean_query)

            # Executar a query original
            return super().run_query(query)

    # Criar classe customizada de agent que aplica normalização às consultas
    class NormalizedAgent(Agent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.normalizer = normalizer
            self.alias_mapping = alias_mapping
            self.df_normalized = df_normalized
            self.text_columns = text_columns
            self.memory = memory
            self.session_user_id = session_user_id or "default_user"
            self.debug_info = {}  # Para armazenar informações de debug

            # Substituir DuckDbTools por versão debug
            for i, tool in enumerate(self.tools):
                if isinstance(tool, DuckDbTools):
                    self.tools[i] = DebugDuckDbTools(debug_info_ref=self)

        def run(self, query: str, debug_mode=False, **kwargs):
            # Limpar debug info anterior
            self.debug_info = {
                "original_query": query,
                "processed_query": "",
                "sql_queries": [],
                "memory_context": "",
            }

            # Normalizar a consulta do usuário
            query_analysis = self.normalizer.normalize_query_terms(
                query, self.alias_mapping
            )

            # Substituir aliases na query original se necessário
            processed_query = query
            for alias, mapping_info in query_analysis["mapped_terms"].items():
                processed_query = processed_query.replace(
                    mapping_info["original_alias"], mapping_info["mapped_column"]
                )

            self.debug_info["processed_query"] = processed_query

            # Recuperar memórias relevantes antes de processar a query
            relevant_memories = self.memory.search_user_memories(
                user_id=self.session_user_id, query=processed_query, limit=5
            )

            # Adicionar contexto da memória se houver memórias relevantes
            if relevant_memories:
                memory_context = "\n".join(
                    [f"Lembrança: {mem.memory}" for mem in relevant_memories]
                )
                processed_query = f"Contexto da conversa anterior:\n{memory_context}\n\nPergunta atual: {processed_query}"
                self.debug_info["memory_context"] = memory_context

            # Executar a consulta processada - queries serão capturadas automaticamente pelo DebugDuckDbTools
            response = super().run(processed_query, **kwargs)

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

    agent = NormalizedAgent(
        model=OpenAIChat(id=selected_model),
        description="Você é um assistente especializado em análise de dados comerciais. Você tem acesso ao dataset DadosComercial_limpo.parquet com normalização de texto aplicada e pode responder perguntas baseadas nesse conteúdo. Você também tem memória contextual para lembrar de conversas anteriores na mesma sessão.",
        tools=[
            ReasoningTools(add_instructions=True),
            CalculatorTools(
                add=True, subtract=True, multiply=True, divide=True, exponentiate=True
            ),
            DuckDbTools(),
        ],
        knowledge=knowledge,
        enable_user_memories=True,
        instructions=f"""
ESCOPO DO PROJETO:
Você é um assistente especializado em análise de dados comerciais, focado exclusivamente no dataset DadosComercial_limpo.parquet. Seu escopo inclui:
- Análises estatísticas, estruturais e contextuais dos dados comerciais
- Interpretação semântica de consultas em linguagem natural
- Geração de insights baseados nos dados disponíveis
- Resposta a perguntas relacionadas ao conteúdo do dataset

IMPORTANTE: Caso uma pergunta fuja significativamente deste escopo, responda educadamente: "Esta pergunta está fora do meu escopo de análise de dados comerciais. Posso ajudá-lo com questões relacionadas ao dataset DadosComercial_limpo.parquet."

METODOLOGIA DE RACIOCÍNIO (ReAct - Reasoning and Acting):
Para cada consulta do usuário, siga esta estrutura de raciocínio:

1. **PENSAMENTO (Think)**: 
   - Analise a pergunta e identifique o que exatamente está sendo solicitado
   - Decomponha problemas complexos em subproblemas menores
   - Identifique quais dados e colunas são relevantes para a resposta
   - Considere o contexto das conversas anteriores (memória contextual)

2. **AÇÃO (Act)**:
   - Planeje a consulta SQL ou análise necessária
   - Execute as ferramentas apropriadas:
        - DuckDB para consultas SQL
        - CalculatorTools para cálculos matemáticos
        - Use CalculatorTools para cálculos matemáticos como porcentagens, proporções, razões, médias, etc.
   - Aplique normalização de texto quando necessário

3. **OBSERVAÇÃO (Observe)**:
   - Analise os resultados obtidos
   - Verifique se os dados respondem completamente à pergunta
   - Identifique padrões, tendências ou insights relevantes
   - Sempre valide se os valores utilizados nos cálculos fazem sentido à luz do dataset.
   - Por exemplo, se um valor de soma total parecer anormalmente alto ou baixo, retorne ao passo anterior para validar a origem.
   - Se possível, imprima os valores intermediários utilizados no cálculo para transparência.

4. **RESPOSTA (Respond)**:
   - Forneça uma resposta estruturada e fundamentada nos dados
   - Justifique conclusões com evidências dos resultados
   - Use tabelas quando apropriado para clareza
   - Indique limitações ou incertezas quando aplicável

IMPORTANTE: Nunca finalize uma resposta apenas com o "Pensamento". Você deve obrigatoriamente concluir com a etapa **RESPOSTA (Respond)**, fornecendo a resposta final formatada ao usuário.

INSTRUÇÕES TÉCNICAS:

ACESSO AOS DADOS:
- Você tem acesso a um dataset comercial com {len(df)} linhas e {len(df.columns)} colunas
- **Para realizar qualquer consulta SQL, você DEVE usar a função `read_parquet`**
- O caminho do arquivo é: '{data_path}'
- Para consultas SQL, use o DuckDB para carregar e analisar os dados do arquivo parquet
- Exemplo de query: SELECT * FROM read_parquet('{data_path}') LIMIT 10;

CÁLCULOS E ANÁLISES NUMÉRICAS:
- Para calcular percentuais, razões, proporções, taxas ou qualquer operação matemática, use a ferramenta CalculatorTools
- Exemplo: após obter o valor de vendas de PE e o total do top 5, calcule (valor_PE / total_top5) * 100 usando a função de divisão e multiplicação
- As operações disponíveis são: adição, subtração, multiplicação, divisão, potenciação, raiz quadrada e fatorial
- Evite fazer esses cálculos manualmente no corpo da resposta, sempre use CalculatorTools para garantir precisão

MEMÓRIA CONTEXTUAL:
- Você tem acesso a memórias de conversas anteriores na mesma sessão
- Use essas memórias para fornecer respostas mais contextualizadas
- Lembre-se de informações pessoais compartilhadas pelo usuário (nome, preferências, etc.)

NORMALIZAÇÃO DE TEXTO:
- O sistema aplica normalização automática de texto (minúsculas, remoção de acentos)
- Colunas de texto normalizadas: {", ".join(text_columns)}
- Use consultas em minúsculas e sem acentos para melhor compatibilidade
- Aliases disponíveis: {alias_mapping}

COLUNAS DISPONÍVEIS: {", ".join(df.columns.tolist())}

INSTRUÇÕES PARA BUSCA DE TEXTO:
- Para buscar por texto específico, use LOWER() e sem acentos
- Exemplo: WHERE LOWER(Municipio_Cliente) LIKE '%sao paulo%' (ao invés de 'São Paulo')
- Exemplo 2: WHERE LOWER(UF_CLIENTE) = 'sc' (ao invés de 'SC')
- Para campos categóricos, use valores normalizados em minúsculas

PRINCÍPIOS DE RESPOSTA:
- Sempre forneça respostas precisas baseadas nos dados disponíveis e no contexto da conversa
- Use tabelas para exibir dados quando apropriado
- Fundamente suas conclusões em evidências dos dados
- Sempre que possível, complemente a resposta com análises adicionais que ajudem na interpretação dos resultados.
- Calcule e apresente métricas complementares relevantes para o contexto da pergunta, como:
  - Participação percentual no total
  - Comparações entre categorias (ex: Top N vs. outros)
  - Conclusões ou insights derivados dos dados apresentados
  - Tendências ou padrões identificados
  - Análises de correlação ou causalidade
- Utilize linguagem clara e objetiva para destacar esses insights.
- Seja transparente sobre limitações e incertezas
- Mantenha foco estrito no escopo definido do projeto
""",
        show_tool_calls=True,
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
