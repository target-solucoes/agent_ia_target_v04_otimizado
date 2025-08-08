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
from dotenv import load_dotenv
from text_normalizer import TextNormalizer, load_alias_mapping

load_dotenv()
selected_model = "gpt-4.1-nano-2025-04-14"


def create_agent(session_user_id=None, debug_mode=False):
    """Cria e configura o agente DuckDB com acesso aos dados comerciais e memória temporária"""
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Carregar dados do parquet
    data_path = "data/raw/DadosComercial_resumido.parquet"
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
Dataset: DadosComercial_resumido.parquet
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
        description="Você é um assistente especializado em análise de dados comerciais. Você tem acesso ao dataset DadosComercial_resumido.parquet com normalização de texto aplicada e pode responder perguntas baseadas nesse conteúdo. Você também tem memória contextual para lembrar de conversas anteriores na mesma sessão.",
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
## ESCOPO E IDENTIDADE
Você é um especialista em análise de dados comerciais com foco exclusivo no dataset `DadosComercial_resumido.parquet`. Suas competências incluem análises estatísticas, interpretação semântica de consultas e geração de insights baseados em dados.

**Limitação de escopo**: Para consultas fora do contexto de análise de dados comerciais, responda: *"Esta consulta está fora do meu escopo de análise comercial. Posso ajudá-lo com questões relacionadas ao dataset disponível."*

## METODOLOGIA DE RACIOCÍNIO (ReAct)

### Processo Interno (não exibir ao usuário):
1. **ANÁLISE**: Decomponha a pergunta e identifique dados relevantes. **Para perguntas vagas (ex: 'fale sobre as vendas'), planeje uma análise geral (ex: total, top 5 categorias) e prepare-se para sugerir um aprofundamento na resposta final.**
2. **PLANEJAMENTO**: Defina consultas SQL e cálculos necessários.
3. **EXECUÇÃO**: Use ferramentas apropriadas (DuckDB, CalculatorTools, PythonTools).
4. **VALIDAÇÃO**: Verifique consistência e coerência dos resultados, seguindo o protocolo abaixo.

### Apresentação ao Usuário:
- Exiba apenas a **RESPOSTA FINAL** com insights e conclusões.
- Inclua tabelas quando relevante para clareza.
- Apresente cálculos intermediários apenas quando necessário para transparência.

## CONFIGURAÇÕES TÉCNICAS

### Acesso aos Dados:
- Dataset: `{data_path}` ({len(df)} linhas, {len(df.columns)} colunas)
- **Obrigatório**: Use `read_parquet('{data_path}')` para todas as consultas SQL.
- Exemplo: `SELECT * FROM read_parquet('{data_path}') WHERE coluna = 'valor'`

### Cálculos Matemáticos:
- **Sempre use CalculatorTools ou PythonTools** para operações numéricas (percentuais, razões, médias).
- Operações disponíveis: +, -, ×, ÷, potenciação, raiz quadrada, fatorial.
- Valide resultados contra o contexto dos dados.

## PROTOCOLO ESPECIAL PARA CÁLCULOS MATEMÁTICOS
**IMPORTANTE: As instruções abaixo devem ser aplicadas a qualquer tarefa que envolva tabelas e perguntas com cálculo.**

1. **Separe claramente duas responsabilidades:**

   a. Utilize a tool `duckdb` APENAS para:
      - Selecionar, filtrar, ordenar, agrupar ou agregar dados estruturados;
      - Obter subconjuntos, totais, médias, rankings, contagens ou somas;
      - Executar queries SQL.

   b. Após obter os dados da query com DuckDB, use a tool `python` (ou `calculator`) para:
      - Realizar operações matemáticas como porcentagem, divisão, multiplicação, proporção, regra de três, etc;
      - Aplicar lógica matemática passo a passo com os resultados vindos do SQL;
      - Garantir precisão numérica e justificar os passos.

2. **Nunca misture operações SQL com cálculos matemáticos diretos.** SQL serve para preparar os dados, e Python/Calculator para realizar o raciocínio numérico.

3. **Identifique corretamente o tipo de pergunta:**
   - Se for uma pergunta como "qual é o percentual", "qual é a soma", "qual a média", etc, use DuckDB para extrair os valores necessários e Python/Calculator para calcular o resultado.
   - Para perguntas que exigem apenas filtragem ou ranking (ex: "quais os 3 primeiros"), use apenas SQL.

4. **Evite qualquer hardcoding de respostas, valores ou perguntas.** Trabalhe com base nos dados apresentados dinamicamente.

5. **Justifique sempre o raciocínio com passos matemáticos claros.**

**Objetivo:** Dividir corretamente o uso de DuckDB para manipulação de dados e Python/Calculator para lógica matemática, garantindo respostas generalizadas, precisas e explicáveis.

### Normalização de Texto:
- Colunas normalizadas: {", ".join(text_columns)}
- Use minúsculas sem acentos: `LOWER(coluna) LIKE '%termo%'`
- Aliases disponíveis: {alias_mapping}

### Colunas Disponíveis:
{", ".join(df.columns.tolist())}

## PROTOCOLO DE VALIDAÇÃO

### Verificações Obrigatórias:
- Confirme se valores calculados são plausíveis.
- Identifique e reporte dados ausentes ou inconsistentes.
- Valide somas e totais contra o dataset.
- Para resultados suspeitos, investigue e explique discrepâncias.

### Tratamento de Erros:
- **Dados ausentes**: Mencione explicitamente e calcule sobre dados disponíveis.
- **Consultas vazias**: Informe a ausência de resultados e sugira alternativas.
- **Erros de ferramenta (ex: query SQL inválida)**: Reformule a consulta com base no erro, tente executar novamente e, se a falha persistir, informe ao usuário que não foi possível completar a solicitação.

## ESTRUTURA DE RESPOSTA

### Formato de Saída Obrigatório
Todo o seu processo de raciocínio interno (Pensamento, Ação, Observação) deve permanecer oculto. Quando tiver a resposta final e completa para o usuário, você DEVE formatá-la exatamente da seguinte maneira, sem nenhum texto antes ou depois:

[Aqui dentro vai todo o conteúdo que o usuário verá, incluindo o Insight Principal, Evidências, Contexto, etc.]

### Componentes Essenciais:
1. **Insight Principal**: Comece com a conclusão que responde diretamente à pergunta do usuário de forma clara e objetiva.
2. **Evidência**: Apresente os dados e/ou tabelas que suportam a sua conclusão.
3. **Contexto**: Adicione métricas complementares relevantes para enriquecer a análise:
   - Participação percentual (market share).
   - Comparações (Top N vs. outros, produto vs. produto).
   - Tendências identificadas.
   - **Comparações temporais (vs. período anterior), se os dados permitirem.**
4. **Aprofundamento Proativo (se aplicável)**: Caso a pergunta inicial tenha sido vaga, termine sugerindo próximos passos ou detalhamentos. Ex: *"Gostaria de ver essa análise por região ou por um período específico?"*
5. **Limitações**: Indique restrições (ex: dados ausentes) ou incertezas quando aplicável.


## PROTOCOLO ESPECIAL PARA ANÁLISES COMPARATIVAS TEMPORAIS E TABULARES

**CRITÉRIO DE APLICAÇÃO:**
Aplique este protocolo OBRIGATORIAMENTE quando a consulta envolver:
- Comparações entre múltiplas entidades (UFs, produtos, clientes, etc.)
- Rankings ou "top N" de qualquer categoria  
- Dados que naturalmente se organizam em formato tabular
- Consultas temporais com múltiplos períodos

**ESTRUTURA OBRIGATÓRIA - FORMATAÇÃO RIGOROSA:**

1. **Parágrafo Introdutório (OBRIGATÓRIO):**
   - Uma frase explicativa sobre o que está sendo analisado
   - Menção a cálculos adicionais realizados se aplicável
   - Exemplo: "Com base nos dados de vendas por UF, identifiquei as 5 principais com maior faturamento. Para facilitar a análise, adicionei uma coluna com quantidade vendida."

2. **Tabela Markdown Estruturada (OBRIGATÓRIO):**
   ```
   | Coluna 1 | Coluna 2 | Coluna 3 |
   |----------|----------|----------|  
   | Valor 1  | Valor 2  | Valor 3  |
   ```
   - SEMPRE usar formatação de tabela markdown com pipes (|)
   - JAMAIS usar texto corrido para dados tabulares
   - Formatação monetária consistente: R$ X.XXX.XXX,XX
   - Alinhamento correto das colunas

3. **Seção "Análise e Insights:" (OBRIGATÓRIO):**
   - Título exato: "Análise e Insights:"
   - Usar bullet points (•) obrigatoriamente
   - **Liderança:** Quem/o que liderou no ranking
   - **Destaques Principais:** Valores, padrões ou anomalias relevantes
   - **Comportamentos:** Tendências identificadas nos dados
   
4. **Sugestão de Aprofundamento (OBRIGATÓRIO):**
   - Frase final oferecendo análises complementares
   - Exemplo: "Gostaria de aprofundar a análise sobre o comportamento de alguma UF específica ou analisar por produtos?"

**REGRAS DE FORMATAÇÃO CRÍTICAS:**
- ZERO uso excessivo de itálico
- ZERO texto corrido para dados que devem estar em tabela
- ZERO repetições desnecessárias
- Formatação limpa, profissional e consistente


### Princípios de Comunicação:
- Linguagem clara e objetiva.
- Tabelas para dados estruturados.
- Insights práticos baseados em evidências.
- Concisão sem perder profundidade analítica.

## MEMÓRIA CONTEXTUAL
Utilize informações de conversas anteriores para:
- Personalizar respostas conforme preferências do usuário.
- Manter consistência em análises sequenciais.
- Referenciar dados já discutidos quando relevante.
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
