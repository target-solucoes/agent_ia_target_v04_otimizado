import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import warnings
import sys
import uuid

sys.path.append("src")
from duckdb_agent import create_agent
from text_normalizer import TextNormalizer

warnings.filterwarnings("ignore")

load_dotenv()

# Page configuration
st.set_page_config(page_title="Agente IA Target", page_icon="🤖", layout="wide")


def format_sql_query(query):
    """
    Formata uma query SQL para melhor legibilidade
    """
    if not query:
        return query

    # Remove caracteres de escape e limpa a string
    import re

    # Remove ANSI escape sequences
    query = re.sub(r"\x1b\[[0-9;]*m", "", query)

    # Remove caracteres de controle
    query = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", query)

    # Normaliza espaços em branco
    query = " ".join(query.split())

    # Formata as principais palavras-chave SQL
    keywords = [
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
        "LEFT JOIN",
        "RIGHT JOIN",
        "INNER JOIN",
        "GROUP BY",
        "ORDER BY",
        "HAVING",
        "UNION",
        "INSERT",
        "UPDATE",
        "DELETE",
        "AS",
    ]

    formatted_query = query
    for keyword in keywords:
        # Adiciona quebras de linha antes das principais palavras-chave
        if keyword in ["FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING"]:
            formatted_query = re.sub(
                f" {keyword} ", f"\n{keyword} ", formatted_query, flags=re.IGNORECASE
            )
        elif keyword == "SELECT":
            formatted_query = re.sub(
                f"^{keyword} ", f"{keyword}\n    ", formatted_query, flags=re.IGNORECASE
            )

    # Ajusta indentação
    lines = formatted_query.split("\n")
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line.upper().startswith(
            ("SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING")
        ):
            formatted_lines.append(line)
        else:
            formatted_lines.append("    " + line if line else line)

    return "\n".join(formatted_lines)


@st.cache_data
def load_parquet_data():
    """Carrega arquivo Parquet com tratamento robusto de codificação"""
    data_path = "data/raw/DadosComercial_limpo.parquet"

    # Method 1: Try direct pandas loading
    try:
        with st.spinner("🔄 Carregando dados..."):
            df = pd.read_parquet(data_path)

            # Process string columns for encoding issues
            string_cols = df.select_dtypes(include=["object"]).columns
            for col in string_cols:
                try:
                    # Convert to string and clean encoding
                    original_values = df[col].fillna("")
                    cleaned_values = []

                    for val in original_values:
                        if isinstance(val, bytes):
                            # Handle bytes
                            try:
                                cleaned_val = val.decode("utf-8", errors="replace")
                            except:
                                cleaned_val = str(val)
                        else:
                            # Handle strings with potential encoding issues
                            cleaned_val = (
                                str(val)
                                .encode("utf-8", errors="ignore")
                                .decode("utf-8")
                            )
                        cleaned_values.append(cleaned_val)

                    df[col] = cleaned_values

                except Exception as col_error:
                    # If column processing fails, keep original
                    st.warning(
                        f"⚠️ Mantendo coluna {col} original devido a: {col_error}"
                    )
                    continue

            return df, None

    except Exception as e:
        return None, f"Erro ao carregar dados: {str(e)}"


@st.cache_resource
def initialize_agent():
    """Inicializa o agente DuckDB configurado com memória temporária baseada em sessão"""
    try:
        # Gerar um ID único para a sessão do Streamlit se não existir
        if "session_user_id" not in st.session_state:
            st.session_state.session_user_id = str(uuid.uuid4())

        agent, df_agent = create_agent(session_user_id=st.session_state.session_user_id)
        return agent, df_agent, None
    except Exception as e:
        return None, None, str(e)


def main():
    # Enhanced CSS for professional styling
    st.markdown(
        """
    <style>
    .main > div {
        padding-top: 1rem;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #1a2332 0%, #2d3e50 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .app-title {
        color: white !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.5rem;
        font-weight: 300;
        margin: 0;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .app-subtitle {
        color: white !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 1rem;
        font-weight: 300;
        margin: 0.5rem 0 0 0;
        letter-spacing: 1px;
        opacity: 0.95;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .app-description {
        color: rgba(255,255,255,0.8);
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 0.9rem;
        font-weight: 300;
        margin: 1rem 0 0 0;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.5;
    }
    
    .feature-icons {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .feature-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        color: rgba(255,255,255,0.7);
        font-size: 0.8rem;
        font-weight: 300;
    }
    
    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        opacity: 0.8;
    }
    
    /* Chat Container Styling */
    .chat-main-container {
        display: flex;
        flex-direction: column;
        margin: 2rem 0;
    }
    
    .chat-messages-container {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 15px;
        border: 1px solid var(--secondary-background-color);
    }
    
    .chat-input-container {
        padding: 1.5rem 0;
        margin-top: 1rem;
        border-top: 1px solid var(--secondary-background-color);
    }
    
    /* Chat Message Styling - Dark mode friendly */
    .stChatMessage {
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid var(--secondary-background-color);
    }
    
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
        color: white !important;
        margin-left: 2rem;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        border-left: 4px solid #e74c3c;
        margin-right: 2rem;
    }
    
    /* Chat Input Styling - Dark mode friendly */
    .stChatInputContainer {
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent;
    }
    
    .stChatInput > div {
        border-radius: 25px !important;
        border: 2px solid #e74c3c !important;
    }
    
    .stChatInput input {
        border: none !important;
        font-size: 1rem !important;
        padding: 1rem 1.5rem !important;
    }
    
    /* Welcome message styling - Dark mode friendly */
    .welcome-message {
        text-align: center;
        padding: 3rem 2rem;
        font-style: italic;
        border-radius: 15px;
        margin: 2rem 0;
        border: 2px dashed var(--secondary-background-color);
    }
    
    .welcome-message h3 {
        color: #e74c3c;
        margin-bottom: 1rem;
    }
    
    
    /* Delete Chat Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #6c757d 0%, #5a6268 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.4rem 0.8rem;
        font-weight: 400;
        font-size: 0.85rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(108, 117, 125, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6268 0%, #495057 100%);
        transform: translateY(-1px);
        box-shadow: 0 3px 12px rgba(108, 117, 125, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Debug mode toggle styling */
    .stToggle > div {
        background-color: transparent !important;
    }
    
    .stToggle > div > div {
        background-color: #f0f0f0 !important;
        border-radius: 20px !important;
    }
    
    .stToggle > div > div[data-checked="true"] {
        background-color: #e74c3c !important;
    }
    
    /* Debug section styling */
    .debug-section {
        background-color: rgba(231, 76, 60, 0.05);
        border: 1px solid rgba(231, 76, 60, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .debug-title {
        color: #e74c3c;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .app-title {
            font-size: 2rem;
        }
        .feature-icons {
            gap: 1rem;
        }
        .header-container {
            padding: 1.5rem 1rem;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Import selected_model from duckdb_agent
    from duckdb_agent import selected_model

    # Enhanced Professional Header
    st.markdown(
        f"""
        <div class="header-container">
            <h1 class="app-title">🤖 AGENTE IA TARGET</h1>
            <p class="app-subtitle">INTELIGÊNCIA ARTIFICIAL PARA ANÁLISE DE DADOS</p>
            <p class="app-description">
                Converse naturalmente com seus dados comerciais. Faça perguntas em linguagem natural 
                e obtenha insights precisos através de análise inteligente.<br>
                <small style="opacity: 0.7;">Modelo: {selected_model}</small>
            </p>
            <div class="feature-icons">
                <div class="feature-item">
                    <div class="feature-icon">💬</div>
                    <span>Chat Natural</span>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">📊</div>
                    <span>Análise Rápida</span>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">🎯</div>
                    <span>Insights Precisos</span>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">🚀</div>
                    <span>Resultados Instantâneos</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load data and agent silently
    df, data_error = load_parquet_data()
    agent, df_agent, agent_error = initialize_agent()

    # Enhanced Chat interface
    if agent is not None and df is not None:
        # Center the chat interface
        chat_col1, chat_col2, chat_col3 = st.columns([1, 3, 1])

        with chat_col2:
            # Debug mode toggle
            debug_col1, debug_col2 = st.columns([3, 1])
            with debug_col2:
                debug_mode = st.toggle(
                    "Debug",
                    value=False,
                    help="Ativar modo debug para exibir queries SQL e raciocínio do agente",
                )

            # Store debug mode in session state
            st.session_state.debug_mode = debug_mode
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
                # Add welcome message as first assistant message
                welcome_msg = """👋 Olá! Sou o **Agente IA Target**, seu assistente para análise de dados comerciais.

Estou aqui para ajudá-lo a explorar e entender seus dados através de conversas naturais. Você pode me fazer perguntas como:
- "Quais são os produtos mais vendidos?"
- "Mostre o faturamento por região"
- "Analise as tendências de vendas"

Como posso ajudá-lo hoje?"""
                st.session_state.messages.append(
                    {"role": "assistant", "content": welcome_msg}
                )

            # Initialize session user ID for memory if not exists
            if "session_user_id" not in st.session_state:
                st.session_state.session_user_id = str(uuid.uuid4())

            # Delete Chat button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🗑️ Limpar", type="secondary"):
                    # Clear all session state related to chat
                    st.session_state.messages = []
                    if "session_user_id" in st.session_state:
                        del st.session_state.session_user_id
                    # Force app rerun to refresh everything
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # Display chat messages with improved styling
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # Process user input first
            if prompt := st.chat_input(
                "💬 Faça sua pergunta sobre os dados comerciais..."
            ):
                # Add user message to chat history and display
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Get agent response
                with st.spinner("🤔 Analisando..."):
                    try:
                        # Get debug mode from session state
                        debug_mode = st.session_state.get("debug_mode", False)

                        # Run agent with debug mode
                        response = agent.run(prompt, debug_mode=debug_mode)

                        # Prepare response content
                        response_content = response.content

                        # If debug mode is active, add debug information
                        if (
                            debug_mode
                            and hasattr(agent, "debug_info")
                            and agent.debug_info
                        ):
                            debug_content = "\n\n---\n\n## **INFORMAÇÕES DE DEBUG**\n\n"

                            # Original vs Processed Query
                            if agent.debug_info.get(
                                "processed_query"
                            ) != agent.debug_info.get("original_query"):
                                debug_content += f"**📝 Query Original:** `{agent.debug_info.get('original_query', 'N/A')}`\n\n"
                                debug_content += f"**🔄 Query Processada:** `{agent.debug_info.get('processed_query', 'N/A')}`\n\n"

                            # SQL Queries executed
                            if agent.debug_info.get("sql_queries"):
                                debug_content += "**💾 Queries SQL Executadas:**\n"
                                for i, query in enumerate(
                                    agent.debug_info["sql_queries"], 1
                                ):
                                    # Format SQL query for better readability
                                    formatted_query = format_sql_query(query)
                                    debug_content += f"```sql\n{formatted_query}\n```\n"

                            # Tool calls
                            if agent.debug_info.get("tool_calls"):
                                debug_content += "**🔧 Ferramentas Utilizadas:**\n"
                                for tool_call in agent.debug_info["tool_calls"]:
                                    debug_content += (
                                        f"- **{tool_call.get('tool', 'Unknown')}**\n"
                                    )
                                    debug_content += f"  - *Args:* `{tool_call.get('args', 'N/A')}`\n"
                                    if tool_call.get("result"):
                                        debug_content += f"  - *Resultado:* `{tool_call.get('result', 'N/A')}`\n"
                                    debug_content += "\n"

                            response_content += debug_content

                        st.session_state.messages.append(
                            {"role": "assistant", "content": response_content}
                        )
                    except Exception as e:
                        error_msg = f"❌ Erro ao processar: {str(e)}"
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )

                # Rerun to display new messages
                st.rerun()
    else:
        st.error("⚠️ Erro ao inicializar o sistema. Recarregue a página.")
        if data_error:
            st.error(f"Dados: {data_error}")
        if agent_error:
            st.error(f"Agente: {agent_error}")

    # --- Footer Target Data Experience ---
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    # Criação do footer com logotipo
    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

    with footer_col2:
        from PIL import Image
        import base64
        import io

        # Texto do footer
        # Fallback caso a imagem não seja encontrada
        st.markdown(
            """
            <div style="text-align: center; background: linear-gradient(135deg, #1a2332 0%, #2d3e50 100%); 
                        padding: 30px; border-radius: 15px; margin: 20px 0; display: flex; 
                        flex-direction: column; align-items: center; justify-content: center;">
                <div style="color: white; font-family: 'Arial', sans-serif; font-weight: 300; 
                           letter-spacing: 6px; margin: 0; font-size: 24px;">T A R G E T</div>
                <div style="color: #e74c3c; font-family: 'Arial', sans-serif; font-weight: 300; 
                          letter-spacing: 3px; margin: 8px 0 0 0; font-size: 12px;">D A T A &nbsp; E X P E R I E N C E</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
