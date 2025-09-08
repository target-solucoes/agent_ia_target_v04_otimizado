import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
import warnings
import sys
import uuid
import plotly.express as px
import plotly.graph_objects as go
import time

sys.path.append("src")
from chatbot_agents import create_agent
from text_normalizer import TextNormalizer

warnings.filterwarnings("ignore")

load_dotenv()

# Page configuration
st.set_page_config(page_title="Agente IA Target v0.4 (Otimizado)", page_icon="🤖", layout="wide")


def filter_user_friendly_context(context_dict):
    """
    Filtra contexto para mostrar apenas variáveis relevantes para o usuário,
    removendo variáveis técnicas internas.
    """
    if not context_dict:
        return {}
    
    # Variáveis técnicas que devem ser ocultadas (prefixos e nomes específicos)
    technical_prefixes = [
        '_temporal_', '_comparative_', '_requires_', '_preserve_', '_enable_', 
        '_disable_', '_auto_', '_override_', '_expand_', '_allow_'
    ]
    
    technical_keywords = [
        'merge_timestamp', 'merge_operations', 'conflicts_resolved', 'context_age',
        'calculation_required', 'comparison_type', 'temporal_metadata', 
        'growth_type', 'variation_type', 'evolution_granularity'
    ]
    
    # Variáveis que devem sempre ser mostradas (lista de permissão)
    user_relevant_fields = [
        'Municipio_Cliente', 'UF_Cliente', 'Des_Linha_Produto', 'Data_>=', 'Data_<', 
        'Data', 'Cliente', 'Produto', 'Regiao', 'Vendedor', 'sem_filtros'
    ]
    
    filtered_context = {}
    
    for key, value in context_dict.items():
        # Sempre incluir campos relevantes para o usuário
        if key in user_relevant_fields:
            filtered_context[key] = value
            continue
        
        # Verificar se é uma variável técnica
        is_technical = False
        
        # Verificar prefixos técnicos
        if any(key.startswith(prefix) for prefix in technical_prefixes):
            is_technical = True
        
        # Verificar palavras-chave técnicas
        if any(keyword in key.lower() for keyword in technical_keywords):
            is_technical = True
        
        # Verificar se é metadado interno (começando com underscore)
        if key.startswith('_') and key not in user_relevant_fields:
            is_technical = True
        
        # Se não é técnica, incluir no contexto filtrado
        if not is_technical:
            filtered_context[key] = value
    
    return filtered_context


def format_context_for_display(context_dict):
    """
    Formata contexto de forma amigável para exibição no sidebar
    """
    if not context_dict or context_dict.get('sem_filtros') == 'consulta_geral':
        return "🔍 **Consulta Geral**\n\n*Nenhum filtro ativo*"
    
    display_parts = ["✅ **Filtros Ativos**", ""]
    
    # Categorizar filtros
    location_filters = []
    temporal_filters = []
    product_filters = []
    other_filters = []
    
    for key, value in context_dict.items():
        if key in ['Municipio_Cliente', 'UF_Cliente', 'cidade', 'estado', 'municipio', 'uf']:
            location_filters.append((key, value))
        elif key in ['Data_>=', 'Data_<', 'Data', 'periodo', 'mes', 'ano']:
            temporal_filters.append((key, value))
        elif key in ['Des_Linha_Produto', 'Produto', 'produto', 'linha']:
            product_filters.append((key, value))
        else:
            other_filters.append((key, value))
    
    # Formatação por categoria com melhor visual
    if location_filters:
        display_parts.append("📍 **Localização**")
        for key, value in location_filters:
            if key in ['Municipio_Cliente', 'cidade', 'municipio']:
                display_parts.append(f"🏙️ Cidade: **{value}**")
            elif key in ['UF_Cliente', 'estado', 'uf']:
                display_parts.append(f"🗺️ Estado: **{value}**")
            else:
                display_parts.append(f"📍 {key}: **{value}**")
        display_parts.append("")
    
    if temporal_filters:
        display_parts.append("📅 **Período**")
        
        # Detectar se é um range de datas
        start_date = None
        end_date = None
        
        for key, value in temporal_filters:
            if 'Data_>=' in key or key == 'inicio':
                start_date = value
            elif 'Data_<' in key or key == 'fim':
                end_date = value
        
        if start_date and end_date:
            display_parts.append(f"⏰ **Período**: {start_date} até {end_date}")
        else:
            # Exibir filtros temporais individuais
            for key, value in temporal_filters:
                if key == 'Data':
                    display_parts.append(f"📆 Data: **{value}**")
                elif 'mes' in key.lower():
                    display_parts.append(f"📅 Mês: **{value}**")
                elif 'ano' in key.lower():
                    display_parts.append(f"🗓️ Ano: **{value}**")
                else:
                    display_name = key.replace("Data_", "").replace(">=", "A partir de").replace("<", "Antes de")
                    display_parts.append(f"📅 {display_name}: **{value}**")
        display_parts.append("")
    
    if product_filters:
        display_parts.append("🛍️ **Produto**")
        for key, value in product_filters:
            if key in ['Des_Linha_Produto', 'linha']:
                display_parts.append(f"📦 Linha: **{value}**")
            elif key in ['Produto', 'produto']:
                display_parts.append(f"🏷️ Produto: **{value}**")
            else:
                display_parts.append(f"🛍️ {key}: **{value}**")
        display_parts.append("")
    
    if other_filters:
        display_parts.append("📊 **Outros Filtros**")
        for key, value in other_filters:
            # Nomes mais amigáveis para filtros
            display_name = key.replace("_", " ").replace("-", " ").title()
            display_parts.append(f"⚙️ {display_name}: **{value}**")
        display_parts.append("")
    
    # Adicionar rodapé informativo se houver filtros
    if any([location_filters, temporal_filters, product_filters, other_filters]):
        display_parts.extend(["---", "💡 *Filtros aplicados à consulta atual*"])
    
    return "\n".join(display_parts).strip()


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


def format_compact_number(value):
    """
    Formata números grandes em notação compacta (1M, 2.5M, etc.)
    """
    try:
        if value >= 1_000_000_000:
            return f"{value/1_000_000_000:.1f}B"
        elif value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value/1_000:.1f}K"
        else:
            return f"{value:.0f}"
    except:
        return str(value)

def render_plotly_visualization(visualization_data):
    """
    Renderiza gráfico Plotly baseado nos dados de visualização do agente.
    Retorna True se renderizou um gráfico, False se renderizou uma tabela.
    """
    if not visualization_data:
        return False
    
    # Se não é para visualizar como gráfico, não fazer nada
    if visualization_data.get('type') != 'bar_chart' or not visualization_data.get('has_data', False):
        return False
    
    try:
        # Obter dados do DataFrame
        df = visualization_data.get('data')
        config = visualization_data.get('config', {})
        
        if df is None or df.empty:
            return False
        
        # Preparar rótulos compactos para as barras
        df_with_labels = df.copy()
        df_with_labels['value_label'] = df_with_labels['value'].apply(format_compact_number)
        
        # Criar gráfico de barras horizontais
        fig = px.bar(
            df_with_labels,
            x='value',
            y='label',
            orientation='h',
            title=config.get('title', 'Top Resultados'),
            labels={
                'value': 'Valor',
                'label': 'Item'
            },
            text='value_label'  # Usar rótulos compactos
        )
        
        # Configurações de layout para melhor aparência
        fig.update_layout(
            height=max(400, len(df) * 45),  # Altura ligeiramente aumentada para acomodar rótulos
            margin=dict(l=20, r=120, t=50, b=20),  # Margem direita aumentada para rótulos
            xaxis_title="",
            yaxis_title="",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        # Personalizar barras com cores azuis harmoniosas
        fig.update_traces(
            marker_color='#3498db',  # Azul agradável e harmonioso
            marker_line_color='#2980b9',  # Borda azul mais escura
            marker_line_width=1.5,
            opacity=0.85,
            textposition='outside',  # Posição dos rótulos fora das barras
            textfont=dict(size=11, color='#2c3e50', family='Arial')  # Estilo do texto dos rótulos
        )
        
        # Configurações do eixo Y para melhor legibilidade
        fig.update_yaxes(
            categoryorder='total ascending',  # Ordenar por valor
            tickfont=dict(size=12, family='Arial')
        )
        
        # Configurações do eixo X com formatação inteligente e estilo aprimorado
        value_format = config.get('value_format', 'number')
        if value_format == 'currency':
            fig.update_xaxes(
                tickformat=',.0f',  # Formato monetário com separadores de milhares
                tickprefix='R$ ',
                tickfont=dict(size=10, family='Arial'),
                gridcolor='rgba(52, 152, 219, 0.2)',  # Grid sutil em azul
                gridwidth=1
            )
        else:
            fig.update_xaxes(
                tickformat=',.0f',  # Formato numérico com separadores de milhares
                tickfont=dict(size=10, family='Arial'),
                gridcolor='rgba(52, 152, 219, 0.2)',  # Grid sutil em azul
                gridwidth=1
            )
        
        # Ajustar título do gráfico com melhor estilo
        fig.update_layout(
            title_font=dict(size=16, family='Arial', color='#2c3e50'),
            title_x=0.5  # Centralizar título
        )
        
        # Renderizar o gráfico no Streamlit
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        
        return True
        
    except Exception as e:
        # Em caso de erro, não fazer nada e deixar o conteúdo textual aparecer
        st.error(f"Erro ao renderizar gráfico: {str(e)}")
        return False


@st.cache_data
def load_parquet_data():
    """Carrega arquivo Parquet com tratamento robusto de codificação"""
    data_path = "data/raw/DadosComercial_resumido_v02.parquet"

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

    # Import selected_model from chatbot_agents
    from chatbot_agents import selected_model

    # Enhanced Professional Header
    st.markdown(
        f"""
        <div class="header-container">
            <h1 class="app-title">🤖 AGENTE IA TARGET v0.4 (Otimizado)</h1>
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

    # Enhanced Chat interface with Sidebar Context
    if agent is not None and df is not None:
        # Sidebar para contexto
        with st.sidebar:
            st.markdown("## 📊 Contexto da Consulta")
            st.markdown("---")
            
            # Exibir contexto atual ou estado inicial
            if hasattr(st.session_state, 'current_context') and st.session_state.current_context:
                st.markdown(st.session_state.current_context)
            else:
                st.markdown("🔍 **Aguardando consulta...**\n\n*O contexto dos filtros aparecerá aqui*")
        
        # Main content area - usar mais espaço já que sidebar está sendo usado
        chat_col1, chat_col2, chat_col3 = st.columns([0.5, 4, 0.5])

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
                    # NOVA FUNCIONALIDADE: Limpar contexto persistente do agente
                    if agent is not None and hasattr(agent, 'persistent_context'):
                        agent.persistent_context = {}
                    # Force app rerun to refresh everything
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # Display chat messages with improved styling
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        # Verificar se é uma mensagem do assistant com visualização
                        if (message["role"] == "assistant" and 
                            "visualization_data" in message and 
                            message["visualization_data"]):
                            
                            # Tentar renderizar gráfico Plotly primeiro
                            chart_rendered = render_plotly_visualization(message["visualization_data"])
                            
                            # Se renderizou gráfico, não exibir conteúdo textual (ou exibir apenas insights)
                            if chart_rendered:
                                # Extrair apenas insights do conteúdo (tudo que não seja dados tabulares)
                                content_lines = message["content"].split('\n')
                                insights_lines = []
                                response_time_line = ""
                                skip_data_section = False
                                
                                for line in content_lines:
                                    line_lower = line.lower().strip()
                                    
                                    # Capturar linha do tempo de resposta
                                    if "tempo de resposta:" in line_lower:
                                        response_time_line = line
                                        continue
                                    
                                    # Identificar seções de dados para pular
                                    if any(marker in line_lower for marker in ['```', 'tabela:', '|', '1.', '2.', '3.', 'ranking', 'top ']):
                                        if any(marker in line_lower for marker in ['insight', 'observa', 'conclus', 'destaq']):
                                            skip_data_section = False
                                        else:
                                            skip_data_section = True
                                            continue
                                    
                                    # Incluir apenas insights/análises
                                    if not skip_data_section and line.strip():
                                        if any(word in line_lower for word in ['insight', 'anális', 'observ', 'destaq', 'conclus', 'importante']):
                                            insights_lines.append(line)
                                
                                # Exibir apenas insights se houver
                                if insights_lines:
                                    st.markdown('\n'.join(insights_lines))
                                
                                # Sempre exibir o tempo de resposta se existir
                                if response_time_line:
                                    # Garantir formatação correta
                                    if "⏱️" not in response_time_line:
                                        response_time_line = response_time_line.replace("*Tempo de resposta:", "⏱️ *Tempo de resposta:")
                                    st.markdown(f"\n{response_time_line}")
                            else:
                                # Se não conseguiu renderizar gráfico, exibir conteúdo textual normal
                                st.markdown(message["content"])
                        else:
                            # Mensagem normal (usuário ou assistant sem visualização)
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
                        # Capture start time
                        start_time = time.time()
                        
                        # Get debug mode from session state
                        debug_mode = st.session_state.get("debug_mode", False)

                        # Run agent with debug mode
                        response = agent.run(prompt, debug_mode=debug_mode)
                        
                        # Calculate response time
                        response_time = time.time() - start_time

                        # Prepare response content
                        response_content = response.content

                        # ALWAYS display query context above response
                        final_context = {}
                        normalizations_note = ""
                        
                        # Extract context from debug info if available
                        if hasattr(agent, "debug_info") and agent.debug_info:
                            # Get query contexts
                            all_contexts = agent.debug_info.get("query_contexts", [])
                            if all_contexts:
                                # Merge all contexts
                                for ctx in all_contexts:
                                    if isinstance(ctx, dict):
                                        final_context.update(ctx)
                          
                       
                        # Update sidebar context instead of showing in chat
                        if final_context:
                            # Filter and format context for user-friendly display
                            filtered_context = filter_user_friendly_context(final_context)
                            formatted_context = format_context_for_display(filtered_context)
                            st.session_state.current_context = formatted_context
                        else:
                            st.session_state.current_context = "Nenhum filtro específico aplicado"

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

                        # Add response time to content
                        response_content += f"\n\n---\n\n⏱️ *Tempo de resposta: {response_time:.2f}s*"

                        # Preparar dados de visualização para a mensagem
                        message_data = {
                            "role": "assistant", 
                            "content": response_content
                        }
                        
                        # Adicionar dados de visualização se disponíveis
                        if hasattr(agent, "debug_info") and agent.debug_info:
                            visualization_data = agent.debug_info.get("visualization_data")
                            if visualization_data and visualization_data.get('type') == 'bar_chart' and visualization_data.get('has_data'):
                                message_data["visualization_data"] = visualization_data
                        
                        st.session_state.messages.append(message_data)
                    except Exception as e:
                        # Calculate response time even for errors
                        response_time = time.time() - start_time
                        error_msg = f"❌ Erro ao processar: {str(e)}\n\n---\n\n⏱️ *Tempo de resposta: {response_time:.2f}s*"
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
