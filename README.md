# 🤖 Target AI Agent Agno

Um chatbot inteligente para análise de dados comerciais utilizando a plataforma **Agno** com interface web moderna e memória contextual.

## 📋 Descrição

O Target AI Agent Agno é uma solução completa de inteligência artificial para análise de dados comerciais. O sistema combina as capacidades avançadas do framework Agno com DuckDB e OpenAI GPT-4, oferecendo uma interface conversacional intuitiva onde usuários podem fazer perguntas em linguagem natural sobre dados comerciais e obter insights precisos e fundamentados.

## ✨ Principais Características

- **Interface conversacional**: Chat interativo para análise de dados em linguagem natural
- **Memória contextual**: Lembra conversas anteriores dentro da mesma sessão
- **Normalização inteligente**: Tratamento automático de texto para consultas consistentes
- **Análise robusta**: Suporte a consultas SQL complexas via DuckDB
- **Interface profissional**: Design moderno e responsivo com Streamlit
- **Dados massivos**: Processa datasets com mais de 5 milhões de registros

## 🛠️ Tecnologias Utilizadas

### Backend & IA
- **[Agno](https://agno.ai/)**: Framework principal para criação do agente IA
- **OpenAI GPT-4.1**: Modelo de linguagem para processamento de consultas
- **DuckDB**: Engine SQL para análise rápida de dados
- **Pandas**: Manipulação e análise de dados
- **PyArrow/FastParquet**: Leitura eficiente de arquivos Parquet

### Interface & Visualização
- **Streamlit**: Framework para interface web interativa
- **CSS customizado**: Design profissional e responsivo

### Processamento de Dados
- **Text Normalizer**: Sistema personalizado de normalização de texto
- **SQLite**: Armazenamento de memória contextual
- **JSON**: Configuração de aliases e mapeamentos

### Desenvolvimento
- **Python 3.10+**: Linguagem principal
- **python-dotenv**: Gerenciamento de variáveis de ambiente
- **PyProject.toml**: Configuração de dependências

## 📊 Demonstração

O sistema processa um dataset comercial com **5.542.646 linhas** contendo informações sobre:
- Vendas por produto, cliente e região
- Dados temporais de emissão e entrega
- Métricas de faturamento e volume
- Segmentação de clientes e produtos

### Exemplos de Consultas

```
"Quais são os produtos mais vendidos por região?"
"Mostre o faturamento mensal dos últimos 12 meses"
"Analise o desempenho dos vendedores por segmento"
"Qual cliente teve maior volume de compras?"
```

## 🚀 Como Executar Localmente

### Pré-requisitos

- Python 3.10 ou superior
- [uv](https://docs.astral.sh/uv/) (recomendado) ou pip para gerenciamento de dependências
- Chave da API OpenAI
- 8GB+ de RAM recomendado para processamento dos dados

### Passo 1: Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd target_ai_agent_agno_layout_limpo_v01
```

### Passo 2: Instalar Dependências

#### Opção A: Usando uv (Recomendado)

```bash
# Instalar uv se ainda não tiver
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar o projeto e dependências
uv pip install -e .

# Ou criar ambiente virtual e instalar
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
uv pip install -e .
```

#### Opção B: Usando pip tradicional

```bash
# Instalar o projeto e dependências
pip install -e .

# Ou instalar dependências individuais
pip install streamlit pandas agno duckdb openai python-dotenv pyarrow fastparquet chardet sqlalchemy pyyaml numpy
```

### Passo 3: Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
OPENAI_API_KEY=sua_chave_openai_aqui
AGENT_TEMPERATURE=0.0
AGENT_MAX_TOKENS=1000
```

### Passo 4: Preparar os Dados

Certifique-se de que o arquivo de dados está no local correto:
```
data/raw/DadosComercial_limpo.parquet
```

### Passo 5: Executar Testes (Opcional)

```bash
# Teste de validação do setup
python src/validate_setup.py

# Teste de integração completa
python test_integration.py

# Teste simples do agente
python test_simple.py
```

### Passo 6: Iniciar a Aplicação

```bash
streamlit run app.py
```

A aplicação estará disponível em: **http://localhost:8501**

## 📁 Estrutura do Projeto

```
target_ai_agent_agno_layout_limpo_v01/
├── 📱 app.py                          # Interface Streamlit principal
├── 📂 src/                            # Código fonte principal
│   ├── 🤖 duckdb_agent.py             # Agente principal com DuckDB
│   ├── 📝 text_normalizer.py          # Sistema de normalização de texto
│   ├── 🔍 debug_agent.py              # Utilitários de debug
│   └── ✅ validate_setup.py           # Validação do ambiente
├── 📂 data/                           # Dados e configurações
│   ├── 📂 raw/
│   │   └── 📊 DadosComercial_limpo.parquet  # Dataset principal
│   ├── 📂 mappings/
│   │   └── 🗂️ alias.json             # Mapeamento de aliases
│   └── 📂 test_questions/
│       └── ❓ qa_test_data.json       # Perguntas de teste
├── 📂 sessions/                       # Sessões de chat salvas
├── 🧪 test_*.py                       # Scripts de teste
├── ⚙️ pyproject.toml                  # Configuração do projeto
└── 📚 README.md                       # Esta documentação
```

## 🔧 Funcionalidades Detalhadas

### Agente Inteligente
- **Processamento de linguagem natural** para consultas comerciais
- **Análise contextual** com memória de conversas anteriores
- **Geração automática de consultas SQL** baseada nas perguntas
- **Interpretação semântica** de termos de negócio

### Normalização de Texto
- **Remoção automática de acentos** para consultas consistentes
- **Padronização de capitalização** e espaçamento
- **Sistema de aliases** para termos comerciais comuns
- **Mapeamento inteligente** de consultas do usuário

### Interface Avançada
- **Chat responsivo** com histórico completo
- **Feedback visual** durante processamento
- **Design profissional** com gradientes e animações
- **Suporte a temas** claro e escuro

### Performance
- **Cache inteligente** de dados e agente
- **Processamento em memória** com DuckDB
- **Consultas SQL otimizadas** para datasets grandes
- **Carregamento assíncrono** de componentes

## 🔍 Exemplos de Uso Avançado

### Análises Financeiras
```
"Mostre a evolução do faturamento trimestral"
"Identifique os clientes com maior ticket médio"
"Analise a sazonalidade das vendas por produto"
```

### Análises Geográficas
```
"Compare o desempenho de vendas por estado"
"Quais municípios têm maior potencial de crescimento?"
"Mostre a distribuição geográfica dos clientes"
```

### Análises de Produto
```
"Quais produtos têm melhor margem de contribuição?"
"Identifique produtos com queda nas vendas"
"Analise o ciclo de vida dos produtos por família"
```

## 🐛 Solução de Problemas

### Erro de Autenticação OpenAI
```bash
# Verificar se a chave está configurada
echo $OPENAI_API_KEY

# Reconfigurar se necessário
export OPENAI_API_KEY="sua-chave-aqui"
```

### Erro de Memória
```bash
# Para datasets muito grandes, ajustar configurações
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
```

### Problemas de Encoding
O sistema possui tratamento automático de encoding, mas se necessário:
```python
# Forçar encoding UTF-8
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
```

### Porta em Uso
```bash
# Usar porta alternativa
streamlit run app.py --server.port 8502
```

## 📈 Métricas e Performance

- **Dataset**: 5.542.646 linhas × 17 colunas
- **Tempo de carregamento**: ~3-5 segundos
- **Tempo de resposta**: ~2-10 segundos por consulta
- **Memória utilizada**: ~2-4GB para dataset completo
- **Suporte simultâneo**: Múltiplas sessões independentes


## 📄 Licença

Este projeto é proprietário da **Target Data Experience**.

## 🎯 Target Data Experience

Desenvolvido com ❤️ pela equipe **Target Data Experience** - Transformando dados em insights estratégicos através de inteligência artificial avançada.

---

*Para suporte técnico ou dúvidas sobre implementação, consulte a documentação interna ou entre em contato com a equipe de desenvolvimento.*