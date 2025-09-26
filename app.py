# -*- coding: utf-8 -*-
"""
================================================================================
AGENTE GENÉRICO PARA ANÁLISE DE DADOS CSV COM GEMINI, LANGCHAIN E STREAMLIT
================================================================================

Objetivo:
Este script implementa um agente de IA genérico para Análise Exploratória de
Dados (EDA). Ele permite que um usuário carregue um arquivo CSV, faça perguntas
em linguagem natural e receba insights, tabelas e gráficos gerados pelo
modelo Google Gemini através do framework LangChain.

Autor: Rogerio
Baseado na solicitação de: [analise_de_dados/desafio_extra]

Funcionalidades:
- Interface web interativa com Streamlit.
- Carregamento de qualquer arquivo CSV.
- Conexão segura com a API do Google Gemini.
- Agente LangChain que executa código Python (Pandas, Matplotlib) para análise.
- Memória de conversação para perguntas contextuais.
- Geração e exibição automática de gráficos.
- Prompt de sistema detalhado para guiar o comportamento do agente.
"""

# Passo 1: Importação das Bibliotecas Essenciais
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# ==============================================================================
# CONSTANTES E CONFIGURAÇÕES GLOBAIS
# ==============================================================================

# Definição do prompt do sistema (persona do agente) como uma constante.
# Isso melhora a legibilidade e facilita a manutenção do prompt.
AGENT_PERSONA = """
Você é um analista de dados e especialista em Python, Pandas, Matplotlib e Seaborn. Sua tarefa é analisar os dados carregados e apresentar análises detalhadas.
- Responda a perguntas sobre o DataFrame (dados) `df` carregado.
- Você tem acesso a uma ferramenta para executar código Python internamente.
- Para responder, DEVE sempre gerar e executar um código Python válido utilizando a ferramenta `python_repl_ast`.
- O DataFrame está disponível na variável `df`.
- IMPORTANTE para gráficos: Se a pergunta solicitar um gráfico, GERE O CÓDIGO necessário para criar o gráfico usando Matplotlib ou Seaborn. NÃO utilize `st.pyplot()` no código gerado. Utilize apenas o código padrão de plotagem, pois a interface cuidará da exibição.
- Baseie-se exclusivamente nos dados carregados do CSV.
- Para dados numéricos decimais, retorne apenas duas casas decimais.
- Não invente informações: respalde-se sempre nos dados disponíveis.
- Seja conciso e direto ao ponto.
- Sempre que possível, forneça respostas em formato de tabela para melhor visualização.
- Utilize gráficos para ilustrar tendências, distribuições e correlações nos dados.
- Os gráficos recomendados incluem histogramas, gráficos de barras, de linhas, de dispersão, mapas de calor ou outros adequados à análise.
- Sempre que possível, utilize gráficos para complementar suas respostas.
- Explique suas conclusões de forma clara e objetiva.
- Caso não saiba a informação solicitada, responda: "Não sei informar o que você pediu. Estou pronto para sua próxima pergunta ou instrução."
- Após gerar um gráfico, forneça também uma breve explicação textual sobre o que o gráfico representa.
- Não mostre o código Python gerado, a menos que seja explicitamente solicitado. Apresente apenas o resultado (texto, tabelas ou gráficos).
- Seja um analista de dados crítico e detalhista.
- Responda sempre em português, não traga respostas em inglês.
"""

# O armazenamento do histórico de chat é um dicionário global simples.
# Isso funciona bem para o modelo de execução do Streamlit, onde cada sessão de usuário
# tem seu próprio processo, mas não persistirá se o servidor for reiniciado.
store = {}

# ==============================================================================
# FUNÇÕES AUXILIARES E DE LÓGICA
# ==============================================================================

@st.cache_resource
def get_llm(google_api_key):
    """
    Inicializa e armazena em cache o modelo de linguagem (LLM) do Google Gemini.
    O cache (@st.cache_resource) evita recarregar o modelo e reestabelecer
    conexões a cada interação do usuário, economizando tempo e recursos.

    Args:
        google_api_key (str): A chave de API para autenticação no Google Gemini.

    Returns:
        ChatGoogleGenerativeAI: Uma instância do modelo de linguagem pronto para uso.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        # Temperatura 0 para respostas mais determinísticas e factuais,
        # ideal para análise de dados onde a precisão é crucial.
        temperature=0,
        convert_system_message_to_human=True
    )

@st.cache_data
def load_csv(uploaded_file):
    """
    Carrega o arquivo CSV enviado pelo usuário em um DataFrame do Pandas.
    O cache (@st.cache_data) evita recarregar e processar o mesmo arquivo a
    cada pergunta, melhorando a performance da aplicação.

    Args:
        uploaded_file: O objeto de arquivo carregado via Streamlit.

    Returns:
        pd.DataFrame or None: O DataFrame carregado ou None em caso de erro.
    """
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo CSV: {e}")
        return None

def get_session_history(session_id: str):
    """
    Obtém ou cria um histórico de chat para uma sessão específica.
    Utiliza o dicionário global 'store' para manter históricos separados
    para cada sessão de usuário.

    Args:
        session_id (str): O identificador único da sessão de chat.

    Returns:
        InMemoryChatMessageHistory: O objeto de histórico da sessão.
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def initialize_session_state():
    """
    Inicializa as variáveis de estado da sessão do Streamlit se elas não existirem.
    Isso garante que o histórico de mensagens e o ID da sessão persistam
    entre as interações do usuário na mesma sessão.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        # Cria um ID de sessão único baseado no timestamp atual.
        st.session_state.session_id = f"session_{pd.Timestamp.now().timestamp()}"

def clear_chat_history():
    """
    Limpa o histórico de mensagens e reseta o estado do arquivo carregado.
    Força a criação de um novo ID de sessão para efetivamente "esquecer"
    a conversa anterior.
    """
    # Gera um novo ID de sessão para "esquecer" o histórico anterior
    st.session_state.session_id = f"session_{pd.Timestamp.now().timestamp()}"
    st.session_state.messages = []
    if 'df_loaded' in st.session_state:
        del st.session_state['df_loaded']
    st.success("A conversa foi reiniciada!")

def initialize_agent(llm, df):
    """
    Cria e configura o agente LangChain para análise de DataFrame.

    Args:
        llm: A instância do modelo de linguagem (LLM).
        df (pd.DataFrame): O DataFrame a ser analisado.

    Returns:
        RunnableWithMessageHistory: O agente configurado com memória.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", AGENT_PERSONA),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        prompt=prompt_template,
        verbose=False,  # Mantido como False para uma UI limpa.
        # ATENÇÃO: Habilitar código perigoso é necessário para que o agente execute
        # código Python gerado por ele mesmo. Use isso com cautela, idealmente em
        # ambientes controlados, pois permite a execução de código arbitrário.
        allow_dangerous_code=True,
        # Este argumento ajuda o agente a se recuperar de erros de formatação
        # na resposta do LLM, tornando-o mais robusto.
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

    return RunnableWithMessageHistory(
        agent,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

def render_sidebar():
    """
    Renderiza a barra lateral com as configurações da aplicação.

    Returns:
        tuple: Contém a chave de API e o arquivo carregado.
    """
    with st.sidebar:
        st.header("Configurações")

        try:
            google_api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("Chave de API carregada com segurança!")
        except (KeyError, FileNotFoundError):
            st.warning("Chave de API não encontrada nos segredos do Streamlit.")
            google_api_key = st.text_input(
                "Insira sua Chave de API do Google Gemini",
                type="password",
                help="Você pode configurar a chave de forma permanente no arquivo .streamlit/secrets.toml"
            )

        uploaded_file = st.file_uploader(
            "Carregue seu arquivo CSV",
            type="csv"
        )

        st.button("Sair e Limpar Conversa", on_click=clear_chat_history, use_container_width=True)

        st.info(
            """
            **Como usar:**
            1. Insira sua **API Key** do Google AI Studio.
            2. Carregue um **arquivo CSV** para análise.
            3. **Faça suas perguntas** no chat principal.

            **Exemplos de perguntas:**
            - `Quais são os tipos de dados numéricos, categóricos, etc..? Existes dados ausentes ou dupplicados?`
            - `Qual a variável ou variáveis pode ou podem ser ou serem usada(s) para gerar um gráfico de outliers?`
            - `Gere o(s) gráfico(s) de outliers da variável ou variáveis que podem ser ou serem usada(s). Caso tenha mais de um coloque lado a lado.`
            - `Quais são as estatísticas descritivas básicas? (média, mediana, moda, desvio padrão, etc.)`
            - `Quais as principais conclusões que você tira destes dados?`
            """
        )
    return google_api_key, uploaded_file

def handle_chat_interaction(agent_with_memory):
    """
    Gerencia a entrada do usuário, a invocação do agente e a exibição da resposta.

    Args:
        agent_with_memory: O agente LangChain com memória.
    """
    if prompt := st.chat_input("Faça sua pergunta sobre os dados..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analisando os dados e gerando a resposta..."):
                try:
                    # Limpa qualquer figura Matplotlib anterior para evitar sobreposição
                    # de gráficos entre diferentes perguntas.
                    plt.clf()
                    
                    config = {"configurable": {"session_id": st.session_state.session_id}}
                    response = agent_with_memory.invoke({"input": prompt}, config=config)
                    output = response["output"]

                    # Captura a figura gerada pelo Matplotlib, se houver.
                    # fig.get_axes() retorna True se a figura contiver algum eixo (plot).
                    fig = plt.gcf()
                    if fig.get_axes():
                        st.pyplot(fig)
                        # Anexa o gráfico à mensagem para ser re-renderizado corretamente.
                        ai_message = AIMessage(content=output, additional_kwargs={"plot": fig})
                    else:
                        st.markdown(output)
                        ai_message = AIMessage(content=output)

                    st.session_state.messages.append(ai_message)

                except Exception as e:
                    error_message = f"Ocorreu um erro durante a análise: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append(AIMessage(content=error_message))


# ==============================================================================
# FLUXO PRINCIPAL DA APLICAÇÃO
# ==============================================================================

def main():
    """
    Função principal que executa a aplicação Streamlit.
    """
    st.set_page_config(
        page_title="Agente de Análise de Dados CSV",
        page_icon="✨",
        layout="wide"
    )

    st.title("✨ Agente Gemini para Análise de Dados CSV")
    st.write(
        "**Bem-vindo!** Este agente usa o poder do Google Gemini para analisar arquivos CSV. "
        "Para começar, insira sua chave de API do Google na barra lateral, carregue seu arquivo CSV e faça suas perguntas."
    )

    # Inicializa e renderiza os componentes da UI e obtém as configurações.
    google_api_key, uploaded_file = render_sidebar()
    initialize_session_state()

    # Verifica se os pré-requisitos para iniciar a análise foram atendidos.
    if not google_api_key:
        st.warning("Por favor, insira sua chave de API na barra lateral para começar.")
        return

    if uploaded_file is None:
        st.info("Aguardando o carregamento de um arquivo CSV...")
        return

    # Exibe o histórico do chat na interface a cada nova interação.
    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)
            # Se a mensagem da IA tiver um gráfico associado, exibe-o.
            if "plot" in message.additional_kwargs:
                st.pyplot(message.additional_kwargs["plot"])

    # Carrega os dados do CSV. A função é cacheada para eficiência.
    df = load_csv(uploaded_file)

    if df is not None:
        # Exibe uma prévia dos dados apenas uma vez por arquivo carregado.
        if not st.session_state.get('df_loaded', False):
            st.success("Arquivo CSV carregado com sucesso! Amostra dos dados:")
            st.dataframe(df.head())
            st.session_state.df_loaded = True

        try:
            # Inicializa o LLM e o agente de análise.
            llm = get_llm(google_api_key)
            agent_with_memory = initialize_agent(llm, df)
            
            # Processa a interação do chat.
            handle_chat_interaction(agent_with_memory)

        except Exception as e:
            st.error(f"Ocorreu um erro crítico ao inicializar o agente: {e}")

# Ponto de entrada do script.
if __name__ == "__main__":
    main()