Agente de Análise de Dados com Gemini e Streamlit
Um agente de IA interativo que utiliza o poder do Google Gemini e LangChain para realizar Análise Exploratória de Dados (EDA) em arquivos CSV através de uma interface web construída com Streamlit.

Este projeto permite que usuários, mesmo sem conhecimento técnico em programação, carreguem seus próprios conjuntos de dados em formato CSV e façam perguntas em linguagem natural para obter insights, tabelas e gráficos.

Demo
(Nota: Você pode usar uma ferramenta como o ScreenToGif para gravar uma pequena demonstração da sua aplicação e substituir o link acima.)

✨ Funcionalidades Principais
Interface Web Interativa: Construído com Streamlit para uma experiência de usuário limpa e responsiva.

Upload de CSV: Permite que o usuário carregue qualquer arquivo CSV para análise.

Consultas em Linguagem Natural: Interaja com os dados fazendo perguntas em português, como "qual a correlação entre as variáveis?".

Análise com IA: Utiliza o Google Gemini como o cérebro por trás da análise, interpretando as perguntas e gerando o código de análise necessário.

Geração de Gráficos e Tabelas: Gera e exibe automaticamente visualizações de dados (gráficos de barras, histogramas, etc.) e tabelas para ilustrar os resultados.

Memória de Conversação: O agente se lembra do contexto das perguntas anteriores, permitindo consultas de acompanhamento.

🛠️ Tecnologias Utilizadas
Python 3.10+

Frameworks de IA e Dados:

LangChain: Para orquestrar a lógica do agente e a integração com o LLM.

langchain-google-genai: Para conectar ao modelo Gemini do Google.

Pandas: Para manipulação e análise dos dados do CSV.

Matplotlib: Para a geração dos gráficos.

Interface Web:

Streamlit: Para a criação da interface do usuário.

🚀 Como Executar o Projeto
Siga os passos abaixo para configurar e executar o projeto em sua máquina local.

Pré-requisitos
Python 3.10 ou superior

Git

1. Clone o Repositório
Bash

git clone https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
cd NOME_DO_REPOSITORIO
2. Crie e Ative um Ambiente Virtual
É uma boa prática usar um ambiente virtual para isolar as dependências do projeto.

Bash

# Criar o ambiente virtual
python -m venv .venv

# Ativar no Windows
.venv\Scripts\activate

# Ativar no macOS/Linux
source .venv/bin/activate
3. Instale as Dependências
Crie um arquivo chamado requirements.txt na raiz do projeto com o seguinte conteúdo:

requirements.txt

streamlit
pandas
matplotlib
langchain
langchain-google-genai
langchain-experimental
Em seguida, instale todas as bibliotecas com um único comando:

Bash

pip install -r requirements.txt
4. Configure sua API Key do Google
A aplicação precisa de uma chave de API do Google Gemini para funcionar. A maneira mais segura de configurá-la é usando os segredos do Streamlit.

Crie uma pasta chamada .streamlit na raiz do seu projeto.

Dentro dela, crie um arquivo chamado secrets.toml.

Adicione sua chave de API ao arquivo da seguinte forma:

.streamlit/secrets.toml

Ini, TOML

GOOGLE_API_KEY = "SUA_CHAVE_DE_API_DO_GEMINI_AQUI"
5. Execute a Aplicação
Com tudo configurado, inicie o servidor do Streamlit:

Bash

streamlit run app_desafio_extra.py
A aplicação será aberta automaticamente no seu navegador padrão.

💬 Exemplo de Uso
Após iniciar a aplicação, a interface web será exibida.

A chave de API será carregada automaticamente a partir do arquivo secrets.toml.

Use a barra lateral para carregar seu arquivo CSV.

Uma vez que o arquivo for carregado, uma prévia dos dados será mostrada.

Use a caixa de chat na parte inferior da página para fazer suas perguntas.

Exemplos de perguntas que você pode fazer:

Quais são os tipos de dados? Existem dados ausentes ou duplicados?

Qual a variável pode ser usada para gerar um gráfico de outliers?

Gere um gráfico de outliers para a variável X.

Quais são as estatísticas descritivas básicas? (média, mediana, moda, etc.)

Quais as principais conclusões que você tira destes dados?

📄 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
