import streamlit as st
import requests
import json
import os
from typing import Optional, Dict, Any, List

# --- Configuration ---

FASTAPI_URL = "http://bot-api:8000"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Multi-Agent Bot Console",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'create_agent' 
if 'selected_agent_id' not in st.session_state:
    st.session_state.selected_agent_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {} 
if 'agents' not in st.session_state:
    st.session_state.agents = [] 
if 'uploaded_agent_config' not in st.session_state:
    st.session_state.uploaded_agent_config = {}

# --- Helper Functions for FastAPI Communication ---

def send_request_to_fastapi(method: str, endpoint: str, data: Optional[dict] = None):
    """Sends an HTTP request to the FastAPI backend."""
    url = f"{FASTAPI_URL}/{endpoint}"
    try:
        if method.lower() == 'post':
            response = requests.post(url, json=data)
        elif method.lower() == 'get':
            response = requests.get(url)
        elif method.lower() == 'delete':
            response = requests.delete(url)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None

        response.raise_for_status() 
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to FastAPI backend at {FASTAPI_URL}. Please ensure it is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"FastAPI error: {e.response.status_code} - {e.response.text}")
        return None
    except json.JSONDecodeError:
        st.error(f"Failed to decode JSON response from FastAPI: {response.text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def get_agents():
    """Fetches list of agents from FastAPI backend."""
    response = send_request_to_fastapi('get', 'agents/list')
    if response:
        st.session_state.agents = response 
    return st.session_state.agents

def create_agent_on_backend(agent_data: dict):
    """Sends new agent data to FastAPI backend to create an agent."""
    response = send_request_to_fastapi('post', 'agents/create', agent_data)
    if response:
        st.success(f"Agent '{response.get('name', 'Unknown')}' created successfully!")
        # Refresh the list of agents in the sidebar
        st.session_state.agents = get_agents()
        st.session_state.current_page = 'list_agents'
        st.rerun() 
    return response

def chat_with_agent_on_backend(agent_id: str, message: str):
    """Sends a chat message to a specific agent and gets a response."""
    endpoint = f"agents/{agent_id}/chat"
    data = {"message": message}
    response = send_request_to_fastapi('post', endpoint, data)
    if response:
        return response.get("response", "No response from agent.")
    return "Error: Could not get response from agent."

# --- UI Components ---

def get_default_secrets_template(llm_provider: str) -> Dict[str, Any]:
    """Provides a default JSON structure for secrets based on the selected LLM provider."""
    base_secrets = {
        "discord_bot_token": "YOUR_DISCORD_BOT_TOKEN",
        "telegram_api_id": "YOUR_TELEGRAM_API_ID",
        "telegram_api_hash": "YOUR_TELEGRAM_API_HASH",
        "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
        "serpapi_api_key": "YOUR_SERPAPI_API_KEY",
        "newsapi_org_api_key": "YOUR_NEWSAPI_ORG_API_KEY",
        "finnhub_api_key": "YOUR_FINNHUB_API_KEY",
        "quandl_api_key": "YOUR_QUANDL_API_KEY",
        "cohere_api_key": "YOUR_COHERE_API_KEY"
    }
    
    # Add LLM-specific keys
    if llm_provider == "groq":
        base_secrets["groq_api_key"] = "YOUR_GROQ_API_KEY"
    elif llm_provider == "google":
        base_secrets["google_api_key"] = "YOUR_GOOGLE_API_KEY"
    elif llm_provider == "openai":
        base_secrets["openai_api_key"] = "YOUR_OPENAI_API_KEY"
    elif llm_provider == "anthropic":
        base_secrets["anthropic_api_key"] = "YOUR_ANTHROPIC_API_KEY"
    
    return base_secrets

def create_agent_page():
    """Renders the 'Create New Agent' form."""
    st.title("➕ Create New Agent")
    st.write("Upload a JSON configuration file or fill the form to create a new AI agent.")

    # JSON Upload Option
    uploaded_file = st.file_uploader("Upload character.json file", type="json")
    if uploaded_file is not None:
        try:
            file_content = json.load(uploaded_file)
            st.session_state.uploaded_agent_config = file_content
            st.success("JSON file uploaded successfully! Review details below.")
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid JSON.")
            st.session_state.uploaded_agent_config = {} 

    st.subheader("Agent Details")
    with st.form("create_agent_form"):
        default_name = st.session_state.uploaded_agent_config.get('name', '')
        default_system_prompt = st.session_state.uploaded_agent_config.get('system', st.session_state.uploaded_agent_config.get('persona', ''))
        default_bio = ", ".join(st.session_state.uploaded_agent_config.get('bio', [])) if isinstance(st.session_state.uploaded_agent_config.get('bio'), list) else st.session_state.uploaded_agent_config.get('bio', '')
        default_lore = ", ".join(st.session_state.uploaded_agent_config.get('lore', [])) if isinstance(st.session_state.uploaded_agent_config.get('lore'), list) else st.session_state.uploaded_agent_config.get('lore', '')
        default_knowledge = ", ".join(st.session_state.uploaded_agent_config.get('knowledge', [])) if isinstance(st.session_state.uploaded_agent_config.get('knowledge'), list) else st.session_state.uploaded_agent_config.get('knowledge', '')
        default_message_examples = json.dumps(st.session_state.uploaded_agent_config.get('messageExamples', []), indent=2)
        default_style = json.dumps(st.session_state.uploaded_agent_config.get('style', {}), indent=2)

        # LLM Provider selection
        llm_providers = ["Groq", "Google", "OpenAI", "Anthropic", "Ollama"]
        default_llm_provider_from_config = st.session_state.uploaded_agent_config.get('modelProvider', 'Groq').capitalize()
        default_llm_provider_idx = llm_providers.index(default_llm_provider_from_config) if default_llm_provider_from_config in llm_providers else 0
        
        llm_provider = st.selectbox(
            "LLM Provider",
            options=llm_providers,
            index=default_llm_provider_idx,
            key="llm_provider_select"
        )
        
        # LLM Model input
        default_llm_model = st.session_state.uploaded_agent_config.get('settings', {}).get('model', '')
        llm_model = st.text_input("LLM Model (Optional)", value=default_llm_model, help="Specific model name (e.g., 'llama3-70b-8192', 'gemini-pro', 'claude-3-opus-20240229', 'gpt-4'). Leave blank for provider default.")
        
        name = st.text_input("Agent Name", value=default_name, help="A unique name for your agent.")

        uploaded_secrets = st.session_state.uploaded_agent_config.get('settings', {}).get('secrets', {})
        if uploaded_secrets:
            secrets_json_str_initial = json.dumps(uploaded_secrets, indent=2)
        else:
            secrets_json_str_initial = json.dumps(get_default_secrets_template(llm_provider.lower()), indent=2)

        secrets_json_str = st.text_area(
            "Secrets (JSON)", 
            value=secrets_json_str_initial, 
            height=250, 
            help="API keys for tools and LLMs. Fill in required keys for your chosen LLM provider and any tools you want to enable. Remove unused keys."
        )
        system_prompt = st.text_area("System Prompt", value=default_system_prompt, height=150, help="The core system instruction or prompt for the agent.")
        
        bio = st.text_area("Bio (comma-separated)", value=default_bio, height=100, help="A brief description of your agent's background or purpose. Enter comma-separated values.")
        lore = st.text_area("Lore (comma-separated)", value=default_lore, height=100, help="Background story or specific lore for the agent. Enter comma-separated values.")
        knowledge = st.text_area("Knowledge Areas (comma-separated)", value=default_knowledge, height=100, help="Specific domains or topics your agent is knowledgeable about. Enter comma-separated values.")
        message_examples_str = st.text_area("Message Examples (JSON Array)", value=default_message_examples, height=200, help="A JSON array of example conversations. Each example is an array of message objects. E.g., [[{'user': 'user1', 'content': {'text': 'Hello'}}, {'user': 'agent', 'content': {'text': 'Hi'}}]]")
        style_str = st.text_area("Style (JSON Object)", value=default_style, height=150, help="A JSON object defining the agent's conversational style. E.g., {'all': ['formal', 'concise']}.")
        
        submitted = st.form_submit_button("Create Agent")

        if submitted:
            try:
                secrets_dict = json.loads(secrets_json_str)
                message_examples_parsed = json.loads(message_examples_str) if message_examples_str.strip() else []
                style_parsed = json.loads(style_str) if style_str.strip() else {}

                bio_list = [item.strip() for item in bio.split(',') if item.strip()] if bio else []
                lore_list = [item.strip() for item in lore.split(',') if item.strip()] if lore else []
                knowledge_list = [item.strip() for item in knowledge.split(',') if item.strip()] if knowledge else []

                agent_data = {
                    "name": name,
                    "modelProvider": llm_provider.lower(), 
                    "settings": {
                        "model": llm_model if llm_model else "", 
                        "temperature": 0.7, 
                        "maxTokens": 8192, 
                        "secrets": secrets_dict,
                        "voice": None 
                    },
                    "system": system_prompt,
                    "bio": bio_list,
                    "lore": lore_list,
                    "knowledge": knowledge_list,
                    "messageExamples": message_examples_parsed,
                    "style": style_parsed
                }
                create_agent_on_backend(agent_data)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON in Secrets, Message Examples, or Style field: {e}. Please correct it.")
            except Exception as e:
                st.error(f"An error occurred during agent creation: {e}")

def chat_page(agent_id: str):
    """Renders the chat interface for a selected agent."""
    agents = get_agents() 
    agent_name = next((a['name'] for a in agents if a['id'] == agent_id), "Unknown Agent")
    st.title(f"💬 Chat with {agent_name}")

    if agent_id not in st.session_state.chat_history:
        st.session_state.chat_history[agent_id] = []

    for message in st.session_state.chat_history[agent_id]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Say something to your agent..."):
        # Add user message to history
        st.session_state.chat_history[agent_id].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.spinner("Agent is thinking..."):
            agent_response = chat_with_agent_on_backend(agent_id, prompt)

        # Add agent response to history
        st.session_state.chat_history[agent_id].append({"role": "assistant", "content": agent_response})
        with st.chat_message("assistant"):
            st.markdown(agent_response)

def list_agents_page():
    """Renders the list of available agents."""
    st.title("📋 Available Agents")
    agents = get_agents()

    if not agents:
        st.info("No agents created yet. Use the 'Create New Agent' option in the sidebar.")
    else:
        for agent in agents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(agent.get('name', 'N/A'))
                st.write(f"ID: `{agent.get('id', 'N/A')}`")
                st.write(f"Persona: {agent.get('persona', 'No persona provided.')}")
                st.write(f"Bio: {', '.join(agent.get('bio', [])) if isinstance(agent.get('bio'), list) else agent.get('bio', 'No bio provided.')}")
                st.write(f"Lore: {', '.join(agent.get('lore', [])) if isinstance(agent.get('lore'), list) else agent.get('lore', 'No lore provided.')}")
                st.write(f"Knowledge: {', '.join(agent.get('knowledge', [])) if isinstance(agent.get('knowledge'), list) else agent.get('knowledge', 'No knowledge provided.')}")
                st.markdown(f"**LLM Provider:** `{agent.get('modelProvider', 'N/A').capitalize()}`")
                st.markdown(f"**LLM Model:** `{agent.get('settings', {}).get('model', 'Default')}`")
                st.markdown(f"**Message Examples:**")
                st.json(agent.get('messageExamples', []))
                st.markdown(f"**Style:**")
                st.json(agent.get('style', {}))

            with col2:
                if st.button(f"Chat with {agent.get('name', 'Agent')}", key=f"chat_{agent.get('id')}"):
                    st.session_state.selected_agent_id = agent['id']
                    st.session_state.current_page = 'chat'
                    st.rerun()
            st.markdown("---")

# --- Sidebar Navigation ---
with st.sidebar:
    st.header("Agent Console")
    if st.button("➕ Create New Agent", use_container_width=True):
        st.session_state.current_page = 'create_agent'
        st.session_state.selected_agent_id = None 
        st.session_state.uploaded_agent_config = {} 
        st.rerun()

    st.markdown("---")
    st.subheader("Your Agents")

    # Fetch and display agents in sidebar
    agents_in_sidebar = get_agents()
    if not agents_in_sidebar:
        st.info("No agents yet.")
    else:
        for agent in agents_in_sidebar:
            if st.button(agent.get('name', 'Unnamed Agent'), key=f"sidebar_agent_{agent.get('id')}", use_container_width=True):
                st.session_state.selected_agent_id = agent['id']
                st.session_state.current_page = 'chat'
                st.rerun()

# --- Main Content Rendering ---
if st.session_state.current_page == 'create_agent':
    create_agent_page()
elif st.session_state.current_page == 'chat' and st.session_state.selected_agent_id:
    chat_page(st.session_state.selected_agent_id)
else: 
    list_agents_page()
