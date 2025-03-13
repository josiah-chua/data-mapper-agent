import os
import streamlit as st
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
import pickle
import datetime
import glob
import json
from pathlib import Path

# LangChain and LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.sql_database.tool import (
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
    InfoSQLDatabaseTool
)
from langchain_core.messages.system import SystemMessage
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel
import json

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Data Mapping Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state for app navigation
if "page" not in st.session_state:
    st.session_state.page = "landing"

# Initialize session state for database selection
if "selected_databases" not in st.session_state:
    st.session_state.selected_databases = {}

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "waiting_for_feedback" not in st.session_state:
    st.session_state.waiting_for_feedback = False

if "current_tool_call_id" not in st.session_state:
    st.session_state.current_tool_call_id = None

if "current_message" not in st.session_state:
    st.session_state.current_message = None

# Define state for LangGraph
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]  # basically passes the history through

# Define LLM instances
@st.cache_resource
def load_llm():
    llm = AzureChatOpenAI(
        api_key=os.environ['AZURE_OPENAI_KEY'],
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_ID'],
        api_version=os.environ['AZURE_OPENAI_API_VERSION'],
        temperature=0,
        max_tokens=4096,
        timeout=60,
        max_retries=2,
    )
    return llm

# Initialize databases - Make this more dynamic
@st.cache_resource
def load_all_databases():
    """Dynamically load all available SQLite databases in the current directory"""
    databases = {}
    # Find all SQLite database files in the current directory
    db_files = [f for f in os.listdir('.') if f.endswith('.db')]
    
    for db_file in db_files:
        db_name = os.path.splitext(db_file)[0]  # Get name without extension
        try:
            # Connect to the database
            db = SQLDatabase.from_uri(f"sqlite:///{db_file}")
            databases[db_name] = db
        except Exception as e:
            st.error(f"Failed to load database {db_file}: {str(e)}")
    
    return databases

# Function to clear database cache
def refresh_databases():
    """Clear the database cache to force reloading of all databases"""
    # Clear the load_all_databases cache
    load_all_databases.clear()
    # Set flag to force page refresh
    st.session_state.rerun_needed = True

# Function to load only selected databases
def get_selected_databases(all_dbs):
    """Return only the databases selected by the user"""
    if not st.session_state.selected_databases:
        return {}
        
    selected_dbs = {}
    for db_name, selected in st.session_state.selected_databases.items():
        if selected and db_name in all_dbs:
            selected_dbs[db_name] = all_dbs[db_name]
    
    return selected_dbs

# Initialize tools - update to handle dynamic database loading
def initialize_tools(_databases, _llm):
    """Initialize SQL tools for all available databases"""
    all_tools = []
    
    for db_name, db in _databases.items():
        # Create tools for this database
        list_tool = ListSQLDatabaseTool(db=db)
        info_tool = InfoSQLDatabaseTool(db=db)
        query_checker_tool = QuerySQLCheckerTool(db=db, llm=_llm)
        query_tool = QuerySQLDatabaseTool(db=db)
        
        # Set unique names for each tool
        list_tool.name = f'sql_db_{db_name}_list_tables'
        info_tool.name = f'sql_db_{db_name}_schema'
        query_checker_tool.name = f'sql_db_{db_name}_query_checker'
        query_tool.name = f'sql_db_{db_name}_query'
        
        # Add to tools list
        all_tools.extend([list_tool, info_tool, query_checker_tool, query_tool])
    
    return all_tools

# System message creator function
def create_system_message(databases):
    # Get list of database names
    db_list = ", ".join(databases.keys()) if databases else "(no databases available)"
    
    return SystemMessage(content=f"""You are a data mapping assistant proficient with SQL whose job is to use the databases {db_list} that you are connected to, to match fields with the list of fields in a dataset.
There are 2 main tasks:
1. Match fields, find all possible matches from both databases and give a confidence rating of low, medium, or high based on how confident you are that that database table field maps to the given field.
2. Generate the SQL query to give you the final data table
Make sure to enforce these rules:
1. Do not hallucinate, only use information available from the tools or user
2. Ensure that if columns are joint from different databases/tables match the same field, convert them to the same format; for example, categories of countries might be (US, EU...) in one but in another it is (America, Europe ...) and numerical values are in the same precision/units.

Return the list of matching fields, and the SQL query as the final outputs.
""")

# Create agent
def create_agent(llm, tools, databases):
    # Create bound LLM with tools
    agent_llm = llm.bind_tools(tools)
    
    sys_msg = create_system_message(databases)
    
    # Define agent nodes
    def chatbot(state: State):
        return {"messages": [agent_llm.invoke([sys_msg] + state["messages"])]}
    
    class DBToolsNode:
        """A node that runs the tools requested in the last AIMessage."""
        def __init__(self, tools: list) -> None:
            self.tools_by_name = {tool.name: tool for tool in tools}

        def __call__(self, inputs: dict):
            if messages := inputs.get("messages", []):
                message = messages[-1]
            else:
                raise ValueError("No message found in input")
            outputs = []
            for tool_call in message.tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return {"messages": outputs}
    
    
    db_tools_node = DBToolsNode(tools=tools)
    
    def route_tools(state: State):
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "db_tools"
        return END
    
    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("db_tools", db_tools_node)
    graph_builder.add_conditional_edges("chatbot", route_tools)
    graph_builder.add_edge("db_tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    
    return graph_builder.compile()

# Create sessions directory if it doesn't exist
sessions_dir = Path("sessions")
sessions_dir.mkdir(exist_ok=True)

# Functions for session persistence
def save_session(session_name=None):
    """Save the current session state to disk"""
    if not session_name:
        # Generate a default name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"session_{timestamp}"
    
    # Ensure the filename has a .pkl extension
    if not session_name.endswith('.pkl'):
        session_name += '.pkl'
    
    # Create a dict with the session data we want to save
    session_data = {
        "messages": st.session_state.messages,
        "chat_history": st.session_state.chat_history,
        "waiting_for_feedback": st.session_state.waiting_for_feedback,
        "current_tool_call_id": st.session_state.current_tool_call_id,
        "current_message": st.session_state.current_message,
        "selected_databases": st.session_state.selected_databases,  # Save selected databases
    }
    
    # Save to file
    with open(sessions_dir / session_name, 'wb') as f:
        pickle.dump(session_data, f)
    
    return session_name

def load_session(session_name):
    """Load a session from disk"""
    try:
        # Load from file
        with open(sessions_dir / session_name, 'rb') as f:
            session_data = pickle.load(f)
        
        # Update session state
        st.session_state.messages = session_data["messages"]
        st.session_state.chat_history = session_data["chat_history"]
        st.session_state.waiting_for_feedback = session_data["waiting_for_feedback"]
        st.session_state.current_tool_call_id = session_data["current_tool_call_id"]
        st.session_state.current_message = session_data["current_message"]
        
        # Handle selected_databases - compatible with older session files that might not have this
        if "selected_databases" in session_data:
            st.session_state.selected_databases = session_data["selected_databases"]
        # Set flag to force page refresh
        st.session_state.rerun_needed = True
        
        return True
    except Exception as e:
        st.error(f"Error loading session: {str(e)}")
        return False

def get_available_sessions():
    """Get a list of available session files"""
    return [os.path.basename(f) for f in glob.glob(str(sessions_dir / "*.pkl"))]

def clear_session():
    """Clear the current session state"""
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.waiting_for_feedback = False
    st.session_state.current_tool_call_id = None
    st.session_state.current_message = None
    # Don't clear selected databases as they might want to start a new conversation with same DBs
    
    # Add a flag to indicate that a rerun is needed
    st.session_state.rerun_needed = True

# Navigation functions
def go_to_chat_page():
    # Check if any databases are selected
    if not any(st.session_state.selected_databases.values()):
        st.error("Please select at least one database before proceeding to chat.")
        return
    
    st.session_state.page = "chat"
    st.session_state.rerun_needed = True

def go_to_landing_page():
    st.session_state.page = "landing"
    st.session_state.rerun_needed = True

# Landing page function
def landing_page():
    st.title("Data Mapping Assistant - Database Selection")
    
    # Add refresh databases button
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        st.subheader("Select Databases")
    with col3:
        refresh_db_button = st.button("🔄 Refresh Databases", key="refresh_db_button")
        if refresh_db_button:
            refresh_databases()
    
    # Get all available databases
    all_databases = load_all_databases()
    
    if not all_databases:
        st.warning("No databases were found. Please make sure SQLite (.db) files are in the current directory.")
        return
    
    st.write("Choose which databases you want to use for the data mapping task:")
    
    # Initialize selected_databases with existing databases if not already set
    # Clean up stale entries in selected_databases
    keys_to_remove = [key for key in st.session_state.selected_databases if key not in all_databases]
    for key in keys_to_remove:
        del st.session_state.selected_databases[key]
    
    # Add new databases with default False selection
    for db_name in all_databases.keys():
        if db_name not in st.session_state.selected_databases:
            st.session_state.selected_databases[db_name] = False
    
    # Display database selection with additional info
    cols = st.columns(2)
    db_list = list(all_databases.keys())
    
    for i, db_name in enumerate(db_list):
        col_idx = i % 2
        with cols[col_idx]:
            try:
                db = all_databases[db_name]
                tables = db.get_usable_table_names()
                table_count = len(tables)
                table_info = f"{table_count} tables"
                
                # Display checkbox with db info
                st.session_state.selected_databases[db_name] = st.checkbox(
                    f"{db_name.capitalize()} ({table_info})", 
                    value=st.session_state.selected_databases.get(db_name, False),
                    key=f"db_select_{db_name}"
                )
                
                # Show table details in an expander
                with st.expander(f"View {db_name} tables"):
                    st.write(", ".join(tables))
            except Exception as e:
                st.error(f"{db_name.capitalize()}: Error loading database ({str(e)})")
    
    # Session management in landing page
    st.divider()
    st.subheader("Session Management")
    
    # Load session section
    available_sessions = get_available_sessions()
    if available_sessions:
        st.write("Load an existing session:")
        col1, col2 = st.columns([7, 3])
        with col1:
            selected_session = st.selectbox(
                "Select session", 
                available_sessions, 
                key="load_session_select_landing",
                label_visibility="collapsed"
            )
        with col2:
            if st.button("📂 Load Session", key="load_button_landing"):
                if load_session(selected_session):
                    st.success(f"Session loaded: {selected_session}")
                    # If the loaded session had database selections, we should keep those
                    st.session_state.rerun_needed = True
    else:
        st.info("No saved sessions available")
    
    # Continue to chat button
    st.divider()
    proceed_col1, proceed_col2 = st.columns([4, 1])
    with proceed_col2:
        st.button("Continue to Chat ➡️", on_click=go_to_chat_page, type="primary")

# Chat page function
def chat_page():
    # App header using native Streamlit components
    st.title("Data Mapping Agent", anchor=False)
    
    # Initialize components
    llm = load_llm()
    
    # Load all databases first
    all_databases = load_all_databases()
    
    # Then filter to only selected ones
    databases = get_selected_databases(all_databases)
    
    if not databases:
        st.error("No databases are selected. Please return to the database selection page.")
        st.button("⬅️ Return to Database Selection", on_click=go_to_landing_page)
        return
    
    # Initialize tools with selected databases
    db_tools = initialize_tools(databases, llm)
    
    # Create a card-like container for the intro section using native components
    with st.container():
        intro_container = st.container()
        with intro_container:
            st.write("This agent helps match fields between different databases and generates SQL queries for data transformation.")
        
        # Dynamically get database information - more efficient approach
        with st.expander("**Available Databases**", expanded=True):
            # Display each selected database information
            for db_name, db in databases.items():
                try:
                    tables = db.get_usable_table_names()
                    table_count = len(tables)
                    table_list = ", ".join(tables[:3]) + ("..." if table_count > 3 else "")
                    st.write(f"**{db_name.capitalize()} DB**: {table_count} tables ({table_list})")
                except Exception as e:
                    st.error(f"**{db_name.capitalize()} DB**: Unable to fetch tables ({str(e)})")
    
    # Add a visual separator
    st.divider()
    
    # Chat interface with a heading using native components
    st.subheader("Conversation", anchor=False)
    
    # Create a container for chat messages
    chat_container = st.container()
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
            elif message["role"] == "tool":
                with st.chat_message("tool"):
                    st.markdown(f"**{message['name']}**: {message['content']}")
    
    # Create agent after UI components
    agent = create_agent(llm, db_tools, databases)
    
    # Session management in sidebar
    st.sidebar.header("Session Management")
    
    # Add the database change button to the sidebar at the top level
    change_db_col1, change_db_col2 = st.sidebar.columns([2, 2])
    with change_db_col1:
        st.button("⬅️ Change Databases", on_click=go_to_landing_page, key="change_db_sidebar")
    with change_db_col2:
        st.button("🔄 Refresh DBs", on_click=refresh_databases, key="refresh_db_sidebar")
    
    # Add a separator in the sidebar
    st.sidebar.divider()
    
    # Save session - improved layout
    with st.sidebar.container():
        col1, col2 = st.columns([7, 3])
        with col1:
            session_name = st.text_input("Session name", key="session_name_input", 
                                        placeholder="Enter session name (optional)")
        with col2:
            st.write("")  # Add some spacing for alignment
            save_button = st.button("💾 Save", key="save_button")
        
        if save_button:
            saved_name = save_session(session_name)
            st.sidebar.success(f"Session saved as: {saved_name}")

    # Clear session - Make it more compact
    with st.sidebar.container():
        if st.button("🗑️ Clear Chat", key="clear_session_button", help="Delete current conversation"):
            clear_session()  # This sets rerun_needed flag
            st.sidebar.success("Chat cleared")
    
    # Instructions as expandable section in sidebar
    with st.sidebar.expander("📋 Instructions", expanded=False):
        st.markdown("""
        1. Enter the fields you want to map in the format: "table field: field1, field2, field3"
        2. The agent will match these fields to the available database columns
        3. Provide feedback when asked
        4. Get the final SQL query for your data transformation
        5. Save your session to resume later
        """)
    
    # Handle feedback if waiting for it
    if st.session_state.waiting_for_feedback:
        with st.chat_message("assistant"):
            if st.session_state.current_message:
                st.markdown(st.session_state.current_message.content)
                
            with st.form(key="feedback_form"):
                feedback = st.text_area("Please provide your feedback:", key="feedback_input")
                submit_button = st.form_submit_button("Submit Feedback")
                
                if submit_button and feedback:
                    # Create a tool response message with the user's feedback
                    tool_message = ToolMessage(
                        content=feedback,
                        name="AskHuman",
                        tool_call_id=st.session_state.current_tool_call_id
                    )
                    
                    # Add the feedback to the chat history
                    st.session_state.chat_history.append({"role": "user", "content": feedback})
                    
                    # Continue the conversation with the agent
                    with st.spinner("Processing feedback..."):
                        feedback_state = {"messages": st.session_state.messages + [tool_message]}
                        result = agent.invoke(feedback_state)
                        
                        # Update messages
                        st.session_state.messages = st.session_state.messages + [tool_message] + result["messages"]
                        
                        # Reset feedback state
                        st.session_state.waiting_for_feedback = False
                        st.session_state.current_tool_call_id = None
                        st.session_state.current_message = None
                        
                        # Add the response to chat history
                        final_message = result["messages"][-1]
                        st.session_state.chat_history.append({"role": "assistant", "content": final_message.content})
                        
                        # Set flag for rerun rather than calling st.rerun() directly in form callback
                        st.session_state.rerun_needed = True
    
    # Handle user input with a more descriptive placeholder
    if prompt := st.chat_input("Enter fields to map (e.g., 'table field: user id, email, payment method')"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with agent
        user_message = HumanMessage(content=prompt)
        
        # Save the message to session state
        if len(st.session_state.messages) == 0:
            st.session_state.messages = [user_message]
        else:
            st.session_state.messages.append(user_message)
        
        input_state = {"messages": [user_message] if len(st.session_state.messages) <= 1 else st.session_state.messages}
        
        with st.spinner("Thinking..."):
            # Run the agent
            for result in agent.stream(input_state, stream_mode="values"):
                result["messages"][-1].pretty_print()
            
            # Display response
            final_message = result["messages"][-1]
            
            # Add to session state messages
            st.session_state.messages += result["messages"]
            
            with st.chat_message("assistant"):
                st.markdown(final_message.content)
            st.session_state.chat_history.append({"role": "assistant", "content": final_message.content})

# Initialize rerun_needed flag if it doesn't exist
if "rerun_needed" not in st.session_state:
    st.session_state.rerun_needed = False

# Main app entry point
def main():
    # Handle rerun needed flag if it exists
    if 'rerun_needed' in st.session_state and st.session_state.rerun_needed:
        st.session_state.rerun_needed = False  # Reset the flag
        st.rerun()  # This rerun is outside of callback

    # Route to the appropriate page
    if st.session_state.page == "landing":
        landing_page()
    else:  # chat page
        chat_page()

if __name__ == "__main__":
    main()
