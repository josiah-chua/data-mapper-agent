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

# Initialize the explanation level setting with default of "full"
if "explanation_level" not in st.session_state:
    st.session_state.explanation_level = "full"

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
        temperature=0.1,
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
    
    # Get the current explanation level
    explanation_level = st.session_state.explanation_level
    
    if explanation_level == "none":
        return SystemMessage(content=f"""You are a data mapping assistant proficient with SQL. You are connected to the following databases {db_list}.
Based on the user's question, suggest columns from the various tables from the databases that you are connected.

When suggesting columns you need to fulfill these tasks:
1. Columns: Return only the columns needed for the task and the database they are from. DO NOT return columns that are not used in the final query
2. Join Columns: Return the common join columns to get the table only if needed
3. Generate the SQL queries only if needed to get the final table. Ensure that if columns are from tables in different databases provided the necessary syntax.

***Important***
Do not hallucinate, only use information retrieved from the SQL databases. If you cannot answer the query just say you cannot.
Only execute the SQL query if all the data is from the same database and if explicitly instructed by the user.

Deliverables: Columns, Join Columns, Queries

Return the output in this format:
...
""")

    elif explanation_level == "low":
        return SystemMessage(content=f"""You are a data mapping assistant proficient with SQL. You are connected to the following databases {db_list}.
Based on the user's question, suggest columns from the various tables from the databases that you are connected.

When suggesting columns you need to fulfill these tasks:
1. Columns: Return only the columns needed for the task and the database they are from. DO NOT return columns that are not used in the final query
2. Join Columns: Return the common join columns to get the table only if needed
3. Briefly analyze the matching columns for compatibility
4. Give a simple confidence score (High, Medium, Low)
5. Generate the SQL queries only if needed to get the final table. Ensure that if columns are from tables in different databases provided the necessary syntax.
6. Brief explanation on why this query is appropriate

***Important***
Do not hallucinate, only use information retrieved from the SQL databases. If you cannot answer the query just say you cannot.
Only execute the SQL query if all the data is from the same database and if explicitly instructed by the user.

Deliverables: Columns, Join Columns, Confidence score, Queries with brief explanation

Return the output in this format:
...
""")

    else:  # "full" explanation level (default)
        return SystemMessage(content=f"""You are a data mapping assistant proficient with SQL. You are connected to the following databases {db_list}.
Based on the user's question, suggest columns form the various tables from the databses that you are connected.

When suggesting columns you need fufill these tasks:
1. Columns: Return only the columns needed for the task and the database they are from. DO NOT return columns that are not used in the final query
2. Join Columns: Return the common join columns to get the table only if needed
3. Analyse the matching columns and their data, by checking type, key statistics (e.g. min, max, medium , mode, skew for numeric and sampled value counts for categorical). If there are discrepencies suggest posssible solutions
4. Confidence Scores: Give a confidence score (High, Medium, Low) based on the analysis
5. Generate the SQL queries only if needed to get the final table. Ensure that if columns are from tables in different databases provided the necessary syntax.
6. Explanation on the rational for this query and how it works.

***Important***
Do not hallucinate, only use information retrieved from the SQL databases. If you cannot answer the query just say you cannot.
When comparing columns data get only key statistics e.g. min, max, medium , mode, skew for numeric and sampled value counts for categorical.
Only execute the SQL query if all the data is from the same database and if explicitly instructed by the user.

Deliverables: Columns, Join Columns, Confidence score and analysis, Queries

Return the output in this format:
1. Columns:
    - <Database Name A>:
        - ```<Table Name A>.<column_name>```

        - ```<Table Name B>.<column_name>```

2. Join Columns & Confidence score

    - **<Database Name A>**: ```<Table Name 1>.<column_name i>``` & **<Database Name B>**: ```<Table Name 2>.<column_name ii>```
        - Score: **<Score>**
        - Reason: <Analysis>

    - **<Database Name C>**: ```<Table Name 3>.<column_name iii>``` & **<Database Name D>**: ```<Table Name 4>.<column_name iv>```
        - Score: **<Score>**
        - Reason: <Analysis>

4. Query

    ```<query A using database A data>```

    Query explanation

===== Example (The information in these are not representative of the databases it is just to demonstrate the format)=====
Question:
    Get me a table to find cost per region

Answer:
1. Columns:

    - finance:

        - ```cost_prices.product_id```

        - ```cost_prices.cost_price```

    - shop:

        - ```addresses.address_id```

        - ```addresses.city```

        - ```addresses.state```

        - ```orders.shipping_address_id```

        - ```orders.order_id```

        - ```order_items.amount```

        - ```order_items.product_id```

        - ```order_items.order_id```


2. Join Columns & Confidence Scores

    - **finance** ```cost_prices.product_id``` & **shop** ```order_items.product_id```
        
        - Score: **Medium**
        - Reason: ```cost_prices.product_id``` type TEXT and ```order_items.product_id``` of type REAL. Values present in both are similar, however the values in cost_prices.product_id have a s prefix infront of the number e.g. s1024323, s9920345

    - **shop** ```orders.order_id``` & **shop** ```order_items.order_id```

        - Score: **High**
        - Reason: Both are of type REAL. Both seem to contain the same unique order_id both are (Min: 1231, Median: 6832, Max: 9102, Unique Count: 3829) and the value count of each id are the same.

    - **shop** ```addresses.address_id``` & **shop** ```orders.shipping_address_id```
        
        - Score: **Low**
        - Reason: Both are of type TEXT. Seems to be refering to the same data based on the column names but numbale to verify if they are the same


3. Query

    '''
    SELECT 
        a.city,
        a.state,
        SUM(order_items.amount - cp.cost_price) AS total_profit,
    FROM 
        shop.order_items AS oi
    JOIN 
        finance.cost_prices AS cp ON oi.product_id = cp.product_id
    JOIN 
        shop.orders AS o ON oi.order_id = o.order_id
    JOIN 
        shop.addresses AS a ON o.shipping_address_id = a.address_id
    GROUP BY 
        a.city, a.state;
    ```

    This query is designed to calculate the total profit per region (by city and state) by comparing the revenue from each order item with its cost. Here's the breakdown:

    1. Join Strategy:

        - Order Items & Cost Prices:
        The join between shop.order_items (aliased as oi) and finance.cost_prices (aliased as cp) is done on product_id. This matches each order item with its corresponding cost price.

        - Order Items & Orders:
        The join between oi and shop.orders (aliased as o) uses order_id to ensure that each order item's data is linked to the correct order.

        - Orders & Addresses:
        The join between orders and shop.addresses (aliased as a) is performed on the shipping address ID (shipping_address_id and address_id), which provides the regional (city and state) information.

    2. Profit Calculation:

        - The expression (order_items.amount - cp.cost_price) calculates the profit for each order item by subtracting the cost price from the revenue amount.

        - The SUM(...) function aggregates these profit values for all order items within the same region.

    3. Grouping:

        - The GROUP BY a.city, a.state clause organizes the results by region, so that the sum of profits is calculated per city and state.

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
        "explanation_level": st.session_state.explanation_level,   # Save explanation level preference
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
            
        # Handle explanation_level - compatible with older session files that might not have this
        if "explanation_level" in session_data:
            st.session_state.explanation_level = session_data["explanation_level"]
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
    st.rerun()

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
    
    # Add explanation level selector to the landing page
    st.divider()
    st.subheader("Response Detail Level")
    
    # Create helpful descriptions for each explanation level
    explanation_options = {
        "none": "No Explanation - Just columns and queries",
        "low": "Brief Explanation - Basic confidence scores and brief rationale",
        "full": "Full Explanation - Detailed analysis with comprehensive explanations"
    }
    
    # Use selectbox for explanation level with the current value pre-selected
    selected_level = st.selectbox(
        "Choose how detailed you want the assistant's responses to be:",
        options=list(explanation_options.keys()),
        format_func=lambda x: explanation_options[x],
        index=list(explanation_options.keys()).index(st.session_state.explanation_level),
        key="explanation_level_landing"
    )
    
    # Update the session state if the user changes the explanation level
    if selected_level != st.session_state.explanation_level:
        st.session_state.explanation_level = selected_level
        st.success(f"Response detail level set to: {explanation_options[selected_level]}")
    
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
    
    # Display current explanation level (for reference only)
    st.sidebar.divider()
    explanation_labels = {
        "none": "No Explanation",
        "low": "Brief Explanation",
        "full": "Full Explanation"
    }
    st.sidebar.info(f"Response detail level: **{explanation_labels[st.session_state.explanation_level]}**\n\n"
                    f"To change this setting, return to the database selection page.")
    
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
    if prompt := st.chat_input("Enter your query"):
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
