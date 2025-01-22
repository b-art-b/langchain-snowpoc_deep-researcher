# Import python packages
import os
import _snowflake
import streamlit as st
from snowflake.snowpark.context import get_active_session
from langchain_snowpoc.llms import SQLCortex
import functools

secret = _snowflake.get_generic_secret_string("my_tavily_key")
os.environ["TAVILY_API_KEY"] = secret

MODEL_LLM = "claude-3-5-sonnet"
HISTORY_TABLE = "DEEP_RESEARCHER_DB.PUBLIC.HISTORY"


# Get the current credentials
session = get_active_session()

if "result" not in st.session_state:
    st.session_state.result = ""

if "loading_historical" not in st.session_state:
    st.session_state.loading_historical = False
    st.session_state.input_question_disabled = False

if "research_id" not in st.session_state:
    st.session_state.research_id = None


def save_results_and_clear_input_question():
    q_insert = f"""INSERT INTO {HISTORY_TABLE}(title, content) VALUES
        ($${question}$$, $${st.session_state.result}$$);"""
    inserted_rows = 0
    try:
        inserted_rows = session.sql(q_insert).collect()[0]["number of rows inserted"]

    except Exception as ex:
        if inserted_rows != 1:
            raise Exception(f"Something went wrong while saving data.\n{ex}")
    st.session_state.result = ""
    st.session_state.input_question = ""


def load_historical(research_id):
    res = session.sql(
        f"SELECT research_id, title, content FROM {HISTORY_TABLE} where research_id={research_id}"
    ).collect()[0]
    st.session_state.loading_historical = True
    st.session_state.research_id = res["RESEARCH_ID"]
    st.session_state.input_question_disabled = True
    st.session_state.input_question = res["TITLE"]
    st.session_state.result = res["CONTENT"]


def new_search():
    st.session_state.research_id = None
    st.session_state.loading_historical = False
    st.session_state.result = ""
    st.session_state.input_question = ""
    st.session_state.input_question_disabled = False


def delete_search():
    if st.session_state.research_id:
        session.sql(
            f"""DELETE FROM {HISTORY_TABLE} 
        WHERE RESEARCH_ID={st.session_state.research_id}"""
        ).collect()
    new_search()


def st_progress(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        st.toast(fun.__name__)
        return fun(*args, **kwargs)

    return wrapper


# Write directly to the app
st.title("Deep Researcher :diving_mask:")

question = st.text_input(
    "What topic would you like me to research?",
    key="input_question",
    disabled=st.session_state.input_question_disabled,
)

c1, c2, c3 = st.columns(3)
btn_new = c1.button("New", on_click=new_search, use_container_width=True)
btn_save = c2.button(
    "Save", on_click=save_results_and_clear_input_question, use_container_width=True
)
btn_delete = c3.button("Delete", on_click=delete_search, use_container_width=True)

answer = st.empty()
answer.write(st.session_state.result)


with st.sidebar:
    st.header("Manage")
    no_of_web_search_loops = st.slider("Number of web search loops", min_value=1, max_value=10)
    for row in session.sql(
        f"SELECT research_id, title FROM {HISTORY_TABLE} ORDER BY when_added DESC"
    ).collect():
        st.button(
            row["TITLE"],
            key=row["RESEARCH_ID"],
            kwargs={"research_id": row["RESEARCH_ID"]},
            type="secondary",
            use_container_width=True,
            on_click=load_historical,
        )


# LLM
llm = SQLCortex(session=session, model=MODEL_LLM, options={"temperature": 0, "max_tokens": 4096})


##########################################################################################
#                                                                                  PROMPTS
query_writer_instructions = """Your goal is to generate targeted web search query.

The query will gather information related to a specific topic.

Topic:
{research_topic}

Return your query as a JSON object:
{{
    "query": "string",
    "aspect": "string",
    "rationale": "string"
}}
"""

summarizer_instructions = """Your goal is to generate a high-quality summary of the web search results.

When EXTENDING an existing summary:
1. Seamlessly integrate new information without repeating what's already covered
2. Maintain consistency with the existing content's style and depth
3. Only add new, non-redundant information
4. Ensure smooth transitions between existing and new content

When creating a NEW summary:
1. Highlight the most relevant information from each source
2. Provide a concise overview of the key points related to the report topic
3. Emphasize significant findings or insights
4. Ensure a coherent flow of information

In both cases:
- Focus on factual, objective information
- Maintain a consistent technical depth
- Avoid redundancy and repetition
- DO NOT use phrases like "based on the new results" or "according to additional sources"
- DO NOT add a preamble like "Here is an extended summary ..." Just directly output the summary.
- DO NOT add a References or Works Cited section.
"""

reflection_instructions = """You are an expert research assistant analyzing a summary about {research_topic}.

Your tasks:
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered

Ensure the follow-up question is self-contained and includes necessary context for web search.

Return your analysis as a JSON object:
{{
    "knowledge_gap": "string",
    "follow_up_query": "string"
}}"""

##########################################################################################
#                                                                            CONFIGURATION
import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
from dataclasses import dataclass


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""

    max_web_research_loops: int = no_of_web_search_loops

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})


##########################################################################################
#                                                                                    STATE
import operator
from dataclasses import dataclass, field
from typing_extensions import TypedDict, Annotated


@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None)  # Report topic
    search_query: str = field(default=None)  # Search query
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)  # Research loop count
    running_summary: str = field(default=None)  # Final report


@dataclass(kw_only=True)
class SummaryStateInput(TypedDict):
    research_topic: str = field(default=None)  # Report topic


@dataclass(kw_only=True)
class SummaryStateOutput(TypedDict):
    running_summary: str = field(default=None)  # Final report


##########################################################################################
#                                                                                    UTILS
from langsmith import traceable
from tavily import TavilyClient


@st_progress
def deduplicate_and_format_sources(
    search_response, max_tokens_per_source, include_raw_content=True
):
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results

    Returns:
        str: Formatted string with deduplicated sources
    """

    # st.toast('deduplicate_and_format_sources')
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += (
                f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
            )

    return formatted_text.strip()


@st_progress
def format_sources(search_results):
    """Format search results into a bullet-point list of sources.

    Args:
        search_results (dict): Tavily search response containing results

    Returns:
        str: Formatted string with sources and their URLs
    """
    # st.toast('format_sources')
    return "\n".join(
        f"* {source['title']} : {source['url']}" for source in search_results["results"]
    )


@st_progress
@traceable
def tavily_search(query, include_raw_content=True, max_results=3):
    """Search the web using the Tavily API.

    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        max_results (int): Maximum number of results to return

    Returns:
        dict: Tavily search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""
    # st.toast('tavily_search')
    tavily_client = TavilyClient()
    return tavily_client.search(
        query, max_results=max_results, include_raw_content=include_raw_content
    )


##########################################################################################
#                                                                                   RABBIT
import json

from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph


# Nodes
@st_progress
def generate_query(state: SummaryState):
    """Generate a query for web search"""
    # st.toast('generate_query')

    # Format the prompt
    query_writer_instructions_formatted = query_writer_instructions.format(
        research_topic=state.research_topic
    )

    # Generate a query
    result = llm.invoke(
        [
            SystemMessage(content=query_writer_instructions_formatted),
            HumanMessage(content=f"Generate a query for web search:"),
        ]
    )
    content = json.loads(result.content)["choices"][0]["messages"]
    query = json.loads(content)

    return {"search_query": query["query"]}


@st_progress
def web_research(state: SummaryState):
    """Gather information from the web"""
    # st.toast('web_research')

    # Search the web
    search_results = tavily_search(state.search_query, include_raw_content=True, max_results=1)

    # Format the sources
    search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000)
    return {
        "sources_gathered": [format_sources(search_results)],
        "research_loop_count": state.research_loop_count + 1,
        "web_research_results": [search_str],
    }


@st_progress
def summarize_sources(state: SummaryState):
    """Summarize the gathered sources"""
    # st.toast('summarize_sources')

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"Extend the existing summary: {existing_summary}\n\n"
            f"Include new search results: {most_recent_web_research} "
            f"That addresses the following topic: {state.research_topic}"
        )
    else:
        human_message_content = (
            f"Generate a summary of these search results: {most_recent_web_research} "
            f"That addresses the following topic: {state.research_topic}"
        )

    # Run the LLM
    result = llm.invoke(
        [
            SystemMessage(content=summarizer_instructions),
            HumanMessage(content=human_message_content),
        ]
    )

    content = json.loads(result.content)["choices"][0]["messages"]

    running_summary = content
    return {"running_summary": running_summary}


@st_progress
def reflect_on_summary(state: SummaryState):
    """Reflect on the summary and generate a follow-up query"""
    # Generate a query
    # st.toast('reflect_on_summary')

    result = llm.invoke(
        [
            SystemMessage(
                content=reflection_instructions.format(research_topic=state.research_topic)
            ),
            HumanMessage(
                content=(
                    f"Identify a knowledge gap and generate a follow-up web search query"
                    " based on our existing knowledge: {state.running_summary}"
                )
            ),
        ]
    )

    content = json.loads(result.content)["choices"][0]["messages"]
    follow_up_query = json.loads(content)

    # Overwrite the search query
    return {"search_query": follow_up_query["follow_up_query"]}


@st_progress
def finalize_summary(state: SummaryState):
    """Finalize the summary"""
    # st.toast('finalize_summary')

    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}


@st_progress
def route_research(
    state: SummaryState, config: RunnableConfig
) -> Literal["finalize_summary", "web_research"]:
    """Route the research based on the follow-up query"""
    # st.toast('route_research')

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"


if question and not btn_save and not st.session_state.loading_historical:
    # Add nodes and edges
    with st.spinner("Running"):
        builder = StateGraph(
            SummaryState,
            input=SummaryStateInput,
            output=SummaryStateOutput,
            config_schema=Configuration,
        )
        builder.add_node("generate_query", generate_query)
        builder.add_node("web_research", web_research)
        builder.add_node("summarize_sources", summarize_sources)
        builder.add_node("reflect_on_summary", reflect_on_summary)
        builder.add_node("finalize_summary", finalize_summary)

        # Add edges
        builder.add_edge(START, "generate_query")
        builder.add_edge("generate_query", "web_research")
        builder.add_edge("web_research", "summarize_sources")
        builder.add_edge("summarize_sources", "reflect_on_summary")
        builder.add_conditional_edges("reflect_on_summary", route_research)
        builder.add_edge("finalize_summary", END)

        graph = builder.compile()

        ##########################################################################################
        #                                                                                ASSISTANT

        research_input = SummaryStateInput(research_topic=question)
        summary = graph.invoke(research_input)
        st.session_state.result = summary["running_summary"]
        answer.write(st.session_state.result)
        st.toast(":tada: Done!")
