import os
import shutil
import json
from typing import Literal

from agents import DataIngestionAgent, EdaAgent
from graph_state import PipelineState
from langgraph.graph import StateGraph, END

import vertexai
from vertexai.generative_models import GenerativeModel

# --- 1. The "Brain": The Reasoning Node and its Prompt ---
try:
    project_id = os.popen("gcloud config get-value project").read().strip()
    if not project_id:
        raise ValueError("GCP project ID not found.")
    vertexai.init(project=project_id, location="us-central1")
    gemini_model = GenerativeModel("gemini-2.0-flash-001")
    print("Gemini model initialized successfully.")
except Exception as e:
    print(f"FATAL: Could not initialize Gemini. Error: {e}")
    gemini_model = None

SYSTEM_PROMPT = """
You are "A2A-PM", the AI Project Manager for a credit risk modeling pipeline.
Your goal is to guide the workflow from start to finish by choosing the correct tool at each step based on the current state and user requests.

**Available Tools:**
1. `data_ingestion`: Runs the initial data loading and merging. Use this first.
2. `eda`: Runs exploratory data analysis on the ingested data. Use this after `data_ingestion`.
3. `human_review`: Pauses the workflow to ask the user for input. Use this when a decision or approval is needed.
4. `finish`: Ends the project. Use this when the user is satisfied or the task is complete.

**Current State:**
- Last user request: {user_request}
- Agent History: {agent_history}
- Last Tool Output Summary: {tool_output_summary}

Based on the state, decide the VERY NEXT tool to use.
Respond ONLY with a valid JSON object in the format: `{{"next_tool": "tool_name"}}`

--- EXAMPLES ---
User Request: "Start the pipeline"
History: []
Tool Output: "Pipeline started."
Your JSON Response:
{{"next_tool": "data_ingestion"}}

User Request: "proceed"
History: ["data_ingestion"]
Tool Output: "Data ingestion complete..."
Your JSON Response:
{{"next_tool": "eda"}}

User Request: "stop"
History: ["data_ingestion", "eda"]
Tool Output: "EDA complete..."
Your JSON Response:
{{"next_tool": "finish"}}
--- END EXAMPLES ---
"""

def ceo_reasoning_node(state: PipelineState) -> PipelineState:
    print("\n--- CEO is Thinking... ---")
    if not gemini_model:
        raise ConnectionError("Gemini model not initialized.")
    prompt = SYSTEM_PROMPT.format(
        user_request=state['user_request'],
        agent_history=state['agent_history'],
        tool_output_summary=state['tool_output_summary']
    )
    response = gemini_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    llm_decision = json.loads(response.text)
    print(f"CEO Decision: {llm_decision}")
    state['llm_response'] = llm_decision
    return state

# --- 2. The "Hands": The Tool-Executing Nodes ---
def ingestion_node(state: PipelineState) -> PipelineState:
    print("\n--- Running Tool: Data Ingestion ---")
    agent = DataIngestionAgent(input_dir="./data", output_dir="./outputs/01_ingestion", nrows=5000)
    output_path = agent.execute()
    state['current_artifact_path'] = output_path
    state['agent_history'].append("data_ingestion")
    state['tool_output_summary'] = f"Data ingestion complete. Artifact saved to {output_path}."
    state['llm_response'] = {"next_tool": "human_review"} # Force human review after tool runs
    return state

def eda_node(state: PipelineState) -> PipelineState:
    print("\n--- Running Tool: EDA ---")
    agent = EdaAgent(input_dir=state['current_artifact_path'], output_dir="./outputs/02_eda")
    output_path = agent.execute()
    state['agent_history'].append("eda")
    state['tool_output_summary'] = f"EDA complete. Report saved in {output_path}."
    state['llm_response'] = {"next_tool": "human_review"} # Force human review
    return state

def human_review_node(state: PipelineState) -> PipelineState:
    print("\n--- Pausing for Human Review ---")
    feedback = input("CEO is asking for input. What is your command? (e.g., 'proceed', 'stop'): ")
    state['user_request'] = feedback
    state['tool_output_summary'] = f"Human provided feedback: '{feedback}'"
    return state

# --- 3. The "Nervous System": The Router ---
def router(state: PipelineState) -> Literal["ceo_reasoning_node", "__end__"]:
    print("\n--- Routing ---")
    # After any tool, always go back to the CEO to think, unless the tool was to finish.
    if state['llm_response'].get("next_tool") == "finish":
        return END
    return "ceo_reasoning_node"

# --- 4. Assemble the Graph ---
if os.path.exists("./outputs"):
    shutil.rmtree("./outputs")
os.makedirs("./outputs", exist_ok=True)

workflow = StateGraph(PipelineState)
workflow.add_node("ceo_reasoning_node", ceo_reasoning_node)
workflow.add_node("data_ingestion", ingestion_node)
workflow.add_node("eda", eda_node)
workflow.add_node("human_review", human_review_node)
workflow.set_entry_point("ceo_reasoning_node")

# A simpler routing logic
workflow.add_conditional_edges(
    "ceo_reasoning_node",
    lambda state: state['llm_response'].get("next_tool"),
    {"data_ingestion": "data_ingestion", "eda": "eda", "human_review": "human_review", "finish": END}
)

# After a tool runs, always go back to the CEO
workflow.add_edge("data_ingestion", "ceo_reasoning_node")
workflow.add_edge("eda", "ceo_reasoning_node")
workflow.add_edge("human_review", "ceo_reasoning_node")

app = workflow.compile()

# --- 5. Run the Application ---
if __name__ == "__main__":
    print("\n*** Starting Intelligent A2A Pipeline (LangGraph + Gemini) ***")
    initial_request = "Start the modeling pipeline by ingesting the data."
    initial_state = { "user_request": initial_request, "llm_response": None, "current_artifact_path": None, "tool_output_summary": "Pipeline started.", "agent_history": [], }
    for event in app.stream(initial_state, {"recursion_limit": 25}):
        for key, value in event.items():
            print(f"--- Event: Node '{key}' finished ---")
    print("\n*** A2A Pipeline Finished ***")