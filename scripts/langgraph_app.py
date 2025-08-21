import os
import shutil
from typing import Literal

from agents import DataIngestionAgent, EdaAgent
from graph_state import PipelineState
from langgraph.graph import StateGraph, END

def setup_environment():
    print("--- Setting up clean environment ---")
    if os.path.exists("./outputs"):
        shutil.rmtree("./outputs")
    os.makedirs("./outputs", exist_ok=True)

# --- Nodes (your functions are correct) ---
def ingestion_node(state: PipelineState) -> PipelineState:
    print("\n--- Running Node: Data Ingestion ---")
    ingestion_agent = DataIngestionAgent(input_dir="./data", output_dir="./outputs/01_ingestion", nrows=50000)
    output_path = ingestion_agent.execute()
    state['current_artifact_path'] = output_path
    state['agent_history'].append("DataIngestionAgent")
    state['human_approval_needed'] = True
    return state

def eda_node(state: PipelineState) -> PipelineState:
    print("\n--- Running Node: EDA ---")
    eda_agent = EdaAgent(input_dir=state['current_artifact_path'], output_dir="./outputs/02_eda")
    output_path = eda_agent.execute()
    state['current_artifact_path'] = output_path
    state['agent_history'].append("EdaAgent")
    # For now, we will just end after this step
    state['human_feedback'] = "stop"
    return state

def human_review_node(state: PipelineState) -> PipelineState:
    print("\n--- Pausing for Human Review ---")
    print(f"Latest artifact is at: {state['current_artifact_path']}")
    feedback = input("Type 'approve' to continue, or 'stop' to end: ")
    state['human_feedback'] = feedback
    state['human_approval_needed'] = False # We've received feedback
    return state

# --- Router (your function is correct) ---
def router(state: PipelineState) -> Literal["run_eda", "human_review", "__end__"]:
    print("\n--- Routing ---")
    if state['human_approval_needed']:
        return "human_review"
    
    if state['human_feedback'] == "approve":
        last_agent = state['agent_history'][-1]
        if last_agent == "DataIngestionAgent":
            return "run_eda"
            
    return END

# --- Assemble the Graph (THE FIX IS HERE) ---
setup_environment()
workflow = StateGraph(PipelineState)

# 1. Add the nodes
workflow.add_node("ingestion", ingestion_node)
workflow.add_node("eda", eda_node)
workflow.add_node("human_review", human_review_node)

# 2. Set the entry point
workflow.set_entry_point("ingestion")

# 3. Define the conditional edges
# This is the central routing logic for the entire graph
workflow.add_conditional_edges(
    "ingestion",  # After ingestion, call the router
    router,
    {
        "human_review": "human_review",
        # We don't expect other paths yet, but they would go here
    }
)
workflow.add_conditional_edges(
    "human_review", # After human review, call the router again
    router,
    {
        "run_eda": "eda",
        END: END
    }
)
# After EDA, the flow should just end for now.
workflow.add_edge("eda", END)


# 5. Compile the graph
app = workflow.compile()

# --- Run the Application ---
if __name__ == "__main__":
    print("\n*** Starting A2A Pipeline using LangGraph ***")
    initial_state = { "current_artifact_path": "", "human_approval_needed": False, "human_feedback": None, "agent_history": [], }
    for event in app.stream(initial_state):
        for key, value in event.items():
            print(f"--- Event: Node '{key}' finished ---")