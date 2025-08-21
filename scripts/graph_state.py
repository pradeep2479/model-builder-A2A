from typing import TypedDict, List, Optional

class PipelineState(TypedDict):
    """
    This dictionary represents the shared state of our modeling pipeline.
    It's passed between all agents (nodes) in our graph.
    """
    # The path to the latest data artifact, updated by each agent
    current_artifact_path: str
    
    # A flag to indicate if the flow should pause for human review
    human_approval_needed: bool
    
    # Stores the feedback from the human reviewer
    human_feedback: Optional[str]

    # A log of which agents have run
    agent_history: List[str]