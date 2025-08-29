from typing import TypedDict, List, Optional, Dict

class PipelineState(TypedDict):
    user_request: str
    llm_response: Optional[Dict]
    current_artifact_path: Optional[str]
    tool_output_summary: Optional[str]
    agent_history: List[str]