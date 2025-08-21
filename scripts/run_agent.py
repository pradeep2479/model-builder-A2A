import argparse
import sys
from agents import (
    DataIngestionAgent,
    EdaAgent,
    PerformanceDefinitionAgent,
    SegmentationAgent,
    VariableCreationAgent,
    ModelDevelopmentAgent,
    ScorecardConversionAgent,
    DocumentationAgent
)

# This dictionary maps a command-line name to the actual Agent Class
AGENT_MAP = {
    "ingestion": DataIngestionAgent,
    "eda": EdaAgent,
    "perf_def": PerformanceDefinitionAgent,
    "segmentation": SegmentationAgent,
    "variable_creation": VariableCreationAgent,
    "model_dev": ModelDevelopmentAgent,
    "scorecard": ScorecardConversionAgent,
    "docs": DocumentationAgent
}

def main():
    """
    This is the main function that parses command-line arguments
    and executes the specified agent.
    """
    parser = argparse.ArgumentParser(description="A2A Agent Runner")
    
    # Define the command-line arguments we expect
    parser.add_argument("agent_name", choices=AGENT_MAP.keys(), help="The name of the agent to run.")
    parser.add_argument("--input-dir", required=True, help="Path to the input directory.")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory.")
    # Add an optional argument for sampling, useful for the ingestion agent
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to process (for ingestion).")

    args = parser.parse_args()

    print(f"--- Received request to run agent: {args.agent_name} ---")

    # Look up the correct Agent class from our map
    agent_class = AGENT_MAP[args.agent_name]
    
    # --- Instantiate and execute the agent ---
    # This is a bit of a special case for the docs agent which needs a dict of inputs
    if args.agent_name == "docs":
        print("Documentation agent requires special input handling (not implemented in this runner).")
        sys.exit(1) # Exit with an error
    
    # Handle the 'nrows' argument for the ingestion agent
    if args.agent_name == "ingestion":
        agent = agent_class(
            input_dir=args.input_dir, 
            output_dir=args.output_dir, 
            nrows=args.nrows
        )
    else:
        agent = agent_class(input_dir=args.input_dir, output_dir=args.output_dir)
        
    agent.execute()
    print(f"--- Agent {args.agent_name} finished successfully. ---")


if __name__ == "__main__":
    main()