import subprocess
from pathlib import Path
from prefect import flow, task

# --- The "Brick": A Generic Task to Run Any Agent ---
@task(log_prints=True)
def run_agent_task(agent_name: str, input_dir: str, output_dir: str, extra_args: list = None):
    """
    A generic Prefect task that runs any of our containerized agents.
    """
    print(f"Preparing to run agent: {agent_name}...")
    
    # Define the base command to run our Docker container
    base_command = [
        "docker", "run", "--rm",
        # Mount the input directory as read-only
        "-v", f"{Path.cwd()}/{input_dir}:/app/{input_dir}:ro",
        # Mount the parent of the output directory
        "-v", f"{Path.cwd()}/{Path(output_dir).parent}:/app/{Path(output_dir).parent}",
        "a2a-credit-risk-pipeline", # The name of our image
        "python", "scripts/run_agent.py",
        agent_name,
        f"--input-dir", f"/app/{input_dir}",
        f"--output-dir", f"/app/{output_dir}"
    ]
    
    # Add any extra arguments, like --nrows
    if extra_args:
        base_command.extend(extra_args)
    
    # Run the command using subprocess
    # check=True will raise an exception if the command fails, which Prefect will catch
    try:
        subprocess.run(base_command, check=True, capture_output=True, text=True)
        print(f"Agent '{agent_name}' executed successfully.")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"ERROR executing agent '{agent_name}'!")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

# --- The "House": The Main Workflow Definition ---
@flow(name="A2A Credit Risk Pipeline")
def a2a_credit_risk_flow():
    """
    This flow defines the end-to-end credit risk modeling pipeline
    by chaining together our containerized agents.
    """
    print("Starting A2A Credit Risk Pipeline Flow...")

    # Define the directory structure for clarity
    ingestion_input = "data"
    ingestion_output = "outputs/01_ingestion"
    
    eda_output = "outputs/02_eda"
    perf_def_output = "outputs/03_performance_definition"
    segmentation_output = "outputs/04_segmentation"
    var_creation_output = "outputs/05_variable_creation"
    model_dev_output = "outputs/07_model_development"
    scorecard_output = "outputs/08_scorecard_conversion"
    docs_output = "outputs/09_documentation"

    # --- Define the Dependency Graph ---
    # Prefect automatically understands dependencies. When you pass the output
    # of one task as the input to another, it builds the graph.
    
    # Stage 1
    ingestion_result_dir = run_agent_task(
        "ingestion", 
        ingestion_input, 
        ingestion_output,
        extra_args=["--nrows", "50000"]
    )
    
    # Stage 2
    eda_result_dir = run_agent_task("eda", ingestion_result_dir, eda_output)
    
    # Stage 3
    perf_def_result_dir = run_agent_task("perf_def", ingestion_result_dir, perf_def_output)

    # Stage 4 - Depends on Stage 3
    segmentation_result_dir = run_agent_task("segmentation", perf_def_result_dir, segmentation_output)

    # Stage 5 - Depends on Stage 4
    var_creation_result_dir = run_agent_task("variable_creation", segmentation_result_dir, var_creation_output)

    # ... and so on for the rest of the pipeline
    # For brevity, I'll stop here, but you would continue the chain for all agents.

    print("A2A Credit Risk Pipeline Flow finished successfully!")


# To run this flow, execute this script from your terminal
if __name__ == "__main__":
    a2a_credit_risk_flow()