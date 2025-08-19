from agents import (
    DataIngestionAgent, EdaAgent, PerformanceDefinitionAgent, 
    SegmentationAgent, VariableCreationAgent, ModelDevelopmentAgent,
    ScorecardConversionAgent, DocumentationAgent
)
import os
import shutil
import json
import pickle
import numpy as np

def run_full_pipeline():
    """
    Orchestrates the execution of the full, simplified A2A pipeline.
    """
    print("==================================================")
    print("      STARTING A2A CREDIT RISK PIPELINE      ")
    print("==================================================")

    # --- Define all directories ---
    INPUT_DATA_DIR = "./data/"
    OUTPUT_DIR_AGENT1 = "./outputs/01_ingestion/"
    OUTPUT_DIR_AGENT2 = "./outputs/02_eda/"
    OUTPUT_DIR_AGENT3 = "./outputs/03_performance_definition/"
    OUTPUT_DIR_AGENT4 = "./outputs/04_segmentation/"
    OUTPUT_DIR_AGENT5 = "./outputs/05_variable_creation/"
    OUTPUT_DIR_AGENT7 = "./outputs/07_model_development/"
    OUTPUT_DIR_AGENT8 = "./outputs/08_scorecard_conversion/"
    OUTPUT_DIR_AGENT9 = "./outputs/09_documentation/"

    # --- Stage 1: Data Ingestion ---
    ingestion_agent = DataIngestionAgent(input_dir=INPUT_DATA_DIR, output_dir=OUTPUT_DIR_AGENT1, nrows=50000)
    ingestion_output_path = ingestion_agent.execute()

    # --- Stage 2: EDA ---
    eda_agent = EdaAgent(input_dir=ingestion_output_path, output_dir=OUTPUT_DIR_AGENT2)
    eda_agent.execute()

    # --- Stage 3: Performance Definition ---
    perf_def_agent = PerformanceDefinitionAgent(input_dir=ingestion_output_path, output_dir=OUTPUT_DIR_AGENT3)
    performance_def_output_path = perf_def_agent.execute()

    # --- Stage 4: Segmentation ---
    segmentation_agent = SegmentationAgent(input_dir=performance_def_output_path, output_dir=OUTPUT_DIR_AGENT4)
    segmentation_output_path = segmentation_agent.execute()

    # --- Stage 5: Variable Creation ---
    variable_creation_agent = VariableCreationAgent(input_dir=segmentation_output_path, output_dir=OUTPUT_DIR_AGENT5)
    variable_creation_output_path = variable_creation_agent.execute()

    # --- Stage 6 (Simplified): Model Development ---
    model_dev_agent = ModelDevelopmentAgent(input_dir=variable_creation_output_path, output_dir=OUTPUT_DIR_AGENT7)
    model_dev_output_path = model_dev_agent.execute()

    # --- Stage 7: Scorecard Conversion ---
    scorecard_agent = ScorecardConversionAgent(input_dir=model_dev_output_path, output_dir=OUTPUT_DIR_AGENT8)
    scorecard_agent.execute()

    # --- Stage 8: Documentation ---
    doc_agent = DocumentationAgent(
        input_dirs={
            "eda": OUTPUT_DIR_AGENT2, "perf_def": OUTPUT_DIR_AGENT3, "segmentation": OUTPUT_DIR_AGENT4,
            "model_dev": OUTPUT_DIR_AGENT7, "scorecard": OUTPUT_DIR_AGENT8
        },
        output_dir=OUTPUT_DIR_AGENT9
    )
    doc_agent.execute()

    print("\n\n==============================================")
    print("      A2A PIPELINE EXECUTION COMPLETE      ")
    print("==============================================")
    print(f"Check the final model package in {os.path.join(OUTPUT_DIR_AGENT9, 'deployment_package/')}")

if __name__ == "__main__":
    run_full_pipeline()