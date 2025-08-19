# Add numpy for the segmentation agent
import numpy as np
from agents import DataIngestionAgent, EdaAgent, PerformanceDefinitionAgent, SegmentationAgent
from agents import  VariableCreationAgent, ModelDevelopmentAgent, ScorecardConversionAgent , DocumentationAgent
import pickle
import shutil
import json

# --- Stage 0: Define directories ---
INPUT_DATA_DIR = "./data/"
OUTPUT_DIR_AGENT1 = "./outputs/01_ingestion/"
OUTPUT_DIR_AGENT2 = "./outputs/02_eda/"
OUTPUT_DIR_AGENT3 = "./outputs/03_performance_definition/"
OUTPUT_DIR_AGENT4 = "./outputs/04_segmentation/" # New directory for the new agent
OUTPUT_DIR_AGENT5 = "./outputs/05_variable_creation/" # New directory
#OUTPUT_DIR_AGENT6 = "./outputs/06_feature_engineering/" # New directory
OUTPUT_DIR_AGENT7 = "./outputs/07_model_development/" # New directory (skipping 06)
OUTPUT_DIR_AGENT8 = "./outputs/08_scorecard_conversion/" # New directory
# Define all paths that the DocumentationAgent will need to read from
INPUT_DIRS_FOR_DOCS = {
    "eda": "./outputs/02_eda/",
    "perf_def": "./outputs/03_performance_definition/",
    "segmentation": "./outputs/04_segmentation/",
    "model_dev": "./outputs/07_model_development/",
    "scorecard": "./outputs/08_scorecard_conversion/"
}
OUTPUT_DIR_AGENT9 = "./outputs/09_documentation/" # Final output directory

# ==============================================================================
#           SKIPPING COMPLETED STAGES FOR FASTER DEVELOPMENT
# ==============================================================================
# print("--- SKIPPING STAGE 1: INGESTION (already complete) ---")
# ingestion_agent = DataIngestionAgent(
#     input_dir=INPUT_DATA_DIR,
#     output_dir=OUTPUT_DIR_AGENT1,
#     nrows=50000
# )
# ingestion_output_path = ingestion_agent.execute()
#
# print("\n[-- SKIPPING REVIEW GATE 1 --]")
#
# print("--- SKIPPING STAGE 2: EDA (already complete) ---")
# eda_agent = EdaAgent(
#     input_dir=OUTPUT_DIR_AGENT1,
#     output_dir=OUTPUT_DIR_AGENT2
# )
# eda_agent.execute()
#
# print("\n[-- SKIPPING REVIEW GATE 2 --]")
#
# print("--- SKIPPING STAGE 3: PERFORMANCE DEFINITION (already complete) ---")
# perf_def_agent = PerformanceDefinitionAgent(
#     input_dir=OUTPUT_DIR_AGENT1,
#     output_dir=OUTPUT_DIR_AGENT3
# )
# performance_def_output_path = perf_def_agent.execute()

#print("\n[-- Human Review Gate 3: Performance Definition Complete. Proceeding to Segmentation. --]")

# # --- Stage 4: Run Segmentation Agent ---
# # The input for this agent is the OUTPUT of the last completed data-modifying agent (Agent 3)
# segmentation_agent = SegmentationAgent(
#     input_dir=OUTPUT_DIR_AGENT3, # We use the output from the previous stage
#     output_dir=OUTPUT_DIR_AGENT4
# )
# segmentation_output_path = segmentation_agent.execute()

# print("\n[-- Human Review Gate 4: Segmentation Complete. Proceeding to Variable Creation. --]")

# # --- Stage 5: Run Variable Creation Agent ---
# # The input for this agent is the OUTPUT of the last completed data-modifying agent (Agent 4)
# variable_creation_agent = VariableCreationAgent(
#     input_dir=OUTPUT_DIR_AGENT4,
#     output_dir=OUTPUT_DIR_AGENT5
# )
# variable_creation_output_path = variable_creation_agent.execute()

# print("\n[-- Human Review Gate 5: Variable Creation Complete. Proceeding to Feature Engineering. --]")

# # --- Stage 6: Run Feature Engineering Agent ---
# feature_engineering_agent = FeatureEngineeringAgent(
#     input_dir=OUTPUT_DIR_AGENT5,
#     output_dir=OUTPUT_DIR_AGENT6,
#     iv_threshold=0.02
# )
# feature_engineering_output_path = feature_engineering_agent.execute()

# print("\nWorkflow Finished. Check the outputs folder.")

# ==============================================================================
#           SIMPLIFIED PATH: SKIPPING TO MODEL DEVELOPMENT
# ==============================================================================
# print("\n[-- Bypassing complex feature engineering. Proceeding directly to Model Development. --]")

# # --- Stage 7: Run Model Development Agent ---
# # The input for this agent is the OUTPUT of the Variable Creation agent (Agent 5)
# model_dev_agent = ModelDevelopmentAgent(
#     input_dir=OUTPUT_DIR_AGENT5,
#     output_dir=OUTPUT_DIR_AGENT7
# )
# model_dev_agent.execute()

# print("\nSimplified workflow finished.")


# # ==============================================================================
# #           FINAL STAGE: SCORECARD CONVERSION
# # ==============================================================================
# print("\n[-- Model Development Complete. Proceeding to Scorecard Conversion. --]")

# # --- Stage 8: Run Scorecard Conversion Agent ---
# # The input for this agent is the OUTPUT of the Model Development agent (Agent 7)
# scorecard_agent = ScorecardConversionAgent(
#     input_dir=OUTPUT_DIR_AGENT7,
#     output_dir=OUTPUT_DIR_AGENT8,
#     base_score=600,
#     base_odds=50,
#     pdo=20
# )
# scorecard_agent.execute()

# print("\nFull simplified workflow finished. Awaiting final documentation.")


# ==============================================================================
#           FINAL STAGE: DOCUMENTATION & REPORTING
# ==============================================================================
print("\n[-- Scorecard Conversion Complete. Proceeding to Final Documentation. --]")

# --- Stage 9: Run Documentation Agent ---
doc_agent = DocumentationAgent(
    input_dirs=INPUT_DIRS_FOR_DOCS,
    output_dir=OUTPUT_DIR_AGENT9
)
doc_agent.execute()

print("\n\n==============================================")
print("      A2A PIPELINE EXECUTION COMPLETE      ")
print("==============================================")
print("Check the final model package in ./outputs/09_documentation/deployment_package/")
