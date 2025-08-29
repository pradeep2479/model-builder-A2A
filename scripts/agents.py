import pandas as pd
import numpy as np
import abc
import os
import json
import shutil
from ydata_profiling import ProfileReport

# Note: We are removing unused imports like pickle, sklearn for this simplified version.
# If you add back the model dev agent, you'll need to re-add those.

class BaseAgent(abc.ABC):
    def __init__(self, agent_name: str, input_dir: str, output_dir: str):
        self.agent_name = agent_name
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Initialized {self.agent_name}")

    def load_inputs(self):
        # Default load behavior, can be overridden
        pass

    def run(self, inputs):
        # Core logic to be implemented by subclasses
        pass

    def execute(self):
        print(f"--- Running {self.agent_name} ---")
        inputs = self.load_inputs()
        artifacts = self.run(inputs)
        self.save_outputs(artifacts)
        print(f"--- {self.agent_name} execution complete ---")
        print(f"Outputs saved in: {self.output_dir}")
        return self.output_dir

class DataIngestionAgent(BaseAgent):
    def __init__(self, input_dir: str, output_dir: str, nrows: int = None):
        super().__init__("DataIngestionAgent", input_dir, output_dir)
        self.nrows = nrows

    def load_inputs(self):
        return None # This agent reads from its own specified path

    def run(self, inputs):
        print("Loading and merging raw data files...")
        app_train = pd.read_csv(os.path.join(self.input_dir, 'application_train.csv'), nrows=self.nrows)
        bureau = pd.read_csv(os.path.join(self.input_dir, 'bureau.csv'))
        previous_app = pd.read_csv(os.path.join(self.input_dir, 'previous_application.csv'))
        
        # Aggregate and merge logic
        relevant_ids = app_train['SK_ID_CURR'].unique()
        bureau = bureau[bureau['SK_ID_CURR'].isin(relevant_ids)]
        previous_app = previous_app[previous_app['SK_ID_CURR'].isin(relevant_ids)]
        
        bureau_aggregates = bureau.groupby('SK_ID_CURR').agg({'SK_ID_BUREAU': 'count'}).reset_index()
        bureau_aggregates.columns = ['SK_ID_CURR', 'BUREAU_LOAN_COUNT']
        
        prev_app_aggregates = previous_app.groupby('SK_ID_CURR').agg({'SK_ID_PREV': 'count'}).reset_index()
        prev_app_aggregates.columns = ['SK_ID_CURR', 'PREV_APP_COUNT']
        
        merged_df = pd.merge(app_train, bureau_aggregates, on='SK_ID_CURR', how='left')
        merged_df = pd.merge(merged_df, prev_app_aggregates, on='SK_ID_CURR', how='left')
        return {"analytical_base_table": merged_df}

    def save_outputs(self, artifacts):
        output_file = os.path.join(self.output_dir, "analytical_base_table.parquet")
        artifacts["analytical_base_table"].to_parquet(output_file)

class EdaAgent(BaseAgent):
    def load_inputs(self):
        input_file = os.path.join(self.input_dir, "analytical_base_table.parquet")
        return pd.read_parquet(input_file).copy()

    def run(self, inputs):
        print("Generating EDA report...")
        profile = ProfileReport(inputs, title="Credit Risk EDA Report", explorative=True, minimal=True)
        return {"eda_report": profile}

    def save_outputs(self, artifacts):
        artifacts["eda_report"].to_file(os.path.join(self.output_dir, "eda_report.html"))