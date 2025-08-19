import pandas as pd
import numpy as np
import abc
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle
from optbinning import OptimalBinning
import shutil

# NEW: Import the profiling library
from ydata_profiling import ProfileReport

class BaseAgent(abc.ABC):
    """Abstract Base Class for our modeling agents."""
    def __init__(self, agent_name: str, input_dir: str, output_dir: str):
        self.agent_name = agent_name
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Initialized {self.agent_name}")

    @abc.abstractmethod
    def load_inputs(self):
        pass

    @abc.abstractmethod
    def run(self, inputs):
        """Core logic of the agent. Must return artifacts to be saved."""
        pass

    @abc.abstractmethod
    def save_outputs(self, artifacts):
        pass

    def execute(self):
        """Public method to run the full agent lifecycle."""
        print(f"\n--- Running {self.agent_name} ---")
        inputs = self.load_inputs()
        artifacts = self.run(inputs)
        self.save_outputs(artifacts)
        print(f"--- {self.agent_name} execution complete ---")
        print(f"Outputs saved in: {self.output_dir}")
        return self.output_dir

# --- (DataIngestionAgent class remains the same as before) ---
class DataIngestionAgent(BaseAgent):
    """
    Agent responsible for loading all raw data sources, performing
    aggregations, merging them, and producing a single analytical base table.
    """
    def __init__(self, input_dir: str, output_dir: str, nrows: int = None):
        super().__init__("DataIngestionAgent", input_dir, output_dir)
        self.nrows = nrows

    def load_inputs(self):
        """Loads all necessary CSV files."""
        print("Loading raw data files...")
        app_train = pd.read_csv(os.path.join(self.input_dir, 'application_train.csv'), nrows=self.nrows)
        bureau = pd.read_csv(os.path.join(self.input_dir, 'bureau.csv')) # Load full bureau for better joins
        previous_app = pd.read_csv(os.path.join(self.input_dir, 'previous_application.csv')) # Load full prev_app
        return {"app_train": app_train, "bureau": bureau, "previous_app": previous_app}

    def run(self, inputs):
        """Performs aggregations and merges."""
        print("Running aggregation and merging logic...")
        app_train = inputs['app_train']
        
        # Filter other tables to only include IDs present in the (potentially sampled) app_train
        relevant_ids = app_train['SK_ID_CURR'].unique()
        bureau = inputs['bureau'][inputs['bureau']['SK_ID_CURR'].isin(relevant_ids)]
        previous_app = inputs['previous_app'][inputs['previous_app']['SK_ID_CURR'].isin(relevant_ids)]

        # --- Bureau Aggregations ---
        bureau_aggregates = bureau.groupby('SK_ID_CURR').agg({
            'SK_ID_BUREAU': 'count',
            'CREDIT_DAY_OVERDUE': 'mean',
            'AMT_CREDIT_SUM_DEBT': 'sum'
        }).reset_index()
        bureau_aggregates.columns = ['SK_ID_CURR', 'BUREAU_LOAN_COUNT', 'BUREAU_AVG_OVERDUE', 'BUREAU_TOTAL_DEBT']

        # --- Previous Application Aggregations ---
        prev_app_filtered = previous_app[previous_app['NAME_CONTRACT_STATUS'].isin(['Approved', 'Refused'])]
        prev_app_aggregates = prev_app_filtered.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'count',
            'AMT_ANNUITY': 'mean',
            'AMT_CREDIT': 'mean',
        }).reset_index()
        prev_app_aggregates.columns = ['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_APP_AVG_ANNUITY', 'PREV_APP_AVG_CREDIT']
        
        # --- Merging ---
        print("Merging aggregated features...")
        merged_df = pd.merge(app_train, bureau_aggregates, on='SK_ID_CURR', how='left')
        merged_df = pd.merge(merged_df, prev_app_aggregates, on='SK_ID_CURR', how='left')
        
        print(f"Final merged shape: {merged_df.shape}")
        return {"analytical_base_table": merged_df}

    def save_outputs(self, artifacts):
        """Saves the final merged dataframe to a Parquet file."""
        output_file = os.path.join(self.output_dir, "analytical_base_table.parquet")
        artifacts["analytical_base_table"].to_parquet(output_file)
        print(f"Saved merged data to {output_file}")


# --- NEW: The EDA Agent ---
class EdaAgent(BaseAgent):
    """
    Agent responsible for performing EDA on the analytical base table
    and generating a comprehensive report for human review.
    """
    def __init__(self, input_dir: str, output_dir: str):
        super().__init__("EdaAgent", input_dir, output_dir)

    def load_inputs(self):
        """Loads the merged analytical base table from the previous agent."""
        print("Loading analytical base table...")
        input_file = os.path.join(self.input_dir, "analytical_base_table.parquet")
        print("Loading and de-fragmenting analytical base table...")
        df = pd.read_parquet(input_file).copy()
        return {"analytical_base_table": df}

    def run(self, inputs):
        """Generates the EDA profile report."""
        print("Generating EDA report...")
        df = inputs["analytical_base_table"]

        # --- Domain-specific check: Calculate bad rate ---
        bad_rate = df['TARGET'].mean()
        print(f"Calculated Bad Rate: {bad_rate:.2%}")
        
        # --- Automated report generation ---
        profile = ProfileReport(
            df,
            title="Credit Risk EDA Report",
            explorative=True
        )
        
        return {"eda_report": profile, "key_metrics": {"bad_rate": bad_rate}}

    def save_outputs(self, artifacts):
        """Saves the report to an HTML file and key metrics to JSON."""
        # Save HTML report
        report_path = os.path.join(self.output_dir, "eda_report.html")
        artifacts["eda_report"].to_file(report_path)
        print(f"Saved EDA HTML report to {report_path}")

        # Save key metrics
        metrics_path = os.path.join(self.output_dir, "eda_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(artifacts["key_metrics"], f, indent=4)
        print(f"Saved key metrics to {metrics_path}")


# --- NEW: The Performance Definition Agent ---
class PerformanceDefinitionAgent(BaseAgent):
    """
    Agent responsible for formally defining and validating the
    dependent variable (target) for the model.
    """
    def __init__(self, input_dir: str, output_dir: str):
        super().__init__("PerformanceDefinitionAgent", input_dir, output_dir)
        self.target_column = 'TARGET' # Define the target column name

    def load_inputs(self):
        """Loads the merged analytical base table from the ingestion agent."""
        print("Loading analytical base table...")
        input_file = os.path.join(self.input_dir, "analytical_base_table.parquet")
        # Add .copy() to prevent fragmentation warnings in later stages
        df = pd.read_parquet(input_file).copy()
        return {"analytical_base_table": df}

    def run(self, inputs):
        """Validates the target and creates definition artifacts."""
        print(f"Defining performance using column: '{self.target_column}'")
        df = inputs["analytical_base_table"]

        # --- Validation Step ---
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the dataset.")

        # --- Definition Logic ---
        bad_definition = "TARGET == 1 (Client with payment difficulties)"
        good_definition = "TARGET == 0 (All other cases)"
        bad_rate = df[self.target_column].mean()

        print(f"  - Bad Rate: {bad_rate:.2%}")
        print(f"  - Bad Definition: {bad_definition}")
        print(f"  - Good Definition: {good_definition}")

        # --- Create Artifacts ---
        # 1. Human-readable report
        report_content = f"""
# Performance Definition Report

This document specifies the dependent variable for the credit risk model.

- **Target Column Name**: `{self.target_column}`
- **Bad Definition**: A loan is considered "bad" if `{bad_definition}`.
- **Good Definition**: A loan is considered "good" if `{good_definition}`.
- **Observed Bad Rate**: `{bad_rate:.4f}`
"""

        # 2. Machine-readable summary
        summary = {
            "target_column": self.target_column,
            "bad_definition_rule": "TARGET == 1",
            "bad_rate": bad_rate
        }
        
        # The data artifact is the original dataframe, now formally "blessed"
        # as having a defined target. No columns are changed in this case.
        return {
            "data_with_definition": df,
            "definition_report": report_content,
            "definition_summary": summary
        }

    def save_outputs(self, artifacts):
        """Saves the data and the definition files."""
        # Save the dataset for the next agent
        data_path = os.path.join(self.output_dir, "data_with_performance_definition.parquet")
        artifacts["data_with_definition"].to_parquet(data_path)
        print(f"Saved dataset with defined target to {data_path}")

        # Save the human-readable report
        report_path = os.path.join(self.output_dir, "performance_definition_report.md")
        with open(report_path, 'w') as f:
            f.write(artifacts["definition_report"])
        print(f"Saved performance definition report to {report_path}")

        # Save the machine-readable summary
        summary_path = os.path.join(self.output_dir, "performance_definition_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(artifacts["definition_summary"], f, indent=4)
        print(f"Saved performance definition summary to {summary_path}")


# --- NEW: The Segmentation Agent ---
class SegmentationAgent(BaseAgent):
    """
    Agent responsible for identifying and validating customer segments
    to see if separate models are warranted.
    """
    def __init__(self, input_dir: str, output_dir: str):
        super().__init__("SegmentationAgent", input_dir, output_dir)

    def load_inputs(self):
        """Loads the data with performance definition."""
        print("Loading data with performance definition...")
        input_file = os.path.join(self.input_dir, "data_with_performance_definition.parquet")
        df = pd.read_parquet(input_file).copy()
        return {"data": df}

    def run(self, inputs):
        """Applies segmentation rules and analyzes segment characteristics."""
        print("Running customer segmentation...")
        df = inputs["data"]

        # --- Segmentation Logic: New vs. Existing ---
        # We use our previously engineered feature 'PREV_APP_COUNT'
        # .isna() checks for NaN values, which is our marker for a truly new customer.
        df['segment'] = np.where(df['PREV_APP_COUNT'].isna(), 'NEW_CUSTOMER', 'EXISTING_CUSTOMER')
        
        print("Segmentation rule applied: New vs. Existing Customer.")

        # --- Validation Step: Analyze bad rates per segment ---
        segment_analysis = df.groupby('segment')['TARGET'].agg(['count', 'mean']).rename(columns={'mean': 'bad_rate'})
        print("Segment Analysis:\n", segment_analysis)
        
        # --- Create Artifacts ---
        # 1. Machine-readable report of the analysis
        analysis_report = segment_analysis.reset_index().to_dict(orient='records')
        
        return {
            "segmented_data": df,
            "segment_analysis_report": analysis_report
        }

    def save_outputs(self, artifacts):
        """Saves the segmented data and the analysis report."""
        # Save the dataset with the new 'segment' column
        data_path = os.path.join(self.output_dir, "segmented_data.parquet")
        artifacts["segmented_data"].to_parquet(data_path)
        print(f"Saved segmented data to {data_path}")

        # Save the machine-readable analysis
        report_path = os.path.join(self.output_dir, "segment_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(artifacts["segment_analysis_report"], f, indent=4)
        print(f"Saved segment analysis report to {report_path}")


# --- NEW: The Variable Creation Agent ---
class VariableCreationAgent(BaseAgent):
    """
    Agent responsible for cleaning data, imputing missing values,
    and creating new derived features (ratios).
    """
    def __init__(self, input_dir: str, output_dir: str):
        super().__init__("VariableCreationAgent", input_dir, output_dir)

    def load_inputs(self):
        """Loads the segmented data from the previous agent."""
        print("Loading segmented data...")
        input_file = os.path.join(self.input_dir, "segmented_data.parquet")
        df = pd.read_parquet(input_file).copy()
        return {"data": df}

    def run(self, inputs):
        """Applies imputation and feature creation logic."""
        print("Running variable creation and transformation...")
        df = inputs["data"]

        # --- 1. Create New Ratio Features (NaNs may be created here) ---
        # We add a small epsilon (1e-6) to the denominator to prevent division by zero entirely.
        # This is a robust alternative to replacing inf values later.
        epsilon = 1e-6
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + epsilon)
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + epsilon)
        df['CREDIT_TERM'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + epsilon)
        print("Created 3 new ratio features.")

        # --- 2. Comprehensive Imputation (The Fix) ---
        # Select all categorical columns and fill missing values
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols.remove('segment') # Don't impute our segment column
        print(f"Imputing {df[categorical_cols].isnull().sum().sum()} missing values in categorical columns with 'Missing'.")
        df[categorical_cols] = df[categorical_cols].fillna('Missing')
        
        # Select ALL numerical columns (including the new ones) and fill missing values
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        exclude_cols = ['SK_ID_CURR', 'TARGET'] # Don't impute IDs or the target
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        print(f"Imputing {df[numerical_cols].isnull().sum().sum()} missing values in numerical columns with 0.")
        df[numerical_cols] = df[numerical_cols].fillna(0)

        # --- 3. Final Verification ---
        # We check for nulls in the entire dataframe except the target, which might have nulls in a different problem.
        data_cols = [col for col in df.columns if col != 'TARGET']
        remaining_missing = df[data_cols].isnull().sum().sum()
        
        if remaining_missing > 0:
            print(f"FATAL: {remaining_missing} missing values still remain after imputation!")
            # In a real pipeline, you might raise an exception here
            # raise ValueError(f"{remaining_missing} missing values remain.")
        else:
            print("Imputation complete. No missing values remain in the feature set.")

        return {"transformed_data": df}

    def save_outputs(self, artifacts):
        """Saves the fully transformed and cleaned dataset."""
        data_path = os.path.join(self.output_dir, "transformed_data.parquet")
        artifacts["transformed_data"].to_parquet(data_path)
        print(f"Saved transformed data to {data_path}")


# --- NEW: The Feature Engineering Agent ---
# class FeatureEngineeringAgent(BaseAgent):
#     """
#     Agent responsible for WOE/IV feature engineering and selection,
#     handled independently for each customer segment.
#     """
#     def __init__(self, input_dir: str, output_dir: str, iv_threshold: float = 0.02):
#         super().__init__("FeatureEngineeringAgent", input_dir, output_dir)
#         self.iv_threshold = iv_threshold

#     def load_inputs(self):
#         """Loads the transformed data from the previous agent."""
#         print("Loading transformed data...")
#         input_file = os.path.join(self.input_dir, "transformed_data.parquet")
#         df = pd.read_parquet(input_file).copy()
#         return {"data": df}

#     def run(self, inputs):
#         """
#         For each segment, performs optimal binning, calculates IV, selects features,
#         and transforms the data to its WOE values.
#         """
#         print("Running Feature Engineering (WOE/IV)...")
#         df = inputs["data"]
        
#         # --- Identify features to be binned ---
#         exclude_cols = ['SK_ID_CURR', 'TARGET', 'segment']
#         features = [col for col in df.columns if col not in exclude_cols]
#         categorical_features = [col for col in features if df[col].dtype == 'object']

#         segments = df['segment'].unique()
        
#         # Artifact containers
#         all_binned_data = []
#         all_iv_reports = []
#         all_binning_objects = {}

#         # --- Process each segment independently ---
#         for segment in segments:
#             print(f"\n--- Processing segment: {segment} ---")
#             segment_df = df[df['segment'] == segment].copy()
            
#             X = segment_df[features]
#             y = segment_df['TARGET']

#             # --- NEW: EXPLICIT DEBUGGING STEP ---
#             print("Verifying data integrity before binning...")
            
#             # Check for nulls in the target variable `y`
#             if y.isnull().any():
#                 print(f"FATAL: Found {y.isnull().sum()} nulls in TARGET variable. This should not happen.")
#                 # For now, we will drop them to proceed, but this points to a deeper issue.
#                 valid_indices = y.notnull()
#                 X = X.loc[valid_indices]
#                 y = y.loc[valid_indices]
#                 print("Dropped null TARGET rows for this training step.")

#             # Check for nulls in the features `X`
#             if X.isnull().sum().sum() > 0:
#                 print("FATAL: Found nulls in the feature set (X) that were not cleaned by the previous agent.")
#                 # Find and print the exact columns that are still dirty
#                 dirty_columns = X.isnull().sum()
#                 dirty_columns = dirty_columns[dirty_columns > 0]
#                 print("Dirty columns and their null counts:")
#                 print(dirty_columns)
#                 # We must raise an error here because the binning will fail.
#                 raise ValueError("Null values detected in feature set. Halting execution.")
            
#             print("Data integrity check passed. No nulls detected in X or y.")
#             # --- END DEBUGGING STEP ---


#             # --- Instantiate and fit the OptimalBinning object ---
#             optb = OptimalBinning(
#                 name=segment,
#                 dtype="numerical", 
#                 solver="cp",
#                 cat_vars=categorical_features,
#                 monotonic_trend="auto_asc_desc",
#                 min_prebin_size=0.05
#             )
            
#             print("Fitting OptimalBinning...")
#             optb.fit(X, y)
#             print("Fitting complete.")

#             # --- (Rest of the method is the same) ---
#             iv_report = optb.summary()
#             selected_vars = iv_report[iv_report['iv'] > self.iv_threshold]['name'].tolist()
#             print(f"Segment '{segment}': Found {len(iv_report)} variables, selected {len(selected_vars)} based on IV > {self.iv_threshold}")
#             iv_report['segment'] = segment
#             all_iv_reports.append(iv_report)

#             woe_transformed_df = optb.transform(X[selected_vars], metric="woe")
#             woe_transformed_df.columns = [f"{col}_woe" for col in woe_transformed_df.columns]
            
#             final_segment_df = pd.concat([
#                 segment_df.loc[y.index, ['SK_ID_CURR', 'TARGET', 'segment']],
#                 woe_transformed_df
#             ], axis=1)
            
#             all_binned_data.append(final_segment_df)
#             all_binning_objects[segment] = optb

#         # --- (Rest of the method is the same) ---
#         final_df = pd.concat(all_binned_data, ignore_index=True)
#         final_iv_report = pd.concat(all_iv_reports, ignore_index=True)

#         return {
#             "woe_data": final_df,
#             "iv_report": final_iv_report,
#             "binning_objects": all_binning_objects
#         }

#     def save_outputs(self, artifacts):
#         """Saves the WOE data, IV report, and the trained binning objects."""
#         # Save dataset ready for modeling
#         data_path = os.path.join(self.output_dir, "woe_transformed_data.parquet")
#         artifacts["woe_data"].to_parquet(data_path)
#         print(f"Saved WOE transformed data to {data_path}")

#         # Save the combined IV report
#         report_path = os.path.join(self.output_dir, "iv_summary.csv")
#         artifacts["iv_report"].to_csv(report_path, index=False)
#         print(f"Saved IV summary report to {report_path}")

#         # Save the critical binning objects using pickle
#         # This is a MODEL artifact, not just data.
#         objects_path = os.path.join(self.output_dir, "binning_objects.pkl")
#         with open(objects_path, 'wb') as f:
#             pickle.dump(artifacts["binning_objects"], f)
#         print(f"Saved trained binning objects to {objects_path}")

# --- NEW: The Model Development Agent ---
class ModelDevelopmentAgent(BaseAgent):
    """
    Agent responsible for training a model. It performs one-hot encoding,
    splits data, trains a logistic regression, and evaluates it.
    This is a simplified version that bypasses WOE/IV.
    """
    def __init__(self, input_dir: str, output_dir: str, test_size: float = 0.2):
        super().__init__("ModelDevelopmentAgent", input_dir, output_dir)
        self.test_size = test_size

    def load_inputs(self):
        """Loads the transformed data from the variable creation agent."""
        print("Loading transformed data...")
        input_file = os.path.join(self.input_dir, "transformed_data.parquet")
        df = pd.read_parquet(input_file).copy()
        return {"data": df}

    def run(self, inputs):
        """Performs one-hot encoding, trains, and evaluates the model."""
        print("Running Model Development...")
        df = inputs["data"]

        # --- 1. One-Hot Encoding for Categorical Variables ---
        print("Performing one-hot encoding...")
        # Get a list of categorical columns (type 'object')
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('segment') # Keep segment as is
        
        # Use pandas get_dummies for simple, robust one-hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)
        print(f"Data shape after encoding: {df_encoded.shape}")

        # --- 2. Data Splitting (Train/Test) ---
        print(f"Splitting data into train/test sets ({1-self.test_size:.0%}/{self.test_size:.0%})...")
        # We need to do this PER SEGMENT
        
        all_trained_models = {}
        all_performance_reports = {}

        for segment in df_encoded['segment'].unique():
            print(f"--- Processing segment: {segment} ---")
            segment_df = df_encoded[df_encoded['segment'] == segment]
            
            y = segment_df['TARGET']
            X = segment_df.drop(columns=['SK_ID_CURR', 'TARGET', 'segment'])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42, stratify=y
            )

            # --- 3. Model Training ---
            print(f"Training Logistic Regression for segment '{segment}'...")
            # Using a simple, regularized logistic regression
            model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=500)
            model.fit(X_train, y_train)
            
            # --- 4. Model Evaluation ---
            y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of being 'bad' (class 1)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            print(f"Segment '{segment}' Test Set AUC: {auc_score:.4f}")
            
            all_trained_models[segment] = model
            all_performance_reports[segment] = {"auc": auc_score}

        return {
            "trained_models": all_trained_models,
            "performance_reports": all_performance_reports
        }

    def save_outputs(self, artifacts):
        """Saves the trained model objects and performance reports."""
        # Save the trained models using pickle
        models_path = os.path.join(self.output_dir, "trained_models.pkl")
        with open(models_path, 'wb') as f:
            pickle.dump(artifacts["trained_models"], f)
        print(f"Saved trained model objects to {models_path}")

        # Save the performance report
        report_path = os.path.join(self.output_dir, "performance_reports.json")
        with open(report_path, 'w') as f:
            json.dump(artifacts["performance_reports"], f, indent=4)
        print(f"Saved performance reports to {report_path}")



# --- NEW: The Scorecard Conversion Agent ---
class ScorecardConversionAgent(BaseAgent):
    """
    Agent responsible for converting a trained logistic regression model's
    output (probability) into a scaled, points-based score.
    """
    def __init__(self, input_dir: str, output_dir: str, base_score=600, base_odds=50, pdo=20):
        super().__init__("ScorecardConversionAgent", input_dir, output_dir)
        # --- Business Parameters for Scaling ---
        self.base_score = base_score
        self.base_odds = base_odds # e.g., 50 means 50 to 1 odds
        self.pdo = pdo

    def load_inputs(self):
        """Loads the trained models from the previous agent."""
        print("Loading trained models...")
        models_path = os.path.join(self.input_dir, "trained_models.pkl")
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        return {"models": models}

    def run(self, inputs):
        """
        Calculates the scoring Factor and Offset based on business rules
        and generates a scoring logic report.
        """
        print("Running Scorecard Conversion...")
        models = inputs["models"]
        
        # We will only create a scorecard for the model that works
        if 'EXISTING_CUSTOMER' not in models:
            raise ValueError("Required 'EXISTING_CUSTOMER' model not found.")
        
        # --- Mathematical Calculation of Factor and Offset ---
        # Factor = pdo / ln(2)
        factor = self.pdo / np.log(2)
        
        # Offset = Base Score - Factor * ln(Base Odds)
        offset = self.base_score - factor * np.log(self.base_odds)
        
        print(f"Calculated Scaling Parameters:")
        print(f"  - Factor (Slope): {factor:.4f}")
        print(f"  - Offset (Intercept): {offset:.4f}")
        
        # --- Generate a Human-Readable Report ---
        report_content = f"""
# Credit Scorecard Logic Report

This document outlines the logic for converting the model's probability of default (PD) into a customer-facing score.

## 1. Scoring Parameters (Business Requirements)
- **Base Score**: {self.base_score}
- **Base Odds**: {self.base_odds}:1
- **Points to Double Odds (PDO)**: {self.pdo}

## 2. Derived Scaling Constants
Based on the parameters above, the following constants have been calculated:
- **Factor**: {factor:.4f}
- **Offset**: {offset:.4f}

## 3. Scoring Formula
The final score is calculated using the following formula, where 'p' is the model's output probability:

1. **Calculate Odds**: `Odds = p / (1 - p)`
2. **Calculate Score**: `Score = Offset - Factor * ln(Odds)`

*Note: A lower probability 'p' results in higher odds and thus a higher score.*

## 4. Implementation Note
This scoring logic applies to the 'EXISTING_CUSTOMER' segment model. The model for 'NEW_CUSTOMER' was found to have insufficient predictive power (AUC < 0.6) and should not be used for scoring.
"""
        
        # --- Machine-Readable Artifact ---
        scoring_logic = {
            "segment": "EXISTING_CUSTOMER",
            "parameters": {
                "base_score": self.base_score,
                "base_odds": self.base_odds,
                "pdo": self.pdo,
            },
            "derived_constants": {
                "factor": factor,
                "offset": offset
            },
            "formula": "Score = Offset - Factor * ln(p / (1 - p))"
        }

        return {
            "scoring_logic_report": report_content,
            "scoring_logic_json": scoring_logic
        }

    def save_outputs(self, artifacts):
        """Saves the scoring logic report and JSON."""
        # Save the human-readable report
        report_path = os.path.join(self.output_dir, "scoring_logic_report.md")
        with open(report_path, 'w') as f:
            f.write(artifacts["scoring_logic_report"])
        print(f"Saved scoring logic report to {report_path}")

        # Save the machine-readable JSON
        json_path = os.path.join(self.output_dir, "scoring_logic.json")
        with open(json_path, 'w') as f:
            json.dump(artifacts["scoring_logic_json"], f, indent=4)
        print(f"Saved machine-readable scoring logic to {json_path}")


# --- NEW: The Documentation & Reporting Agent ---
class DocumentationAgent(BaseAgent):
    """
    Agent responsible for collecting all artifacts from the pipeline and
    synthesizing them into a final model documentation report.
    """
    def __init__(self, input_dirs: dict, output_dir: str):
        # This agent takes a dictionary of input paths
        super().__init__("DocumentationAgent", input_dirs, output_dir)

    def load_inputs(self):
        """Loads all the key artifacts from previous agent outputs."""
        print("Loading all pipeline artifacts for documentation...")
        artifacts = {}
        
        # Load EDA metrics
        with open(os.path.join(self.input_dir['eda'], 'eda_metrics.json')) as f:
            artifacts['eda_metrics'] = json.load(f)
        
        # Load Performance Definition report
        with open(os.path.join(self.input_dir['perf_def'], 'performance_definition_report.md')) as f:
            artifacts['perf_def_report'] = f.read()

        # Load Segmentation analysis
        with open(os.path.join(self.input_dir['segmentation'], 'segment_analysis_report.json')) as f:
            artifacts['segment_analysis'] = json.load(f)

        # Load Model Performance report
        with open(os.path.join(self.input_dir['model_dev'], 'performance_reports.json')) as f:
            artifacts['model_performance'] = json.load(f)

        # Load Scorecard Logic report
        with open(os.path.join(self.input_dir['scorecard'], 'scoring_logic_report.md')) as f:
            artifacts['scorecard_report'] = f.read()

        return artifacts

    def run(self, inputs):
        """Assembles the final documentation from the loaded artifacts."""
        print("Synthesizing final model documentation...")

        # --- Use f-strings to build the master Markdown document ---
        
        # Extract key performance metrics for the summary
        existing_cust_auc = inputs['model_performance'].get('EXISTING_CUSTOMER', {}).get('auc', 'N/A')
        new_cust_auc = inputs['model_performance'].get('NEW_CUSTOMER', {}).get('auc', 'N/A')

        # Format segmentation results into a readable string
        seg_report_str = ""
        for seg in inputs['segment_analysis']:
            seg_report_str += f"- **{seg['segment']}**: Population: {seg['count']}, Bad Rate: {seg['bad_rate']:.2%}\n"

        master_report = f"""
# **Final Model Documentation: Home Credit Default Risk**

---

## **1. Executive Summary**

This document details the development and validation of a credit risk model for Home Credit. The pipeline was executed, producing segmented models for 'EXISTING_CUSTOMER' and 'NEW_CUSTOMER' populations.

- **Primary Model Segment**: `EXISTING_CUSTOMER`
- **Model Type**: Logistic Regression with L1 Regularization
- **Key Performance Metric (AUC)**: **{existing_cust_auc:.4f}**

The model for the `NEW_CUSTOMER` segment demonstrated low predictive power (AUC: {new_cust_auc:.4f}) and is **not recommended for deployment**. All subsequent logic and scoring apply only to the `EXISTING_CUSTOMER` model.

---

## **2. Performance Definition**
{inputs['perf_def_report']}

---

## **3. Customer Segmentation**

The applicant pool was segmented based on internal application history.

{seg_report_str}

**Conclusion**: Due to the significant difference in data availability and risk profile, separate models were trained for each segment.

---

## **4. Model Performance**

The final models were evaluated on a 20% hold-out test set. The key performance metric is the Area Under the ROC Curve (AUC).

- **EXISTING_CUSTOMER Model AUC**: **{existing_cust_auc:.4f}**
- **NEW_CUSTOMER Model AUC**: {new_cust_auc:.4f}

---

## **5. Scorecard & Scoring Logic**
{inputs['scorecard_report']}

---
*This document was automatically generated by the A2A DocumentationAgent.*
"""
        return {"final_report": master_report}

    def save_outputs(self, artifacts):
        """Saves the final report and creates a deployment package."""
        print("Saving final report and creating deployment package...")

        # Save the final Markdown report
        report_path = os.path.join(self.output_dir, "final_model_documentation.md")
        with open(report_path, 'w') as f:
            f.write(artifacts["final_report"])
        print(f"Final documentation saved to {report_path}")

        # --- Create a clean deployment package ---
        package_dir = os.path.join(self.output_dir, "deployment_package")
        os.makedirs(package_dir, exist_ok=True)
        
        # Copy the essential artifacts into the package
        shutil.copyfile(report_path, os.path.join(package_dir, "final_model_documentation.md"))
        shutil.copyfile(os.path.join(self.input_dir['model_dev'], 'trained_models.pkl'), os.path.join(package_dir, 'trained_models.pkl'))
        shutil.copyfile(os.path.join(self.input_dir['scorecard'], 'scoring_logic.json'), os.path.join(package_dir, 'scoring_logic.json'))
        
        print(f"Deployment package created at: {package_dir}")