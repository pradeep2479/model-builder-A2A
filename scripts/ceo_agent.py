import json
import subprocess
import shlex
from typing import Optional
import os # Make sure os is imported

# --- NEW: Use the correct, canonical imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- System prompt remains exactly the same ---
SYSTEM_PROMPT = """
You are "OrchestratorAI", a specialized AI agent responsible for managing a credit risk modeling pipeline using Prefect.
Your ONLY goal is to translate natural language commands from a human analyst into a structured JSON object.
You must always and only respond with a valid JSON object. Do not add any conversational text or explanations.
The pipeline is triggered by running a Python script: `python scripts/pipeline.py`.
This script runs a Prefect flow named 'a2a_credit_risk_flow'.
The flow can accept parameters to override the default behavior of the agents. The JSON object you create will be converted into a `--params` argument for the Prefect run command.
Here are the available parameters you can control:
- agent: "ingestion", parameter: "nrows" (integer, e.g., 10000)
- agent: "model_dev", parameter: "test_size" (float between 0.0 and 1.0)
--- EXAMPLES ---
Human Command: "Run the full pipeline on a small sample of 10k rows to test it."
Your JSON Response:
{
  "flow_name": "a2a_credit_risk_flow",
  "params": {
    "ingestion": {
      "nrows": 10000
    }
  }
}
Human Command: "The model needs to be more robust. Re-run the pipeline but use a 30% holdout set for testing."
Your JSON Response:
{
  "flow_name": "a2a_credit_risk_flow",
  "params": {
    "model_dev": {
      "test_size": 0.3
    }
  }
}
Human Command: "Everything looks good. Run the entire thing."
Your JSON Response:
{
  "flow_name": "a2a_credit_risk_flow",
  "params": {}
}
--- END EXAMPLES ---
"""

class CEO_Agent:
    def __init__(self, model_name="gemini-2.5-flash-lite", location="us-central1"):
        self.model = None
        try:
            # --- NEW: Use the proven initialization logic ---
            project_id = os.popen("gcloud config get-value project").read().strip()
            if not project_id:
                raise ValueError("GCP project ID not found. Please run 'gcloud config set project YOUR_PROJECT_ID'.")
            
            vertexai.init(project=project_id, location=location)
            self.model = GenerativeModel(model_name)
            print(f"CEO Agent initialized, using GCP project '{project_id}' and model: {model_name}")

        except Exception as e:
            print(f"FATAL: Could not initialize Vertex AI. Please check your GCP authentication and configuration.")
            print(f"Error details: {e}")

    def get_structured_command(self, user_command: str) -> Optional[dict]:
        if not self.model:
            return None
        
        print(f"\nUser command received: '{user_command}'")
        full_prompt = f"{SYSTEM_PROMPT}\nHuman Command: {user_command}\nYour JSON Response:"
        
        print("Sending request to Google Gemini API...")
        try:
            # Tell Gemini we expect a JSON response for better reliability
            generation_config = {"response_mime_type": "application/json"}
            response = self.model.generate_content(full_prompt, generation_config=generation_config)
            
            structured_command = json.loads(response.text)
            print("Gemini generated structured command:")
            print(json.dumps(structured_command, indent=2))
            return structured_command

        except Exception as e:
            print(f"ERROR: An error occurred during the Gemini API call. {e}")
            # Let's print the raw response if available to help debug
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"Raw response from API: {response.text}")
            return None

    def execute_plan(self, plan: dict):
        if not plan or "params" not in plan:
            print("Invalid plan. Cannot execute.")
            return
        print("\n--- EXECUTING PLAN ---")
        params_str = json.dumps(plan['params'])
        command = ["python", "scripts/pipeline.py", "--params", params_str]
        print(f"Executing command: {' '.join(shlex.quote(c) for c in command)}")
        subprocess.run(command, check=True)
        print("--- PLAN EXECUTION COMPLETE ---")


if __name__ == "__main__":
    ceo = CEO_Agent()
    if ceo.model:
        print("\n--- A2A Pipeline CEO (Gemini Edition) is ready for your command ---")
        print("Type 'exit' to quit.")
        
        while True:
            user_input = input("\nEnter your command: ")
            if user_input.lower() == 'exit':
                break
            
            structured_plan = ceo.get_structured_command(user_input)
            
            if structured_plan:
                ceo.execute_plan(structured_plan)
            else:
                print("Could not proceed with execution due to planning failure.")