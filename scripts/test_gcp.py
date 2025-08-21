import os
print("--- Starting GCP Connection Test ---")

try:
    # This is the official, canonical way to import the Vertex AI SDK
    import vertexai
    from vertexai.generative_models import GenerativeModel

    print("Successfully imported Vertex AI libraries.")

    # Get your project ID from the environment
    # This is a robust way to get the project ID set by 'gcloud config'
    project_id = os.popen("gcloud config get-value project").read().strip()
    
    if not project_id:
        raise ValueError("Could not determine GCP project ID. Please run 'gcloud config set project YOUR_PROJECT_ID'")

    print(f"Initializing Vertex AI for project: {project_id}...")

    # Initialize the Vertex AI SDK
    vertexai.init(project=project_id, location="us-central1")
    
    print("Vertex AI initialized successfully.")

    # Try to instantiate the model
    model = GenerativeModel("gemini-1.5-flash-001")

    print("\nSUCCESS: Successfully instantiated the Gemini model.")
    print(f"Model object: {model}")

except Exception as e:
    print(f"\nERROR: The test failed.")
    print(f"An exception occurred: {e}")

print("\n--- Test Finished ---")