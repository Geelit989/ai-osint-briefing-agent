### This is the main function to orchestrate the workflow <TEST> ###

import subprocess

from osint_agent.retrieval.storage import create_db
from osint_agent.ingestion.load_samples 
import osint_agent.extraction.ner

if __name__ == "__main__":
    """
    Call create_db()
    Call load_samples()
    Call ner()
    Call insert_data()
    """

print("Starting the OSINT Agent workflow...")
print("Step 1: Creating the database...")
create_db()

print("Step 2: Loading sample articles...")
subprocess.run(["python", "src/osint_agent/ingestion/load_samples.py"], check=True)