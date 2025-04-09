import os
import json
import subprocess
import sys
 
# Directory where all the PDFs are located
PDF_DIR = "/home/administrator/finetune_2"
PROCESSED_LIST = "processed.txt"
 
# Default config options
DEFAULT_CONFIG = {
    "base_model": "Meta-Llama-3.1-8B-Instruct",
    "gpu_utilization": 0.8,
    "num_threads": 4
}
 
def load_processed_files():
    if not os.path.exists(PROCESSED_LIST):
        return set()
    with open(PROCESSED_LIST, "r") as f:
        return set(line.strip() for line in f if line.strip())
 
def mark_as_processed(filename):
    with open(PROCESSED_LIST, "a") as f:
        f.write(filename + "\n")
 
def process_pdf(file_path):
    config = DEFAULT_CONFIG.copy()
    config["pdf_file"] = file_path
    config_json = json.dumps(config)
    cmd = ["python3", "pdf_processor.py", config_json]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error processing {file_path}:\n{result.stderr}")
        return False
    else:
        print(f"✅ Successfully processed {file_path}")
        return True
 
def main():
    all_pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    processed = load_processed_files()
 
    for pdf in all_pdfs:
        full_path = os.path.join(PDF_DIR, pdf)
        if full_path in processed:
            print(f"⏩ Skipping already processed file: {pdf}")
            continue
 
        success = process_pdf(full_path)
        if success:
            mark_as_processed(full_path)
        else:
            print("🛑 Stopping execution due to failure.")
            break
 
if __name__ == "__main__":
    main()
