"""Quick readiness check for LLM simulation."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import get_settings
s = get_settings()
print("Primary key:", s.gemini_api_key[:12] + "...")
print("Backup keys:", len(s.backup_api_keys) if s.backup_api_keys else 0)
print("Summary key:", s.summary_api_key[:12] + "..." if s.summary_api_key else "None")
print()

from llm_client import GeminiClient, MODEL_NAME
client = GeminiClient(s.gemini_api_key, s.backup_api_keys)
print("GeminiClient initialized with", len(client.api_keys), "keys")
print("Model:", MODEL_NAME)
print()

# Quick API test
print("Testing API call...")
resp = client._call_with_retry("Say hello in one word.")
print("API response:", repr(resp[:80]) if resp else "FAILED")
print()

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rule_based_training_data.csv")
if os.path.exists(csv_path):
    size = os.path.getsize(csv_path) / 1024
    import csv
    with open(csv_path) as f:
        rows = sum(1 for _ in f) - 1
    print(f"Existing training data: {rows} samples ({size:.0f} KB)")
else:
    print("No existing training data")

print("\n✅ Ready to simulate!" if resp else "\n❌ API not working")
