"""Debug script to see the actual Gemini API error."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
print(f"Key: {api_key[:12]}...{api_key[-4:]}")
print(f"Key length: {len(api_key)}")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('models/gemini-2.0-flash')

try:
    response = model.generate_content("Say hello in one word.")
    print(f"SUCCESS: {response.text}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception full: {e}")
    print(f"Contains '429': {'429' in str(e)}")
    print(f"Contains 'quota': {'quota' in str(e).lower()}")
    print(f"Contains 'GenerateRequestsPerDayPerProjectPerModel': {'GenerateRequestsPerDayPerProjectPerModel' in str(e)}")
