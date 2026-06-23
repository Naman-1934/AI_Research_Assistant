import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print("API Key Loaded Successfully")
    print(api_key)
else:
    print("API Key Missing")