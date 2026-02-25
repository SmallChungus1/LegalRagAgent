import os
from dotenv import load_dotenv
load_dotenv()
try:
    from deepseek import DeepSeekAPI
    api_client = DeepSeekAPI(os.getenv("DEEPSEEK_API_KEY"))
    val = api_client.user_balance()
    print("BALANCE RESULT:", val)
except Exception as e:
    print("ERROR:", e)
