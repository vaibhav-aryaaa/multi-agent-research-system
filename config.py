import os
from dotenv import load_dotenv

load_dotenv()

QUALITY_THRESHOLD: float = float(os.getenv("QUALITY_THRESHOLD", "7.0"))
MAX_RETRIES: int = 3
TOKEN_LIMIT_PER_RUN: int = int(os.getenv("TOKEN_LIMIT_PER_RUN", "60000"))

