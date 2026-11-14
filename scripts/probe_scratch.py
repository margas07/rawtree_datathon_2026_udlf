from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.client import RawTreeClient
client = RawTreeClient()
print("count:", client.query("SELECT count() AS n FROM team_12.scratch"))
for r in client.query("SELECT * FROM team_12.scratch LIMIT 5"):
    print("row:", r)
