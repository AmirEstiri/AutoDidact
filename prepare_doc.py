import json
from mongodb import chunk_db
from tqdm import tqdm
import pandas as pd
from io import StringIO

all_data = {}
content = ""
ids = []
cnt = 0
for r in tqdm(
    chunk_db.database["INFINEON-AUTOMOTIVE_chunks"].find(
        {"data.filename": "Infineon-TC39x-DataSheet-v01_02-EN.pdf"}
    )
):
    if "type" not in r["data"]:
        continue

    if "id" in r["data"] and r["data"]["id"] in ids:
        continue

    if r["data"]["type"] == "text":
        text = r["data"]["llm_context"]
        data_id = r["key"]

    elif r["data"]["type"] == "table":
        df = pd.read_json(StringIO(r["data"]["table"]))
        text = f"Title of table is {r['data']['title']}\n{df.to_markdown()}"
        ids.append(r["data"]["id"])
        data_id = r["data"]["id"]

    content += (
        text.split("This is the summary of the surrounding context:")[0]
        .split("This is the original content:")[-1]
        .strip()
    )
    content += "\n\n"
    cnt += 1
    all_data[data_id] = text

with open("data/text.md", "w") as f:
    f.write(content)

with open("data/chunks.json", "w") as f:
    json.dump(all_data, f, indent=4)

print(cnt)
