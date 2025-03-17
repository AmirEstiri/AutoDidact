from io import StringIO
from typing import List

import json
import pandas as pd

import pymongo
from pymongo import MongoClient
from tqdm import tqdm

from dotenv import load_dotenv
import os

load_dotenv()


class Chunk:

    def __init__(self):
        self.uri = os.getenv("MONGODB_URI")
        self.client = MongoClient(self.uri)
        self.database = self.client["voltai-backend"]

    def batch_insert(self, collection_name: str, chunk_list: List):
        operations = [
            pymongo.UpdateOne({"key": chunk["key"]}, {"$set": chunk}, upsert=True)
            for chunk in chunk_list
        ]
        self.database[collection_name].bulk_write(operations)  # TODO: handle errors

    def delete_many(self, collection_name: str, filters: dict):
        """
        Delete multiple documents from a collection based on the provided filters.

        Args:
            collection_name (str): The name of the collection.
            filters (dict): The filters to apply for deletion.

        Returns:
            dict: The result of the delete operation.
        """
        result = self.database[collection_name].delete_many(filters)
        return {
            "deleted_count": result.deleted_count,
            "acknowledged": result.acknowledged,
        }

    def find_by_keys(self, collection_name: str, chunk_keys: List):
        result = self.database[collection_name].find({"key": {"$in": chunk_keys}})
        chunk_list: List = []
        for chunk in result:
            chunk_list.append(chunk)

        return chunk_list

    def get_lookup_dict(self, collection_name: str, chunk_keys: List):
        if not isinstance(chunk_keys, list):
            chunk_keys = list(chunk_keys)
        chunk_list = self.find_by_keys(collection_name, chunk_keys)
        lookup_dict = {}
        for chunk in chunk_list:
            lookup_dict[chunk["key"]] = chunk["data"]

        return lookup_dict
    
    
    def prepare_data(self, collection_name: str, filenames: List[str]):
        all_data = {}
        content = ""
        ids = []
        cnt = 0
        for r in tqdm(
            self.database[collection_name].find(
                {"data.filename": {"$in": filenames}}
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
                text = f"{r['data']['title']}\n{df.to_markdown()}"
                ids.append(r["data"]["id"])
                data_id = r["data"]["id"]

            cleaned_text = (
                text.split("This is the summary of the surrounding context:")[0]
                .split("This is the original content:")[-1]
                .strip()
            )
            content += cleaned_text
            content += "\n\n"
            cnt += 1
            all_data[data_id] = cleaned_text

        os.makedirs("data", exist_ok=True)
        with open("data/text.md", "w") as f:
            f.write(content)

        with open("data/chunks.json", "w") as f:
            json.dump(all_data, f, indent=4)

        print(cnt)
                


if __name__ == "__main__":
    chunk_db = Chunk()
    chunk_db.prepare_data("INFINEON-AUTOMOTIVE_chunks", ["Infineon-TC39x-DataSheet-v01_02-EN.pdf"])
