from io import StringIO
from typing import Dict, List

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


chunk_db = Chunk()
