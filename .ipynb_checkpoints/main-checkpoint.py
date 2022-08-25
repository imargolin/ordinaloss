# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:02:37 2022

@author: imargolin
"""

from fastapi import FastAPI
app = FastAPI()
from typing import Union

@app.get("/")
def read_root():
    return {"asdasd":"SSS"}

@app.get("/items/{item_id}")
def read_item(item_id: str="DDD", q: Union[int, None] = None):
    return {"item_id": item_id, "q": q}