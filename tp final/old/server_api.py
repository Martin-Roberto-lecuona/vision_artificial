from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import uvicorn
import random
import string
import psutil

app = FastAPI()

# Model for text items
class TextItem(BaseModel):
    text: str

# In-memory storage
storage: Dict[str, str] = {}
current_id_num = 0

# Function to generate a unique ID
def generate_id(current_id_num: int) -> str:
    random_letters = ''.join(random.choices(string.ascii_letters, k=2))
    random_letters_end = ''.join(random.choices(string.ascii_letters, k=2))
    return f"{random_letters.lower()}{current_id_num:04}{random_letters_end.lower()}"

@app.post("/add/")
async def add_text(item: TextItem):
    global current_id_num
    current_id_num += 1
    id = generate_id(current_id_num)
    storage[id] = item.text
    return {"id": id}

@app.get("/get/{item_id}")
async def get_text(item_id: str):
    if item_id not in storage:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"text": storage[item_id]}

@app.delete("/delete/{item_id}")
async def delete_text(item_id: str):
    if item_id not in storage:
        raise HTTPException(status_code=404, detail="Item not found")
    del storage[item_id]
    return {"message": "Item deleted successfully"}

def close_port(port):
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port:
            print(f"Closing port {port} by terminating PID {conn.pid}")
            process = psutil.Process(conn.pid)
            process.terminate()

if __name__ == "__main__":
    puerto = 8000
    close_port(puerto)
    uvicorn.run(app, port=puerto)