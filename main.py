from typing import Annotated, Any, Dict
from fastapi import Body, Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pinecone import Pinecone
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

security = HTTPBearer()

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("recipes")

embeddings = OpenAIEmbeddings()

vector_store = PineconeVectorStore(
    index,
    embeddings,
)


app = FastAPI(
    title="CheftGPT. The best provider of Indian Recipes in the world.",
    description="Give ChefGPT the name of an ingredient and it will give you multiple recipes to use that ingredient on in return.",
    servers=[
        {
            "url": "https://mortgage-likelihood-staff-floors.trycloudflare.com",
        },
    ],
)


class Document(BaseModel):
    page_content: str


@app.get(
    "/recipes",
    summary="Returns a list of recipes.",
    description="Upon receiving an ingredient, this endpoint will return a list of recipes that contain that ingredient.",
    response_description="A Document object that contains the recipe and preparation instructions",
    response_model=list[Document],
    openapi_extra={
        "x-openai-isConsequential": False,
    },
)
def get_recipe(ingredient: str):
    docs = vector_store.similarity_search(ingredient)
    return docs


user_token_db = {"ABCDEF": "nico"}


@app.get(
    "/authorize",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def handle_authorize(client_id: str, redirect_uri: str, state: str):
    return f"""
    <html>
        <head>
            <title>Nicolacus Maximus Log In</title>
        </head>
        <body>
            <h1>Log Into Nicolacus Maximus</h1>
            <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Nicolacus Maximus GPT</a>
        </body>
    </html>
    """


@app.post(
    "/token",
    include_in_schema=False,
)
def handle_token(code=Form(...)):
    return {
        "access_token": user_token_db[code],
    }


user_favorites_db = {}


def get_current_user(credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]):
    return credentials.credentials


@app.post(
    "/favorites",
    openapi_extra={
        "x-openai-isConsequential": False,
    },
)
def add_favorite(document: Document, user: str = Depends(get_current_user)):
    user_favorites_db.setdefault(user, [])
    user_favorites_db[user].append(document)
    return {"ok": True}


@app.get(
    "/favorites",
    response_model=list[Document],
    openapi_extra={
        "x-openai-isConsequential": False,
    },
)
def list_favorites(user: str = Depends(get_current_user)):
    return user_favorites_db.get(user, [])