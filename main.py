from fastapi import FastAPI, Response
import replicate
import requests
from mangum import Mangum
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tempfile
from typing import Union
from io import BytesIO
import os
from dotenv import load_dotenv
import base64 

# Specify the path to the .env file
dotenv_path = ".env"

# Load environment variables from the specified .env file
load_dotenv(dotenv_path)
# Load your API key from an environment variable or secret management service
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")


app = FastAPI()
handler = Mangum(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# return the base64 encoded image to be used in the client side 
@app.get("/")
async def generate_response_api(prompt: str):
    output = generate_response(prompt)
    if output is None:
        return Response(content="Failed to generate response", media_type="text/plain")

    try:
        response = requests.get(output)
        response.raise_for_status()

        # Convert the image to base64
        image_base64 = base64.b64encode(response.content).decode("utf-8")

        return {"image": image_base64}
    except requests.exceptions.RequestException as e:
        return Response(content=str(e), media_type="text/plain")

def generate_response(prompt):
    output = replicate.run(
        # for stable diffusion
        "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
        input={"prompt": prompt}
    )
    print(output)

    # Check if the output is not None
    if output is not None and len(output) > 0:
        return output[0]

    return None