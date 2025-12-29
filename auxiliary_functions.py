from dotenv import load_dotenv
import os
import openai
import pymupdf  # PyMuPDF
import base64
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from litellm import completion
import json
from prompt import prompt_final_response

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener las claves de API desde las variables de entorno
openai_api_key = os.getenv("OPENAI_API_KEY")
another_api_key = os.getenv("ANOTHER_API_KEY")

# Configurar el cliente de OpenAI con la clave de API
openai.api_key = openai_api_key
client = openai.Client(api_key=openai_api_key)

# Guardar embeddings en archivo JSON
def save_vectorstore_to_json(chunks, embeddings, filename="vectorstore.json"):
    data = [{"text": chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)]
    with open(filename, "w") as f:
        json.dump(data, f)

# Cargar embeddings desde archivo JSON
def load_vectorstore_from_json(filename="vectorstore.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    embeddings = [item["embedding"] for item in data]
    return texts, embeddings

def extract_images_and_text(pdf_path):
    document = pymupdf.open(pdf_path)
    text_blocks = []
    image_data = []

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        blocks = page.get_text("blocks")
        text_blocks += blocks

        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            name = img[7]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            bbox = page.get_image_bbox(name)
            x0, y0, x1, y1 = bbox

            if (x1 - x0) > 50 and (y1 - y0) > 50 and y0 > 100:
              image_base64 = base64.b64encode(image_bytes).decode('utf-8')
              closest_text = find_closest_text_block(blocks, x0, y0, x1, y1)
              image_data.append({
                  "page": page_num + 1,
                  "name": name,
                  "x0": x0,
                  "y0": y0,
                  "x1": x1,
                  "y1": y1,
                  "image_ext": image_ext,
                  "image_base64": image_base64,
                  "closest_text": closest_text
                })

    return text_blocks, image_data

def find_closest_text_block(text_blocks, x0, y0, x1, y1):
    min_distance = float('inf')
    closest_text = ""
    image_center_x = (x0 + x1) / 2
    image_center_y = (y0 + y1) / 2

    for block in text_blocks:
        bx0, by0, bx1, by1, text = block[:5]
        block_center_x = (bx0 + bx1) / 2
        block_center_y = (by0 + by1) / 2

        # Calculate Euclidean distance between centers
        distance = ((image_center_x - block_center_x) ** 2 + (image_center_y - block_center_y) ** 2) ** 0.5

        # Filter blocks that are vertically aligned with the image
        if by0 < y1 and by1 > y0:
            if distance < min_distance:
                min_distance = distance
                closest_text = text

    return closest_text

def generate_embeddings(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def convert_image_to_base64(base64_image, image_ext):
    return f"data:image/{image_ext};base64,{base64_image}"

# Función para analizar con GPT usando litellm
def analyze_with_gpt(image_base64, text_context):
    response = completion(
        model="gpt-4o",
        api_key=openai_api_key,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_context},
                    {"type": "image_url", "image_url": {"url": image_base64}},
                ]
            }
        ]
    )
    return response.choices[0].message.content

def generate_final_response(query, text_response, image_response):    

    prompt = prompt_final_response.format(
    query=query,
    text_response=text_response,
    image_response=image_response)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un asistente que analiza contenido de documentos."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

def generate_final_response_image(query, closest_text, analysis):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un asistente que analiza contenido de documentos."},
            {"role": "user", "content": f"Consulta: {query}\n\nResultado más cercano: {closest_text}\n\nAnálisis de GPT: {analysis}\n\nElabora una respuesta en torno a estos elementos."}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()
