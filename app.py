import streamlit as st
from dotenv import load_dotenv
from auxiliary_functions import extract_images_and_text, find_closest_text_block, generate_embeddings, convert_image_to_base64, analyze_with_gpt, generate_final_response
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
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener las claves de API desde las variables de entorno
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configurar el cliente de OpenAI con la clave de API
openai.api_key = openai_api_key
client = openai.Client(api_key=openai_api_key)

# Sidebar contents
with st.sidebar:
    st.title('Unestructured PDF Analyzer')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - Streamlit
    - LangChain
    - OpenAI
    - NTT Data
    ''')

def main():
    st.header("AI Assistant")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_path = pdf.name
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())

        text_blocks, image_data = extract_images_and_text(pdf_path)

        # print("Text Blocks:")
        # for block in text_blocks:
        #     print(block)

        print("\nFiltered Image Data:")
        for img in image_data:
            print(img)

        df = pd.DataFrame(image_data)
        df['text_context'] = df.apply(lambda row: f"Page: {row['page']}\nImage Name: {row['name']}\nText: {row['closest_text']}", axis=1)
        df_embeddings = df[['text_context', 'image_base64', 'image_ext']]
        df_embeddings['embedding'] = df_embeddings['text_context'].apply(generate_embeddings)

        embeddings_data = df_embeddings.to_dict(orient='records')
        with open('Data/embeddings.json', 'w') as f: #Revisar que no se rompa
            json.dump(embeddings_data, f)

        with open('Data/embeddings.json', 'r') as f:
            embeddings_data = json.load(f)

        df_embeddings = pd.DataFrame(embeddings_data)

        print('Text and Image Embeddings Done')

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            # Realizar la consulta en la base de datos vectorial
            query_embedding = generate_embeddings(query)
            similarities = cosine_similarity([query_embedding], df_embeddings['embedding'].tolist())
            closest_index = np.argmax(similarities)
            closest_match = df_embeddings.iloc[closest_index]

            context = closest_match['text_context']
            image_base64 = closest_match['image_base64']
            image_ext = closest_match['image_ext']
            image_response = analyze_with_gpt(convert_image_to_base64(image_base64, image_ext), context)

            print(f'Query: {query}')
            print(f'Closest Match: \n{closest_match}')

            final_image_response =  generate_final_response(query, context, image_response)

            # Realizar la consulta en el texto del PDF
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            store_name = pdf.name[:-4]
           
            ##### NUEVO #####

            persist_directory = f"Data/{store_name}"
            embedding_model = OpenAIEmbeddings()

            # Cargar si ya existe
            if os.path.exists(persist_directory):
                VectorStore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
                print('VectoreStore loaded')
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)
                VectorStore = Chroma.from_texts(chunks, embedding=embedding_model, persist_directory=persist_directory)
                VectorStore.persist()  # Guarda el vectorstore en disco
                print(f'New VectorStore created for {store_name} file')

            ##### FIN NUEVO #####

            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                text_response = chain.run(input_documents=docs, question=query)
                print(cb)

            # Generar la respuesta final
            final_response = generate_final_response(query, text_response, final_image_response)

            print('Final Response Generated')

            st.write(final_response)

    else:
        prompt = st.text_input("Ask a question:")

        print('No PDF loaded, accesing to Open GPT chat')

        def get_openai_response(question):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                max_tokens=100,
                temperature=0
            )
            return response.choices[0].message.content.strip()

        if prompt:
            response = get_openai_response(prompt)
            st.write(response)

if __name__ == '__main__':
    main()

