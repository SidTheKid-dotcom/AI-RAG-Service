from fastapi import FastAPI, Request, HTTPException
import requests
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import astrapy
from astrapy.exceptions import CollectionNotFoundException


app = FastAPI()
origins = [
    "http://localhost",  # Allow localhost (can be specific port)
    "http://localhost:3000",  # For example, a frontend running on port 3000
    "https://example.com",  # A domain you want to allow
    "*",  # Allows all origins (not recommended for production)
]


# Add CORSMiddleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


gemini_api_key = "AIzaSyD8yjZrZx29L90x9AZkj7ofi1cnZBIdKTc"
genai.configure(api_key=gemini_api_key)
from astrapy import DataAPIClient
client = DataAPIClient("AstraCS:ZuAaoZrOQjHDBEIoQTpFGDzZ:2a45ed1b058aef015c652b68d88adb201838c52c05a3c000586e10746cce1533")
database = client.get_database("https://fec84a84-0a9a-45df-8c4a-d39da233e4d6-us-east-2.apps.astra.datastax.com")


from astrapy.constants import VectorMetric



import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
import json

def clean_text(text: str) -> str:
    # Remove extra spaces, newlines, and non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces and newlines
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters if any
    return text.strip()


def fetch_and_clean_file(url):

    try:
        GITHUB_TOKEN = "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"

        HEADERS = {"Authorization": f"token {GITHUB_TOKEN}", "User-Agent": "nilaytayade"}
        response = requests.get(url,headers=HEADERS)
        response.raise_for_status()  # Raise an error for bad status codes (4xx or 5xx)

        file_content = response.text  # Get the content of the file as text
        cleaned_content = '\n'.join([line.strip() for line in file_content.splitlines() if line.strip()])

        return cleaned_content

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def ask_llm(prompt,query):
    generation_config = {
    "temperature": 1.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
    system_instruction=f"""HELPFUL CODE ASSITANT THAT IS PROVIDED WITH A FILE FROM A REPO AND QUERY\n
    PLEASE ANSWER THE QUERY WRT TO GIVEN FILE TEXT/SOURCE CODE \n
    IN MARKDOWN FORMAT\nLONG OUTPUT AND CODE SNIPPETS\nalways provide the source link for the same file in reply\n sometime wrong file text will be provided\n
     ask user to mention proper repo name and file name to be explained...if user query involves multiple files or repos let user no current 
     version of this platform supports singular file at a time ...use emojis...dont send the ```markdown ...send markdown text only
\n""",
    )

    chat_session = model.start_chat(
    history=[
    ]
    )
    client = DataAPIClient("AstraCS:ZuAaoZrOQjHDBEIoQTpFGDzZ:2a45ed1b058aef015c652b68d88adb201838c52c05a3c000586e10746cce1533")
    database = client.get_database("https://fec84a84-0a9a-45df-8c4a-d39da233e4d6-us-east-2.apps.astra.datastax.com")
    collection = database.test

    results_ite = collection.find(
        {},
        projection={"*": 1},
        limit=10,
        sort={"$vectorize": query},)
    
    prompt = f"YOU ARE A HELPFUL CODE ASSISTANT WHICH ANSWERS QUERIES BASED ON CHUNKS OF DOCUMENTS START BY GREETING THE USER USE THIS TIME AS REFERENCE {get_ist_time()} \n GREETING SHOULD BE GOOD [MORNING,AFTERNOON,EVENING]\nHere is a query: {query}\n\n"
    prompt += """Based on the query(which is a natural language may contain repository name and file name), one probably source file question is about and the
    provided file chunks, please provide a comprehensive and informative answer. DONT MENTION THE FILE CHUNKS IN THE ANSWER USER DOESTNT KNOW HOW INTERNALLY EVERTHING WORKS AND ISNT AWARE ABOUT THE FILE CHUNKS...
    BUT ALWAYS GIVE RELEVANT GITHUB SOURCE LINKS FROM FILE NAMES
    state source using each document name + include code snippets whenever possible
    (name has github src...mention that..use file names too)
    make sure out is MARKDOWN format provide conclusions and make the reply as long as possible
    Documents(These can be from repos/same repo diffrent files...
    name of file identifies each 
    document construct a cohernt answer from this mention the source/github blogs as links
    at very top of each ans dont hallucinate, dont mix and match repos ) use emojis whenever possible:
    \nFILE CHUNKS FROM OTHER RELATED FILES:\n"""


    query = results_ite.get_sort_vector()
    for doc in results_ite:

            prompt += f"File: {doc['name']}\n"
            prompt += f"Content: {doc['$vectorize'].strip().replace(' ', '')}\n\n"
    response = chat_session.send_message(prompt)

    return response.text



def new_code(query):
        generation_config = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1000,
        "response_schema": content.Schema(
            type = content.Type.OBJECT,
            properties = {
            "file_url": content.Schema(
                type = content.Type.STRING,
            ),
            },
        ),
        "response_mime_type": "application/json",
        }

        model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
        generation_config=generation_config,
        system_instruction="for the given query and diffrent repos ( structured ) decide which repo and file user is talking about and once decided create the link to file from repo link and file path...return json called {file_url} always give a valid github raw file link choose file wisely if user doesn't provide valid query regarding repo then tell user to be specific and try again.....make the final link in following format https://raw.githubusercontent.com/Mercury-Copilot/Spam-Detect/refs/heads/main/Detect/spam.ipynb instead of https://github.com/Mercury-Copilot/Spam-Detect/blob/main/Detect/spam.ipynb\n",
        )

        chat_session = model.start_chat(
        history=[
            {
            "role": "model",
            "parts": [
                "```json\n{\n  \"file_url\": \"https://raw.githubusercontent.com/Mercury-Copilot/Spam-Detect/refs/heads/main/Detect/spam.ipynb\"\n}\n```",
            ],
            },
        ]
        )
        response = requests.get("http://13.127.245.117/api/upload/github")
        response.raise_for_status()  # Raise an error for bad status codes (4xx or 5xx)

        repos = response.json()  # Parse the JSON response into a list of repositories

        # Create an array of stringified repo details
        repo_details_array = [
            json.dumps({
                "repoUrl": repo["repoUrl"],
                "repoName": repo["repoName"],
                "owner": repo["owner"],
                "description": repo["description"]
                , "structure":repo["structure"]
            }) for repo in repos
        ]

        # print(repo_details_array) 
        
        prompt=f"""QUERY={query}\n REPOS ARE (almost all queries are related to these...so predict and return atleast one file url repo for each query)  final link should be of format like (.../refs/heads/..) e.g  https://raw.githubusercontent.com/Mercury-Copilot/Spam-Detect/refs/heads/main/Detect/spam.ipynb=>\n"""

        for repo in repo_details_array:
            prompt+=repo+"\n"
        

        response = chat_session.send_message(prompt)
        python_object = json.loads(json.dumps(response.text))
        file_url= python_object.split('"')[-2]
        print(file_url)
        cleaned_file_content = fetch_and_clean_file(file_url)
        ans=ask_llm(f"QUERY= {query}\n PROBABLE FILE\nFILE SOURCE: {file_url}\nFILE TEXT:\n {cleaned_file_content}",query)
        return ans




def index_repo(repo_url):
    """
    Indexes a GitHub repository into Elasticsearch, including the blob URL for each file.

    Args:
        repo_url: The URL of the GitHub repository.

    Returns:
        None
    """
    # Elasticsearch configuration
    # Extract repository name and create an Elasticsearch index name
    repo_name = repo_url.split("/")[-1].split(".")[0]
    index_name = f"{repo_name}_index"
    


    # Construct GitHub API URL for repository contents
    repo_api_url = f"https://api.github.com/repos/{'/'.join(repo_url.split('/')[-2:])}/contents/"

    # Authorization token for GitHub API
    github_token = "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"
    headers = {"Authorization": f"token {github_token}"}

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants and configurations
GITHUB_TOKEN = "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"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}


def delete_collection_if_exists(database, collection_name):
    """
    Safely delete a collection if it exists in the database.
    
    Args:
        database: AstraDB database instance
        collection_name: Name of the collection to delete
        
    Returns:
        bool: True if collection was deleted, False if it didn't exist
    """
    try:
        # First check if collection exists
        collection = database.get_collection(collection_name)
        if collection:
            database.drop_collection(name_or_collection=collection_name)
            print(f"Collection '{collection_name}' successfully deleted.")
            return True
    except CollectionNotFoundException:
        print(f"Collection '{collection_name}' does not exist. Nothing to delete.")
        return False
    except Exception as e:
        print(f"An error occurred while trying to delete collection '{collection_name}': {str(e)}")
        return False

ASTRA_DB_APPLICATION_TOKEN = "AstraCS:ZuAaoZrOQjHDBEIoQTpFGDzZ:2a45ed1b058aef015c652b68d88adb201838c52c05a3c000586e10746cce1533"
ASTRA_DB_API_ENDPOINT = "https://fec84a84-0a9a-45df-8c4a-d39da233e4d6-us-east-2.apps.astra.datastax.com"



# Initialize the client
my_client = astrapy.DataAPIClient()
my_database = my_client.get_database(
    ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)




path=""

def fetch_and_chunk(api_url, repo_url):
    global path
    """
    Recursively fetches files and directories from a GitHub repository,
    splits file content into chunks, and prepares it for indexing.

    Args:
        api_url (str): GitHub API URL of the directory to fetch.
        repo_url (str): The base URL of the repository.

    Returns:
        list: A list of documents, each representing a chunk with metadata.
    """
    GITHUB_TOKEN = "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"
    HEADERS = {"Authorization": f"token {GITHUB_TOKEN}", "User-Agent": "nilaytayade"}
    documents = []
    code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.html', '.css', '.ts', '.json', '.xml', '.yml', '.yaml', '.sh', '.bat','md','txt','ipynb'}

    try:
        response = requests.get(api_url, headers=HEADERS)
        print("First request done")
        response.raise_for_status()
        items = response.json()


        print("Index done")
        print("Fetching and chunking files...")

        for item in items:
            # Skip ignored paths
            if any(ignore_path in item['path'] for ignore_path in {'node_modules', 'packages', 'db', 'ml models','.json'}):
                print(f"Ignoring directory: {item['path']}")
                continue

            if item['type'] == 'file':
                # Process only code file extensions
                if not any(item['path'].endswith(ext) for ext in code_extensions):
                    print(f"Ignoring non-code file: {item['path']}")
                    continue

                # Fetch file content
                print(f"Fetching file: {item['download_url']}")
                file_content_response = requests.get(item['download_url'], headers=HEADERS)
                print("Second request done")
                file_content_response.raise_for_status()
                file_content = file_content_response.text

                # Split the file content into 500-character chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
                file_content = clean_text(file_content)
                chunks = text_splitter.split_text(file_content)

                # Add metadata and chunk number
                for chunk_no, chunk in enumerate(chunks, start=1):
                    documents.append({
                        'file_path': item['path'],
                        'repo_url': repo_url,
                        'blob_url': item['html_url'],  # Add GitHub blob URL
                        'chunk_content': chunk,
                        'chunk_no': chunk_no
                    })

                # Insert into database
                print(f"Processed file: {item['path']} with {len(chunks)} chunks.")
                print("Sending to Astra...")

            elif item['type'] == 'dir':
                # Recursively fetch and process directory contents
                print(f"Processing directory: {item['path']}")
                documents.extend(fetch_and_chunk(item['url'], repo_url))

    except requests.exceptions.RequestException as req_err:
        print(f"Request error processing URL {api_url}: {req_err}")
    except Exception as e:
        print(f"Error processing URL {api_url}: {e}")

    return documents



from datetime import datetime, timedelta
import pytz

def get_ist_time():
    # Get the current UTC time
    utc_time = datetime.utcnow()
    
    # Convert UTC time to IST
    ist_timezone = pytz.timezone("Asia/Kolkata")
    ist_time = utc_time.replace(tzinfo=pytz.utc).astimezone(ist_timezone)
    
    # Format IST time in a human-readable format
    human_readable = ist_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    return human_readable


def search_and_answer(query,repo_name,structure):
  """Searches the specified repository's Elasticsearch index for the most relevant files,
  sends them to the Gemini LLM, and returns the generated answer.

  Args:
    query: The query string.
    repo_name: The name of the GitHub repository.

  Returns:
    The generated answer or an error message.
  """
  client = DataAPIClient("AstraCS:ZuAaoZrOQjHDBEIoQTpFGDzZ:2a45ed1b058aef015c652b68d88adb201838c52c05a3c000586e10746cce1533")
  database = client.get_database("https://fec84a84-0a9a-45df-8c4a-d39da233e4d6-us-east-2.apps.astra.datastax.com")
  collection = database.get_collection(repo_name)

  prompt = f"""You are a helpful code assistant that answers queries based on chunks of documents. Start by greeting the user using the current time as a reference. Here is the current time: {get_ist_time()}.

Greeting should be:
- "Good morning" if the time is before 12:00 PM
- "Good afternoon" if the time is between 12:00 PM and 5:00 PM
- "Good evening" if the time is after 5:00 PM

**Query:**
{query}

Based on the query (which may include a repository name and file name) and the provided file chunks, please provide a comprehensive and informative answer. Your response should include:

1. **Source Attribution**: Clearly state the source of information using each document name.
2. **Code Snippets**: Include code snippets wherever applicable to enhance understanding.
3. **GitHub References**: Mention GitHub sources, including repository names and file names.
4. **GitHub Links**: Provide a direct link to the relevant GitHub file every link should be clickable in markdown.
5. **Markdown Format**: Ensure the output is in Markdown format, using headings, lists, and code blocks as needed.
6. **Conclusion**: Summarize the information in a conclusion.
7. **Emojis**: Use emojis to make the response more engaging use emojis for document names and output markdown sections appropriately.

**Documents** (These can be from different repositories or the same repository but different files. The name of the file identifies each document):
- Construct a coherent answer from these.
- Mention the sources and GitHub blogs as links.
- Ensure the answer is as detailed and long as possible.
- Do not hallucinate or mix and match repositories.

Please ensure your response is accurate, detailed, and well-structured."""



  # Connect to Elasticsearch using appropriate configuration
  results_ite = collection.find(
        {},
        projection={"*": 1},
        limit=15,
        sort={"$vectorize": query},)
  

  query = results_ite.get_sort_vector()
  for doc in results_ite:
            print(f"docname/path/src={doc['name']} \ncontent=> {doc['$vectorize']}")


            prompt += f"File: {doc['name']}\n"
            prompt += f"Content: {doc['$vectorize'].strip().replace(' ', '')}\n\n"
  with open('prompt-code.txt', 'w') as f:
                f.write(prompt)


  # Use a try-except block for error handling with informative message
  try:
      model = genai.GenerativeModel("gemini-1.5-flash-8b")
      response = model.generate_content(prompt)
      return response.text
  except Exception as e:
      print(f"Error generating response: {e}")
      return "An error occurred while processing your query. Please try again later."

database = client.get_database("https://fec84a84-0a9a-45df-8c4a-d39da233e4d6-us-east-2.apps.astra.datastax.com")


@app.post("/index_repo")
async def index_repo_endpoint(repo_url: str):
    try:
        repo_api_url = f"https://api.github.com/repos/{'/'.join(repo_url.split('/')[-2:])}/contents/"
        documents=fetch_and_chunk(repo_api_url, repo_url)
        converted_documents = []
        for doc in documents:
                    # Create 'name' by concatenating repo_url, blob_url, and chunk_no
                    name = f"src_url={doc['blob_url']} =>chunk_no={doc['chunk_no']}"
                    # Create '$vectorize' as a string of the whole document
                    vectorize = doc['chunk_content']
                    converted_documents.append({
                        "name": name,
                        "$vectorize": vectorize
                    })

        # for doc in converted_documents:
        #     print(doc)
        repo_name = repo_url.split("/")[-1].split(".")[0].replace("-","_")
        index_name = f"{repo_name}_index"
        delete_collection_if_exists(database, index_name)

        my_collection = database.create_collection(
    index_name,
    dimension=1024,
    metric=VectorMetric.DOT_PRODUCT,
    service={
        "provider": "nvidia",
        "modelName": "NV-Embed-QA"
    }
)


        collection = database.get_collection(index_name)
        for doc in converted_documents:
            collection.insert_one(doc)

        results_ite = collection.find(
        {},
        projection={"*": 1},
        limit=20,
        sort={"$vectorize": "TEST QUERY HERE"},)

        return {"message": "Repository indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_and_answer")
async def search_and_answer_endpoint(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(content={"error": "Invalid JSON format", "details": str(e)}, status_code=400)
    query = data.get("query")
    repo = data.get("repo")
    structure = repo["structure"]
    url= repo["repoUrl"]
    repo_name = url.split("/")[-1].split(".")[0].replace("-","_")
    
    return {"ans":search_and_answer(query,f"{repo_name}_index",structure) }
    

    if not query :
        raise HTTPException(status_code=400, detail="Query and repo_name are required")

    try:
        answer = search_and_answer(query, )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





from pydantic import BaseModel
import requests
from io import BytesIO
from PyPDF2 import PdfReader
import docx
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urlparse



# Model to represent the file URL in request
class FileRequest(BaseModel):
    file_url: str

# Helper function to extract text from different file formatsimport requests
import requests
from io import BytesIO
from fastapi import HTTPException
import fitz  # PyMuPDF
import docx
import mimetypes

def extract_text_from_file(url: str) -> str:
    response = requests.get(url)
    
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="File not found at the provided URL")
    
    file_content = BytesIO(response.content)
    file_content.seek(0)
    
    # Determine the file type using mimetypes
    mime_type, _ = mimetypes.guess_type(url)
    file_content.seek(0)
    
    if mime_type is None:
        raise HTTPException(status_code=400, detail="Unable to determine file type")
    
    if "pdf" in mime_type:
        # For PDF files
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
            return text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading PDF file: {str(e)}")
    
    elif "msword" in mime_type or "vnd.openxmlformats-officedocument.wordprocessingml.document" in mime_type:
        # For DOCX files
        try:
            doc = docx.Document(file_content)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading DOCX file: {str(e)}")
    
    elif "plain" in mime_type or "text" in mime_type:
        # For TXT files
        try:
            return file_content.read().decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading text file: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")




@app.post("/process-file/")
async def process_file(request: FileRequest):
    file_url = request.file_url
    try:
        # Extract and clean the text from the file
        text = extract_text_from_file(file_url)
        cleaned_text = clean_text(text)
        
        # Split the text into chunks using RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        chunks = splitter.split_text(cleaned_text)
        
        # Prepare the response list
        result = []
        for idx, chunk in enumerate(chunks):
            result.append({
                "fileurl": file_url,
                "chunk_content": chunk,
                "chunk_number": idx + 1
            })

        formatted_chunks=[]
        for chunk in result:
            formatted_chunks.append({
                "name": f"filename={chunk['fileurl'].split('/')[-1]} src_url={chunk['fileurl']} =>chunk_no={chunk['chunk_number']}",
                "$vectorize": chunk['chunk_content']
            })

        client = DataAPIClient("AstraCS:ZuAaoZrOQjHDBEIoQTpFGDzZ:2a45ed1b058aef015c652b68d88adb201838c52c05a3c000586e10746cce1533")
        database = client.get_database("https://fec84a84-0a9a-45df-8c4a-d39da233e4d6-us-east-2.apps.astra.datastax.com")

        
        collection = database.docs
        
        for chunk in formatted_chunks:
            collection.insert_one(chunk)
        
        return {"chunks": formatted_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-document")
async def query_document(query: str):
    try:
       
        # data = await request.json()
        query = query
        prompt = f"""You are a helpful document assistant that answers queries based on chunks of documents. Start by greeting the user using the current time as a reference. Here is the current time: {get_ist_time()}. 

Greeting should be:
- "Good morning" if the time is before 12:00 PM
- "Good afternoon" if the time is between 12:00 PM and 5:00 PM
- "Good evening" if the time is after 5:00 PM

**Question:**
{query}

Based on the query (which may mention the document name), provide a comprehensive and informative answer using the relevant document chunks. Your response should include:

1. **Source Attribution**: Always mention the source of documents as links.
2. **Markdown Format**: Ensure the output is in Markdown format.
3. **Accuracy**: Do not hallucinate; provide a quick rhetorical answer if asked, and then go in-depth with sources.
4. **Repetition**: Mention the source once for repeating documents.
5. **Emojis**: Use emojis to enhance the response when appropriate.

**Documents** (These can be from different files):
- Construct a coherent answer from these.
- Mention the sources and GitHub blogs as links.
- Ensure the answer is detailed, accurate, and well-structured.
- Do not mix and match different repositories.

"""

        # 1. Query Astra DB for the document content based on the user's query
        client = DataAPIClient("AstraCS:ZuAaoZrOQjHDBEIoQTpFGDzZ:2a45ed1b058aef015c652b68d88adb201838c52c05a3c000586e10746cce1533")
        database = client.get_database("https://fec84a84-0a9a-45df-8c4a-d39da233e4d6-us-east-2.apps.astra.datastax.com")
        collection = database.docs
        
        results_ite = collection.find(
        {},
        projection={"*": 1},
        limit=10,
        sort={"$vectorize": query},)
        
        query = results_ite.get_sort_vector()
        for doc in results_ite:
            print(f"docname/path/src={doc['name']} \ncontent=> {doc['$vectorize']}")


            prompt += f"File: {doc['name']}\n"
            prompt += f"Content: {doc['$vectorize'].strip().replace(' ', '')}\n\n"
        
            #write prompt to file
        with open('prompt-doc.txt', 'w') as f:
                f.write(prompt)

        try:
            model = genai.GenerativeModel("gemini-1.5-flash-8b")
            response = model.generate_content(prompt)
            return {"answer": response.text}
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while processing your query. Please try again later."


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    





# @app.post("/code_answer")
# async def echo(query: str):



import pandas as pd
import requests
import psycopg2
from io import StringIO
from sqlalchemy import create_engine, text
import numpy as np

def fetch_csv_from_url(url):
    """
    Fetch CSV data from a URL and return it as a pandas DataFrame with proper NA handling
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), na_values=['', 'NA', 'N/A'], keep_default_na=True)
    except Exception as e:
        raise Exception(f"Error fetching CSV: {str(e)}")

def infer_sql_schema(df):
    """
    Analyze DataFrame and return SQL column definitions with better type handling
    """
    type_mapping = {
        'object': 'TEXT',
        'int64': 'BIGINT',
        'float64': 'DOUBLE PRECISION',
        'datetime64[ns]': 'TIMESTAMP',
        'bool': 'BOOLEAN',
        'category': 'TEXT'
    }
    
    column_definitions = []
    
    for column in df.columns:
        clean_column = ''.join(e for e in column if e.isalnum() or e == '_').lower()
        dtype = str(df[column].dtype)
        sql_type = type_mapping.get(dtype, 'TEXT')
        column_definitions.append(f"{clean_column} {sql_type}")
    
    return column_definitions

structure= ''''''
def create_table_sql(table_name, column_definitions):
    """
    Generate SQL CREATE TABLE statement
    """
    columns_sql = ",\n    ".join(column_definitions)
    
    global structure
    structure= f'''
    TABLE {table_name} (
    {columns_sql}
    )
    '''

    return f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        {columns_sql}
    )
    """

def setup_postgres_connection(db_params):
    """
    Create PostgreSQL connection using SQLAlchemy with error handling
    """
    try:
        connection_string = (
            f"postgresql://{db_params['user']}:{db_params['password']}@"
            f"{db_params['host']}:{db_params['port']}/{db_params['database']}"
        )
        return create_engine(connection_string)
    except Exception as e:
        raise Exception(f"Database connection error: {str(e)}")

def migrate_data(df, table_name, engine):
    """
    Migrate data from DataFrame to PostgreSQL table with better error handling
    """
    df.columns = [''.join(e for e in col if e.isalnum() or e == '_').lower() 
                  for col in df.columns]
    
    df = df.replace([np.inf, -np.inf], None)
    df = df.where(pd.notnull(df), None)
    
    try:
        df.to_sql(
            table_name,
            engine,
            if_exists='append',
            index=False,
            chunksize=500,
            method='multi'
        )
        return True
    except Exception as e:
        raise Exception(f"Error migrating data: {str(e)}")

def start_etl(csv_url, table_name, db_params):
    """
    Main ETL function with better error handling and debugging
    """
    try:
        # Fetch CSV data
        print("Fetching CSV data...")
        df = fetch_csv_from_url(csv_url)
        print(f"Retrieved {len(df)} rows and {len(df.columns)} columns")
        
        print("\nData Types:")
        print(df.dtypes)
        
        # Infer schema
        print("\nAnalyzing data schema...")
        column_definitions = infer_sql_schema(df)
        print("Column definitions:", column_definitions)
        
        # Create table
        print("\nCreating PostgreSQL table...")
        engine = setup_postgres_connection(db_params)
        create_table_query = create_table_sql(table_name, column_definitions)
        print("Table creation SQL:", create_table_query)
        
        # Fixed: Using text() to make the SQL executable and removing extra whitespace
        with engine.connect() as connection:
            connection.execute(text(create_table_query.strip()))
            connection.commit()
        
        # Migrate data
        print("\nMigrating data...")
        migrate_data(df, table_name, engine)
        
        print("ETL process completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in ETL process: {str(e)}")
        print("Error details:", str(e.__class__.__name__))
        return False

# Database connection parameters
db_params = {
    'host': 'iitb.clk2846ssidu.ap-south-1.rds.amazonaws.com',
    'port': '5432',
    'database': 'test_db',  
    'user': 'postgres',
    'password': 'saymyname'
}

import random
import string
import uuid

@app.post("/csv-process")
async def csv_process(url: str):

    csv_url = url
    table_name = ''.join(random.choices(string.ascii_lowercase, k=5)) + '_' + str(uuid.uuid4())[:8]
    print("Table name:", table_name)

    start_etl(csv_url, table_name, db_params)
    RESULT= {"table_name": table_name , "structure": structure, "csv_url": csv_url,"file_name": csv_url.split("/")[-1]} 

    #uncomment the migrate script
    print("\nCreating PostgreSQL table...")
    engine = setup_postgres_connection(db_params)
    new_data = f"INSERT INTO all_tables VALUES ('{table_name}', '{structure}', '{csv_url}', '{csv_url.split('/')[-1]}')"
    print("Table creation SQL:",new_data )
    
    # Fixed: Using text() to make the SQL executable and removing extra whitespace
    with engine.connect() as connection:
        connection.execute(text(new_data.strip()))
        connection.commit()

    return RESULT

import psycopg2
from typing import Dict, List, Optional, Union

def query_postgres(
    table_name: str, 
    custom_query: Optional[str] = None, 
    db_params: Optional[Dict] = None
) -> Dict[str, Union[bool, str, List[Dict]]]:
    """
    Query PostgreSQL database and return results as a structured dictionary.
    
    Args:
        table_name (str): Name of the table to query
        custom_query (str, optional): Custom SQL query to execute. If None, selects all from table_name
        db_params (dict, optional): Database connection parameters
        
    Returns:
        dict: A dictionary containing:
            - success (bool): Whether the query was successful
            - message (str): Status message or error description
            - columns (list): List of column names
            - data (list): List of dictionaries, each representing a row
            - row_count (int): Number of rows returned
    """
    # Default database parameters if none provided
    if db_params is None:
        db_params = {
            'host': 'iitb.clk2846ssidu.ap-south-1.rds.amazonaws.com',
            'port': '5432',
            'database': 'test_db',
            'user': 'postgres',
            'password': 'saymyname'
        }
    
    # Initialize return structure
    result = {
        'success': False,
        'message': '',
        'columns': [],
        'data': [],
        'row_count': 0
    }
    
    try:
        # Establish connection
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Execute query
        if custom_query:
            query = custom_query
        else:
            query = f"SELECT * FROM {table_name} LIMIT 1000"  # Limiting to 1000 rows by default
            
        cursor.execute(query)
        
        # Get column names
        result['columns'] = [desc[0] for desc in cursor.description]
        
        # Fetch results and convert to list of dictionaries
        rows = cursor.fetchall()
        result['data'] = [
            dict(zip(result['columns'], row))
            for row in rows
        ]
        
        result['row_count'] = len(result['data'])
        result['success'] = True
        result['message'] = 'Query executed successfully'
        
    except psycopg2.Error as e:
        result['message'] = f"Database error occurred: {str(e)}"
        
    except Exception as e:
        result['message'] = f"An error occurred: {str(e)}"
        
    finally:
        if 'conn' in locals():
            cursor.close()
            conn.close()
            
    return result

import psycopg2

def run_query_and_format_results(query, table_name, db_params):
    """
    Runs a given SQL query on a specified table and returns the complete result set
    as a plain string, formatted for readability.

    Args:
        query (str): The SQL query to execute.
        table_name (str): The name of the table to query.
        db_params (dict): A dictionary containing database connection parameters.

    Returns:
        str: The formatted result set as a plain string.

    Raises:
        psycopg2.OperationalError: If an error occurs during database connection.
        psycopg2.Error: If an error occurs during query execution.
    """

    try:
        # Establish a secure database connection with parameter style for protection against SQL injection
        connection = psycopg2.connect(**db_params, cursor_factory=psycopg2.extras.RealDictCursor)
        cursor = connection.cursor()

        # Execute the query with parameter style
        cursor.execute(query, {'table_name': table_name})

        # Fetch all results and format them with column headers
        results = cursor.fetchall()
        if results:
            column_names = [col.name for col in cursor.description]
            formatted_results = "\n".join([", ".join([str(row[col]) for col in column_names]) for row in results])
            return f" '{table_name}':\n{formatted_results}"
        else:
            return f"No rows found for table '{table_name}'."

    except (psycopg2.OperationalError, psycopg2.Error) as error:
        return f"Error connecting to database or executing query: {error}"

    finally:
        # Always close the database connection even on errors
        if connection:
            connection.close()

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Optional, Union
import logging

def get_single_row(query: str, params: tuple = None, db_params: Dict = None) -> Union[Dict, None]:
    """
    Execute a query and return a single row as a dictionary.
    
    Args:
        query (str): SQL query to execute
        params (tuple, optional): Query parameters to prevent SQL injection
        db_params (dict, optional): Database connection parameters
        
    Returns:
        dict: Row data as dictionary with column names as keys, or None if no row found
    """
    # Default database parameters
    if db_params is None:
        db_params = {
            'host': 'iitb.clk2846ssidu.ap-south-1.rds.amazonaws.com',
            'port': '5432',
            'database': 'test_db',
            'user': 'postgres',
            'password': 'saymyname'
        }
    
    conn = None
    cursor = None
    
    try:
        # Establish connection
        conn = psycopg2.connect(**db_params)
        
        # Create cursor with dictionary cursor factory
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Execute query with parameters if provided
        cursor.execute(query, params if params else ())
        
        # Fetch single row
        row = cursor.fetchone()
        
        return dict(row) if row else None
        
    except psycopg2.Error as e:
        logging.error(f"Database error occurred: {str(e)}")
        return None
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return None
        
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

def get_query (query,structure ):

    

    import os
    import google.generativeai as genai
    from google.ai.generativelanguage_v1beta.types import content

    genai.configure(api_key="AIzaSyD8yjZrZx29L90x9AZkj7ofi1cnZBIdKTc")

    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
    type = content.Type.OBJECT,
    properties = {
        "sql_query": content.Schema(
        type = content.Type.STRING,
        ),
    },
    ),
    "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
    system_instruction="natural language to sql query assistant for the given schema of table ...user may or may not call tables as csv ...ignore and proceed ....output should always be a query ...no new line ...single line query that can be independantly executed (read only query)...",
    )

    chat_session = model.start_chat(
    history=[
    {
        "role": "user",
        "parts": [
        "what is the average value for the data ....schema=     TABLE lrjps_ce770135 (\n    description TEXT,\n    industry TEXT,\n    level BIGINT,\n    size TEXT,\n    line_code TEXT,\n    value BIGINT\n    )",
        ],
    },
    {
        "role": "model",
        "parts": [
        "```json\n{\"sql_query\": \"SELECT avg(value) FROM lrjps_ce770135\"}\n```",
        ],
    },
    ]
    )

    response = chat_session.send_message(f"""{query} \nschema={structure}""")

    return response.text




@app.post("/csv-query")
async def csv_query(query: str, table_name: str):
    """
    Generate a SQL query from a natural language query and return it to the user

    Args:
    query (str): The natural language query
    table_name (str): The name of the table to query

    Returns:
    Nothing, the query is returned directly to the user
    """
    try:
        select_all_tables = f"SELECT * FROM all_tables WHERE table_name=%s" 
        table = get_single_row(select_all_tables, params=(table_name,))

        if not table:
            raise HTTPException(status_code=404, detail="Table not found")  

        query_from_llm= get_query(query, table["structure"]).split('"')[-2]
        print(query_from_llm)

        table=query_postgres(table_name, custom_query=query_from_llm)
        return table


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def get_query_visualize (query,structure ):
    """
    Install an additional SDK for JSON schema support Google AI Python SDK

    $ pip install google.ai.generativelanguage
    """

    genai.configure(api_key="AIzaSyD8yjZrZx29L90x9AZkj7ofi1cnZBIdKTc")

    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
    type = content.Type.OBJECT,
    properties = {
    "sql_query": content.Schema(
    type = content.Type.STRING,
    ),
    },
    ),
    "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""SQL QUERY ASSITANT FOR GRAPHS""",
    )

    chat_session = model.start_chat(
    history=[
    ]
    )

    response = chat_session.send_message(f"""sql assistant=> From the given natural language query and SQL schema
                                         construct a sql query that gets all required data to construct a graph...
                                         final query should return data that can be interpreted by llm and a graph can be created , limit the number of records to 5 or the specified number max should be 15
                                         common error type: Database error occurred: column \"lrjps_ce770135.size\" must appear in the GROUP BY clause or be used in an aggregate function\nLINE 1: SELECT industry, size FROM lrjps_ce770135 GROUP BY industry....keep this in mind
                                         \nnatural language query=>{query} \nschema=>{structure}""")

    print(response.text)

    return response.text 


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
        
    return dict(items)


import json

def convert_string_to_dict(string):
    """Converts a string representation of a dictionary to an actual dictionary.

    Args:
        string: The string representation of the dictionary.

    Returns:
        The converted dictionary.
    """

    # Remove unnecessary characters and split the string into key-value pairs
    cleaned_string = string.replace('\n', '').replace(' ', '').strip('{}')
    key_value_pairs = cleaned_string.split(',')

    # Create a dictionary from the key-value pairs
    result_dict = {}
    for pair in key_value_pairs:
        key, value = pair.split(':')
        result_dict[key] = json.loads(value)

    return result_dict


def get_graph_data (query,data):
    genai.configure(api_key="AIzaSyD8yjZrZx29L90x9AZkj7ofi1cnZBIdKTc")

    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type = content.Type.OBJECT,
        properties = {
        "labels": content.Schema(
            type = content.Type.STRING,
        ),
        "data": content.Schema(
            type = content.Type.STRING,
        ),
        },
    ),
    "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="from given query and data give two json outputs to draw a chart labels and data .... intelligently choose the labels and data dont rely on user provided data it can have extra unrelated fields too dont",
    )

    chat_session = model.start_chat(
    history=[
        {
        "role": "user",
        "parts": [
            "query => top 6 industries data=>{\n  \"success\": true,\n  \"message\": \"Query executed successfully\",\n  \"columns\": [\n    \"industry\",\n    \"count\"\n  ],\n  \"data\": [\n    {\n      \"industry\": \"Agriculture\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Education & training\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Manufacturing\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Construction\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Textile, clothing, footwear, & leather\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Metal product\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Rental, hiring, & real estate services\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Electricity, gas, water, & waste services\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Other professional scientific\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Financial & insurance services\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Computer systems design\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Arts & recreation services\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Insurance\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Food, beverage, & tobacco\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Other services\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Publishing\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Other wholesale trade\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Non-metallic mineral product\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Forestry & logging\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Information media & telecommunications\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Wood & paper product\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Printing, publishing, & recorded media\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Telecommunications\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Motion picture\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Agriculture, forestry, & fishing\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Wholesale trade\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Other machinery & equipment\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Mining\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Health care & social assistance\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Accommodation & food services\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Petroleum, coal, chemical, & associated product\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Machinery & equipment wholesaling\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Commercial fishing\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Other manufacturing\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Administrative & support services\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Finance\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Transport and industrial machinery & equipment\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Agriculture, forestry, & fishing support services\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Retail trade\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Transport, postal, & warehousing\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"total\",\n      \"count\": 705\n    },\n    {\n      \"industry\": \"Professional, scientific, & technical services\",\n      \"count\": 141\n    },\n    {\n      \"industry\": \"Auxiliary\",\n      \"count\": 141\n    }\n  ],\n  \"row_count\": 43\n}",
        ],
        },
        {
        "role": "model",
        "parts": [
            "```json\n{\"data\": \"[\\n  141,\\n  141,\\n  141,\\n  141,\\n  141,\\n  141\\n]\", \"labels\": \"[\\n  \\\"Agriculture\\\",\\n  \\\"Education & training\\\",\\n  \\\"Manufacturing\\\",\\n  \\\"Construction\\\",\\n  \\\"Textile, clothing, footwear, & leather\\\",\\n  \\\"Metal product\\\"\\n]\"}\n```",
        ],
        },
    ]
    )

    response = chat_session.send_message(f"query=>{query}\n data =>{json.dumps(data)}")

    result =  json.loads(response.text)
    data=json.loads(result["data"])
    labels=json.loads(result["labels"])
    final={"data":data,"labels":labels}
    print(final)
    return final

    
  

@app.post("/csv-visualize")
async def csv_visualize(query: str, table_name: str):
    try:
        select_all_tables = f"SELECT * FROM all_tables WHERE table_name=%s" 
        table = get_single_row(select_all_tables, params=(table_name,))

        if not table:
            raise HTTPException(status_code=404, detail="Table not found")  

        query_from_llm= get_query_visualize(query, table["structure"]).split('"')[-2]
        print(query_from_llm)

        table=run_query_and_format_results( query_from_llm,table_name,db_params)
        
        final =get_graph_data(query,table)
        return final

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



from fastapi import FastAPI, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter

import requests

from typing import List
import re
import os



def clean_text(text: str) -> str:
    """Clean scraped text by removing extra whitespace and special characters."""
    # Remove HTML tags if any remain
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()


api_key="fc-c66cf0b2efb347769b69874dc155c2b5"

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader

from collections import namedtuple

def document_to_string(documents):
  """Converts a list containing a single Document object to a string.

  Args:
      documents: A list containing a single Document object.

  Returns:
      A string representation of the Document object.

  Raises:
      ValueError: If the input list does not contain exactly one Document object.
  """
  if len(documents) != 1:
    raise ValueError("Input list must contain exactly one Document object.")

  document = documents[0]
  Document = namedtuple('Document', ['metadata', 'page_content'])

  # Extract relevant information from metadata
  title = document.metadata.get('title')
  
  # Extract and clean the page content (removes unnecessary formatting)
  content = document.page_content.strip()  # Remove leading/trailing whitespace
  content = content.replace("\\", "")  # Remove escaped backslashes

  # Create the string representation
  string_representation = f"Title: {title}\n\nContent:\n{content}"
  return string_representation

# Example usage

def scrape_webpage(url: str) -> str:
    """Scrape content from a webpage."""
    try:
        print("Begin crawling the website...")
        loader = FireCrawlLoader(
        api_key=api_key, url=url, mode="scrape")
        docs = loader.load()
        print("Finished crawling the website.")
        
    # Convert metadata values to strings if they are lists
        return document_to_string(docs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to scrape URL: {str(e)}")

def split_text(text: str) -> List[str]:
    """Split text into chunks of approximately 500 tokens."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    return splitter.split_text(text)

@app.post("/scrape")
async def scrape_and_store(url: str):
    """
    Endpoint to scrape a webpage, split it into chunks, and store in AstraDB.
    """
    try:
        # Scrape the webpage
        text = scrape_webpage(url)
        
        # Split into chunks
        chunks = split_text(text)


        result = []
        for idx, chunk in enumerate(chunks):
            result.append({
                "fileurl": url,
                "chunk_content": chunk,
                "chunk_number": idx + 1
            })

        formatted_chunks=[]
        for chunk in result:
            formatted_chunks.append({
                "name": f"site_url={url} =>chunk_no={chunk['chunk_number']}",
                "$vectorize": chunk['chunk_content']
            })

        client = DataAPIClient("AstraCS:ZuAaoZrOQjHDBEIoQTpFGDzZ:2a45ed1b058aef015c652b68d88adb201838c52c05a3c000586e10746cce1533")
        database = client.get_database("https://fec84a84-0a9a-45df-8c4a-d39da233e4d6-us-east-2.apps.astra.datastax.com")

        
        collection = database.wiki
        
        for chunk in formatted_chunks:
            print(chunk)
            collection.insert_one(chunk)


        
        return {
            "status": "success",
            "url": url,
            "num_chunks": len(chunks),
            "message": f"Successfully processed and stored {len(chunks)} chunks"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/wiki")
async def query_document(query: str):
    try:
        prompt = f"""You are a helpful document assistant that answers queries based on chunks of documents. Start by greeting the user using the current time as a reference. Here is the current time: {get_ist_time()}.

Greeting should be:

"Good morning" if the time is before 12:00 PM
"Good afternoon" if the time is between 12:00 PM and 5:00 PM
"Good evening" if the time is after 5:00 PM
Question: {query}

Based on the query (which may mention the document name), provide a comprehensive and informative answer using the relevant document chunks. Your response should include:
Mention site_url as reference
Source Attribution: Always mention the source of documents as links.
Markdown Format: Ensure the output is in Markdown format.
Accuracy: Do not hallucinate; provide a quick rhetorical answer if asked, and then go in-depth with sources.
Repetition: Mention the source once for repeating documents.
Emojis: Use emojis to enhance the response when appropriate.
Documents (These can be from different files):

Construct a coherent answer from these.
Mention the sources and GitHub blogs as links.
Ensure the answer is detailed, accurate, and well-structured.
Do not mix and match different repositories.

"""

        # 1. Query Astra DB for the document content based on the user's query
        client = DataAPIClient("AstraCS:ZuAaoZrOQjHDBEIoQTpFGDzZ:2a45ed1b058aef015c652b68d88adb201838c52c05a3c000586e10746cce1533")
        database = client.get_database("https://fec84a84-0a9a-45df-8c4a-d39da233e4d6-us-east-2.apps.astra.datastax.com")
        collection = database.wiki
        
        results_ite = collection.find(
        {},
        projection={"*": 1},
        limit=20,
        sort={"$vectorize": query},)
        
        query = results_ite.get_sort_vector()
        for doc in results_ite:
            print(f"main_url={doc['name']} \ncontent=> {doc['$vectorize']}")


            prompt += f"File: {doc['name']}\n"
            prompt += f"Content: {doc['$vectorize'].strip().replace(' ', '')}\n\n"
    

        try:
            model = genai.GenerativeModel("gemini-1.5-flash-8b")
            response = model.generate_content(prompt)
            return {"answer": response.text}
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while processing your query. Please try again later."


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    


import google.generativeai as genai
gemini_api_key = "AIzaSyD8yjZrZx29L90x9AZkj7ofi1cnZBIdKTc"
genai.configure(api_key=gemini_api_key)
from fastapi import FastAPI, File, HTTPException, UploadFile





def get_transcript_summary_store(file_path):
    try:
        myfile = genai.upload_file(file_path)
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        result = model.generate_content([myfile, """2000Summarize the complete audio...It is probably a audio recording of meeting or talk
                                        include all the important parts/ 
                                        details which might be needed in future
                                        TIMESTAMPS,DATES,TIME,LOCATIONS,FIGURES...if it mentions ..else ignore etc"""])
        print(f"{result.text=}\n")

        
        vectorDBText=model.generate_content(f""" Generate 400 Tokens or less semantic search vector text=> 
                                            extract keywords and other probable commanly searched
                                            terms to store as vector...so when people search for following meeting
                                            transcript ...400 tokens only ...include tags and other keywords for fast semantic search
                                            TRANSCRIPT\n {result.text}
""")
        print(vectorDBText.text)
        final={"name": f"meeting_name={file_path}",
        "$vectorize": file_path+"\n"+vectorDBText.text , "transcript":result.text}

        # result = []
        # for idx, chunk in enumerate(chunks):
        #     result.append({
        #         "chunk_content": chunk,
        #         "chunk_number": idx + 1
        #     })

        # formatted_chunks=[]
        # for chunk in result:
        #     formatted_chunks.append({
        #         "name": f"meeting_name={file_path}=>chunk_no={chunk['chunk_number']}",
        #         "$vectorize": chunk['chunk_content']
        #     })

        client = DataAPIClient("AstraCS:ZuAaoZrOQjHDBEIoQTpFGDzZ:2a45ed1b058aef015c652b68d88adb201838c52c05a3c000586e10746cce1533")
        database = client.get_database("https://fec84a84-0a9a-45df-8c4a-d39da233e4d6-us-east-2.apps.astra.datastax.com")

        
        collection = database.meetings
        collection.insert_one(final)
        
        # for chunk in formatted_chunks:
        #     print(chunk)
        #     collection.insert_one(chunk)

        return final
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")




@app.post("/uploadfile/")
async def upload_meeting_file(file: UploadFile = File(...)):
    file_path = f"{file.filename}"
    with open(file_path, "wb+") as file_object:
        file_object.write(file.file.read())
    get_transcript_summary_store(file_path)
    return {"filename": file.filename, "filepath": file_path}

@app.post("/meeting")
async def query_meetings(query: str):
    try:
        prompt = f"""
{get_ist_time()}.

Greeting should be:

"Good morning" if the time is before 12:00 PM
"Good afternoon" if the time is between 12:00 PM and 5:00 PM
"Good evening" if the time is after 5:00 PM,

Question: {query}

Guidelines:

Comprehensive Response: Provide a detailed and informative answer to the query, drawing information from the meeting transcript.
Markdown Formatting: Format the response in Markdown for better readability.
Accuracy and Factuality: Avoid hallucinations and provide accurate information based solely on the given transcript.
Concise and Relevant: Keep the response focused and avoid unnecessary details.
Clarity and Coherence: Structure the response logically and use clear language.
Engagement: Use appropriate language and tone to engage the user.
Multiple Meeting transcripts: Multiple meeting transcripts are provided and query will mention the meeting name use that to choose correct transcript as source
Dont mix diffrent transcripts to answer query\n\n
Meeting Transcript In Vector DB Retrived wrt to user query: \n
"""

        # 1. Query Astra DB for the document content based on the user's query
        client = DataAPIClient("AstraCS:ZuAaoZrOQjHDBEIoQTpFGDzZ:2a45ed1b058aef015c652b68d88adb201838c52c05a3c000586e10746cce1533")
        database = client.get_database("https://fec84a84-0a9a-45df-8c4a-d39da233e4d6-us-east-2.apps.astra.datastax.com")
        collection = database.meetings
        
        results_ite = collection.find(
        {},
        projection={"*": 1},
        limit=3,
        sort={"$vectorize": query},)
        
        query = results_ite.get_sort_vector()
        i=0
        for doc in results_ite:
            i+=1
            print(f"main_url={doc['name']} \ncontent=> {doc['transcript']}")


            prompt += f"Transcript {i}: {doc['name']}\n"
            prompt += f"Content: {doc['transcript'].strip().replace(' ', '')}\n\n"
    

        try:
            model = genai.GenerativeModel("gemini-1.5-flash-8b")
            response = model.generate_content(prompt)
            return {"answer": response.text}
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while processing your query. Please try again later."


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    


