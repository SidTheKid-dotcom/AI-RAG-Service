
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
        GITHUB_TOKEN = "-----------------------------------"

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
    client = DataAPIClient("-----------------------------------")
    database = client.get_database("-----------------------------------")
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
    github_token = "-----------------------------------"
    headers = {"Authorization": f"token {github_token}"}

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants and configurations
GITHUB_TOKEN = "-----------------------------------"
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

