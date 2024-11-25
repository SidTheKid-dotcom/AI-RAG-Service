
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

