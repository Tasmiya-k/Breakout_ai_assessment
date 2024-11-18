import os
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from langchain import SerpAPIWrapper, LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks import StreamlitCallbackHandler
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from itertools import product
from sentence_transformers import util
from langchain.docstore.document import Document
import time
from io import BytesIO
from dotenv import load_dotenv


load_dotenv()
st.set_page_config(page_title="InfoXtract", page_icon="logo_1.png")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
llm = ChatMistralAI(model='mistral-large-latest', temperature=0, streaming=True)
search = SerpAPIWrapper()

# Load CSV file
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


# FAISS Embedding
def create_faiss_index(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    index = FAISS.from_documents(docs, embeddings)
    return index

# Question Answering Chain
def initialize_qa_chain(output_column):
    llm = ChatMistralAI(model='mistral-large-latest', temperature=0)
    prompt_template = f"""
        You are given a list of queries and their corresponding responses. Extract a single keyword for each response 
        that represents the '{output_column}' and return it as a list.

        Example:
        Query: Find headquarter of ByteDance, Response: The headquarters of ByteDance is located in Haidian, Beijing.
        Output: Beijing

        Query: Find headquarter of SpaceX, Response: The headquarters of SpaceX is in Hawthorne, California.
        Output: California

        Query: Find headquarter of Stripe, Response: Stripe has headquarters in San Francisco, USA and Dublin, Ireland.
        Output: Dublin

        Data:
       {{context}}

        Provide the outputs as a list of single keywords, one per response. I only need single column. No special symbols other than text. 
    """
    med_prompt = PromptTemplate.from_template(prompt_template)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=med_prompt)
    return chain


# Process responses with chain
def process_responses_with_chain(queries, responses, output_columns, chain):
    all_keywords = {}
    for col in output_columns:
        documents = [
            Document(page_content=f"Query: {query}, Response: {response}")
            for query, response in zip(queries, responses)
        ]
        try:
            response = chain({"input_documents": documents, "question": col})
            keywords = response.get("output_text", "Unknown").splitlines()
        except Exception as e:
            st.error(f"Error processing responses for column {col}: {e}")
            keywords = ["Unknown"] * len(queries)
        all_keywords[col] = keywords

    df = pd.DataFrame(all_keywords)
    return df




def web_search(query):    

     # Streamlit container for streaming responses
    
    final_response_container = st.empty()

    # Optional: LLMMathChain for math-related queries
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)

    # Define tools for the agent
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for answering questions about current events or retrieving recent information."
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="Useful for answering math-related questions."
        ),
    ]

    # Create the agent with Chat Mistral AI and SerpAPI
    agent = initialize_agent(
        tools, llm, agent="chat-zero-shot-react-description", verbose=False
    )
    response_container = st.empty()
    

    # Callback for streaming output to Streamlit
    callback_handler = StreamlitCallbackHandler(response_container, expand_new_thoughts=False)

    response = agent.run(query, callbacks=[callback_handler])
    final_response_container.write(f"**Final Response:** {response}")
    return response

# Batch Web Search with Retry Logic
def web_search_batch(queries, batch_size=5, delay=1, max_retries=3):
    """
    Perform batched web searches to avoid rate-limit issues.
    
    :param queries: List of queries to process.
    :param batch_size: Number of queries to process in each batch.
    :param delay: Delay (in seconds) between batches.
    :param max_retries: Maximum retry attempts in case of rate limit exceeded error.
    :return: List of responses.
    """
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        batch_results = []
        
        for query in batch:
            attempt = 0
            while attempt < max_retries:
                try:
                    # Call the web_search function
                    response = web_search(query)
                    batch_results.append(response)
                    break  # Exit the retry loop if successful
                
                except Exception as e:
                    if "429" in str(e):  # Rate limit exceeded
                        wait_time = 2 ** attempt  # Exponential backoff (1, 2, 4, 8, ...)
                        print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)  # Wait before retrying
                        attempt += 1
                    else:
                        print(f"Error: {e}")
                        batch_results.append(f"Error: {e}")
                        break  # Exit retry loop for other errors
            
            if attempt == max_retries:
                print(f"Max retries reached for query: {query}")
                batch_results.append(f"Max retries reached for query: {query}")
        
        results.extend(batch_results)
        
        # Delay before next batch
        if i + batch_size < len(queries):  # Avoid delay after the last batch
            print(f"Waiting {delay} seconds before processing next batch...")
            time.sleep(delay)
    
    return results


  
def save_to_excel(df, output_file):
    if os.path.exists(output_file):
        with pd.ExcelWriter(output_file, mode='a', if_sheet_exists='overlay') as writer:
            if 'Sheet1' in writer.sheets:
                max_col = writer.sheets['Sheet1'].max_column
            else:
                max_col = 0
            df.to_excel(writer, index=False, sheet_name="Sheet1", startcol=max_col)
    else:
        df.to_excel(output_file, index=False) 


# Streamlit interface and main workflow
def main():
    st.title("Welcome to InfoXtract: Upload Your Spreadsheet and Query Data by Any Column")

    # Step 1: Upload CSV
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        data = load_csv(uploaded_file)
        st.write("CSV Data:", data.head())

        save_path = "uploaded_file.csv"
        data.to_csv(save_path, index=False)

        # Step 2: Select Columns for Entities
        entity_columns = st.multiselect("Select one or more columns to use as entities", data.columns)
        query_template = st.text_input("Enter your query template (e.g., 'Find the headquarters location of {company} in {country}')")
        queries =[]
        responses=[]
        if query_template:
            entity_combinations = product(*(data[col].dropna().unique() for col in entity_columns))
            queries = [query_template.format(**dict(zip(entity_columns, combination))) for combination in entity_combinations]
        
    
            print(queries)
            # Run web search 
            if "responses" not in st.session_state:
                with st.spinner("Processing queries..."):
                    responses = web_search_batch(queries, batch_size=5, delay=2)
                    st.session_state["responses"] = responses  # Save responses in session state
                    st.session_state["queries"] = queries
            else:
                responses = st.session_state["responses"]
                queries = st.session_state["queries"]

            # Display results
            for query, response in zip(queries, responses):
                st.write(f"Query: {query}")
                st.write(f"Response: {response}")
        
            output_columns = st.text_input("Enter output column names (comma-separated):").split(',')
            chain = initialize_qa_chain(output_columns[0] if output_columns else "Output")
            

            if output_columns:
                all_keywords = process_responses_with_chain(queries, responses, output_columns, chain)
                result_df = pd.DataFrame(all_keywords)
                        
                    
                result_df = pd.concat([data, result_df], axis=1)

               
                output_buffer = BytesIO()
                with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False, sheet_name="Sheet1")
                     # Prepare file for download
                st.download_button(
                        label="Download Results",
                        data=output_buffer.getvalue(),
                        file_name="output_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                              

if __name__ == "__main__":
    main()
