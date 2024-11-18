# InfoXtract: Query Data from Spreadsheets with Advanced NLP 

## Overview

**InfoXtract** is a Streamlit-based application designed to process CSV files, extract data from columns, and query information using advanced natural language processing techniques. It leverages cutting-edge tools like LangChain, FAISS, HuggingFace embeddings, and Chat Mistral AI to provide efficient and insightful data extraction and summarization.

---

## Features

- **Upload CSV files**: Easily upload and view your dataset.
- **Dynamic Querying**: Generate queries using a template and selected column combinations.
- **Batch Processing**: Perform web searches in batches with retry logic to handle rate limits.
- **Keyword Extraction**: Extract meaningful keywords from query responses using AI-driven question-answering chains.
- **FAISS Embedding**: Create embeddings for advanced similarity searches.
- **Downloadable Output**: Save the processed data as an Excel file with the added results.

---

## Technologies Used

- **Streamlit**: For building the web interface.
- **LangChain**: For NLP chains and integration with LLMs.
- **FAISS**: For fast and efficient similarity searches.
- **Chat Mistral AI**: For question answering and conversational intelligence.
- **HuggingFace Transformers**: For embedding generation.
- **SerpAPI**: For real-time web searches.
- **Pandas**: For data manipulation.
- **Openpyxl**: For Excel file generation.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repository/InfoXtract.git
   cd InfoXtract 
   ```
2. **Set up a virtual environment (optional but recommended)**:
For a clean setup, it’s advised to use a virtual environment:
 
   `python -m venv venv`\
   `source venv/bin/activate`\
   `venv\Scripts\activate`

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```
4. **Set up environment variables:**\
   Create a .env file in the project root directory.\
   Add the following keys and their respective values:
   ```bash 
   MISTRAL_API_KEY=<Your Mistral API Key>
   HUGGINGFACE_API_TOKEN=<Your HuggingFace API Token>
   SERPAPI_API_KEY=<Your SerpAPI API Key>
   ```
## Run the application 
To start the application, execute: \
`streamlit run app.py`

## Usage

1. **Upload CSV File**:  
   Drag and drop your CSV file into the uploader section of the application.

2. **Select Entity Columns**:  
   Choose one or more columns from your dataset for querying purposes.

3. **Enter Query Template**:  
   Provide a dynamic query template. For example:  
   ```plaintext
   Find the headquarters of {company} in {country}.

4. **Generate Queries**:
The system will create a query for each row in the dataset using the provided template.

5. **Perform Web Searches**:
Queries are processed in batches using SerpAPI, with retry logic for API limits.

6. **Extract Keywords**:
Extract relevant keywords or entities from the responses using LangChain's NLP chains. Format the sheet with output column.

7. **Download the Results**:
Export the processed dataset as an Excel file, complete with added columns from the query.

## Example Workflow
1. Upload a CSV file with the following data:
| Company    | Year Establish  |
|------------|------------|
| ByteDance  | 2012       | 
| SpaceX     | 2002       | 

2. Select Company as entity columns.
3. Enter a query template:
`Find the headquarters of {Company} `

4. The system processes each query and extracts responses such as:
`ByteDance → Beijing
SpaceX → California`

5. Take input output column name. 
Here, `Headquarters`

6. The final output is saved as an Excel file with additional columns like Headquarters.

## Contact
For further queries, reach out to Tasmiya Khan at tasmiyakhan2798@gmail.com.

## Link to video
 Link to the video: https://drive.google.com/file/d/1Ta4ayMVUT-khfp8Ffp44KIHjpf51bZLJ/view?usp=drivesdk

