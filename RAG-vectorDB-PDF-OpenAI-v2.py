#current Python 3.12

import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"]='OPENAI_API_KEY'

loader = DirectoryLoader("./documents/",glob="./*.txt",loader_cls=TextLoader)

documents = loader.load()
##print(documents)

#apply chunks for LLM
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
##print(texts)

print(len(texts)) #number of chunks for LLM to consume

#apply vectorDB using chromaDB

persist_directory = "db"

#pip install -U langchain-openai
embedding = OpenAIEmbeddings()

#On execution - the creation of the db folder.
vectorDB = Chroma.from_documents(
    documents = texts,
    embedding = embedding,
    persist_directory = persist_directory
)

# search_params = {
#     'k': 5,  # Retrieve the top 5 nearest neighbors
#     'filter': {'category': 'science'},  # Filter by category
#     'distance_metric': 'cosine',  # Use cosine similarity
#     'include': ['metadata'],  # Include metadata in results
#     'max_distance': 0.5  # Limit results to within a distance of 0.5
# }
search_params = {
    'k': 2,  # Retrieve the top 5 nearest neighbors
}

#retriever = vectorDB.as_retriever(search_kwargs={"k":2})
retriever = vectorDB.as_retriever(search_kwargs=search_params)
print(retriever)

#turbo_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
turbo_llm = ChatOpenAI(temperature=1, model_name="gpt-4o-mini")

rag_chain = RetrievalQA.from_chain_type(
    llm=turbo_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def _get_llm_response_sources(query, llm_response):
    print('\n\n' + query)
    print('\n\n' + llm_response["result"])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

def _get_llm_response(query,llm_response):
    print('\n\n' + query)
    print(llm_response["result"])

query = "what is about the dragon"
llm_response = rag_chain(query)
##_get_llm_response_sources(llm_response)
_get_llm_response(query, llm_response)

query = "Do these documents contain any patents? If so, please summarize them. If you cannot provide a summary of these patents then summary the tiger story."
llm_response = rag_chain(query)
##_get_llm_response_sources(llm_response)
_get_llm_response(query, llm_response)

#################################

while True:
    user_query = input("\n\nPlease enter your question/prompt (or type quit to exit): ")

    # Check if the user wants to quit
    if user_query.lower() == 'quit':
        print("Exiting the program.")
        break
    # Once the user has entered their question, pass it to the qa_chain function
    llm_response = rag_chain(user_query)

    # Process the LLM response as needed
    # Replace the following line with the actual processing function if different
    _get_llm_response(user_query, llm_response)

#suggest prompts:
# is there a topic about healthcare?
# summary these healthcare documents
# summary about fluffington
# is there a tiger in these document and what is it about?
# is there a hawk and condor in any of these documents? please summary this.
# summary about the story for a hawk and a condor
