# Agent = [CognitionTestHandler]
# Tools = [QuestionGenerator, AnswerValidator-Script, ScoreCalculator-Script, Cognitive Analyzer]
# while true:
#     if
# Chain = promt_template(self introduction)

import os
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import JSONLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentExecutor, create_json_chat_agent
from langchain.memory import ConversationBufferMemory
from pathlib import Path
from pprint import pprint
import json
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env file
load_dotenv()

# Define the directory containing the text file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "questions", "quiz_questions.json")
db_dir = os.path.join(current_dir, "db2")

# Testing the json loader - Good!!
data = json.loads(Path(file_path).read_text())

# Load the document containing the quiz questions
loader = JSONLoader(file_path=file_path,
                    jq_schema='.[]',
                    text_content=False)

# Load data into documents
documents = loader.load()

# Create an embedding for documents
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Function to create and persist vector store
def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")


# Create a vector store for retrieval after splitting

print("\n--- Using JSON Text Splitting ---")
# json_splitter = RecursiveJsonSplitter(max_chunk_size=70)
# json_chunks = json_splitter.split_json(json_data=data)
create_vector_store(documents, "chroma_db_json")

# Create a ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Load the existing vector store with the embedding function
chroma_db_json = Chroma(persist_directory=db_dir, embedding_function=embeddings)
retriever = chroma_db_json.as_retriever()

# Create a RetrievalQA chain to fetch questions based on user input or query
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

def fetch_question(query):
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(chroma_db_json.as_retriever(), combine_docs_chain)

    rag_chain.invoke({"input" : query})

prompt = hub.pull("hwchase17/react-chat-json")

# Define tools to be used by the agent
tools = [
    Tool(
        name="RetrievalQA",
        func=fetch_question,
        description="Use this to retrieve quiz questions."
    )
]

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# Initialize the agent with tools
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent="conversational-react-description",  # A conversational agent type # type: ignore
#     verbose=True,
#     memory=memory
# )

agent = create_json_chat_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

ans = 0
while ans == 0:
    agent_executor.invoke({"input": "Ask me one of the math questions"})


# def conduct_quiz(agent, num_questions=5):
#     print("Welcome to the Quiz! Answer the questions below:")
    
#     for i in range(num_questions):
#         # Retrieve a question using the agent
#         response = agent.run(input="Fetch the next quiz question.")
        
#         # Extract the question and options from the response
#         question = response['question']
#         options = response['options']
        
#         print(f"\nQuestion {i+1}: {question}")
#         for idx, option in enumerate(options):
#             print(f"{idx + 1}. {option}")

#         # Get user's answer
#         user_answer = input("Enter your answer (1/2/3/4): ")
        
#         # Check if the answer is correct
#         correct_answer = response['answer']
#         if options[int(user_answer) - 1].strip().lower() == correct_answer.strip().lower():
#             print("Correct!\n")
#         else:
#             print(f"Wrong! The correct answer is: {correct_answer}\n")
    
#     print("Quiz Completed!")

# # Run the quiz
# conduct_quiz(agent)
