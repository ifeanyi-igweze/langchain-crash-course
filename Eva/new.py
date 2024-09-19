from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.structured import StructuredOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the dictionary of questions
questions_dict = {
    1: "What is your name?",
    2: "How old are you?",
    3: "What is your favourite colour?",
}

# Define the prompt template for asking questions
prompt_template_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a quiz bot asking the user a multiple choice math question set"),
        ("human", "introduce yourself")
    ]
)

# Initialize the OpenAI LLM (you can use any LLM of your choice)
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Use the appropriate model

answers = {}
key_list = list(questions_dict.keys())

# Lambda function that iterates over the dictionary and collects responses
def ask_questions_lambda(bank:dict):
    key = key_list[-1] # Get the next key
    response = bank["question"].format(questions_dict[key])   
    
    # Simulate asking the user (in an actual implementation, you'd capture real user input)
    user_input = input(f"{response}\n")  # Asking the question to the user

    # Store the response
    answers[key] = user_input
    key_list.pop(-1)

    return answers

# Create the combined chain using LangChain Expression Language (LCEL)
chain =  prompt_template_1 | llm | RunnableLambda(ask_questions_lambda)
# chain = prompt_template | model

# Run the chain
result = chain.invoke({"bank": questions_dict, "answers": answers})

# Output
print(result)
