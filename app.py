from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

from decouple import config

import os
os.environ["OPENAI_API_KEY"] = config('openai_key')


def main_app(query, file_name):

    # Load the document
    pdf_path = f"./{file_name}"
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(pages, embedding=embeddings, 
                                    persist_directory=".")
    vectordb.persist()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8) , vectordb.as_retriever(), memory=memory)


    result = pdf_qa({"question": query})
    res = result["answer"]
    return res


if __name__ == "__main__":
    file_name = input("Enter the file name: ")
    user_input = input("Enter your question: ")
    answer = main_app(user_input, file_name)

    _file_name = str(file_name).replace(".pdf", ".txt")
    with open(_file_name, 'a+') as f:
        f.write(f"Q: " + user_input + "\n\n")
        f.write("Ans: "+answer + "\n\n\n\n")