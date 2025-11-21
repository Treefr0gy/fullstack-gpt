import json
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
import streamlit as st


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message.replace("$", "\$"))

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
memory = st.session_state["memory"]

if "caches" not in st.session_state:
    st.session_state["caches"] = []


def load_memory(_):
    return memory.load_memory_variables({})["history"]


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})
    if role == "human":
        memory.save_context({"input": message}, {"output": ""})
    elif role == "ai":
        memory.save_context({"input": ""}, {"output": message})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message.replace("$", "\$"))
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def save_cache(question, answer):
    st.session_state["caches"].append({"question": question, "answer": answer})


caches_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            아래 List의 Object 중에 Object의 question이 아래 질문과 똑같은 question을 가진 Object가 있어?
            
            만약 있다면, correct는 True로 반환하고, answer를 해당 Object의 answer로 반환해.
            
            없다면, correct는 False로 반환하고, answer는 ""로 반환해.

            항상 correct와 answer를 가지고 있는 객체 형식으로 반환해.

            아래의 주어진 List의 Object 중에 Object의 question들과 질문으로만 비교하고, List: []이면 False를 반환해.
            
            절대 지어내지마.

            List: {list}
            질문: {question}
            """,
        )
    ]
)


def find_cache(message):
    function = {
        "name": "find_question",
        "description": "function that takes a list of questions and answers and returns a answer and correct",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string",
                },
                "correct": {
                    "type": "boolean",
                },
            },
            "required": ["question", "answer", "correct"],
        },
    }


    caches_llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
    ).bind(
        function_call={
            "name": "find_question",
        },
        functions=[
            function,
        ],
    )

    caches_chain = caches_prompt | caches_llm
    response = caches_chain.invoke(
        {
            "list": st.session_state["caches"],
            "question": message,
        }
    )


    response = response.additional_kwargs["function_call"]["arguments"]
    if json.loads(response)["correct"]:
        return json.loads(response)["answer"]
    else:
        return False


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    history = inputs["history"]
    answers_llm = ChatOpenAI(
        temperature=0.1,
    )
    answers_chain = answers_prompt | answers_llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
        "history": history,
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the follwing pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recentones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    history = inputs["history"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
            "history": history,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        retriever = load_website(url)
        paint_history()
        message = st.chat_input("Ask a question to the website.")
        if message:
            send_message(message, "human")
            cache = find_cache(message)
            if cache:
                send_message(cache, "ai")
            else:
                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                        "history": RunnableLambda(load_memory),
                    }
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )
                with st.chat_message("ai"):
                    chain.invoke(message)
            save_cache(message, st.session_state["messages"][-1]["message"])            


else:
    st.session_state["messages"] = []