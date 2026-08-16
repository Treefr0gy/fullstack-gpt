import openai as client
import time
import streamlit as st

assistant_id = "asst_7jrL5GQPPh3kU2kwhxjw50nk"


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def wait_on_run(run_id, thread_id):
    run = get_run(run_id, thread_id)
    while run.status == "queued" or run.status == "in_progress":
        run = get_run(run_id, thread_id)
        time.sleep(0.5)
    return run


st.set_page_config(
    page_title="BookAssistant",
    page_icon="📃",
)


@st.cache_data(show_spinner="Uploading file...")
def upload_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    return client.files.create(
        file=client.file_from_path(file_path), purpose="assistants"
    )


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


st.markdown(
    """
    # BookAssistant

    Welcome to BookAssistant.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    uploaded_file = upload_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        if st.session_state["thread_id"] is None:
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                        "attachments": [
                            {"file_id": uploaded_file.id, "tools": [{"type": "file_search"}]}
                        ],
                    }
                ]
            )
            st.session_state["thread_id"] = thread.id
        else:
            client.beta.threads.messages.create(
                thread_id=st.session_state["thread_id"],
                role="user",
                content=message,
            )
        run = client.beta.threads.runs.create(
            thread_id=st.session_state["thread_id"],
            assistant_id=assistant_id,
        )
        run = wait_on_run(run.id, st.session_state["thread_id"])
        result = get_messages(st.session_state["thread_id"])[-1].content[0].text.value
        send_message(result.replace("$", "\$"), "ai")
else:
    st.session_state["messages"] = []
    st.session_state["thread_id"] = None
