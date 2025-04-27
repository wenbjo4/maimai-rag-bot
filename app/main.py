import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 1. 載入 .env 並設定 API KEY
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 2. 初始化 embedding 模型與 LLM (GPT-4o)
embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

# 3. 載入 embedding 結果 (JSON)
with open("data/embedding_results.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

# 4. 建立 FAISS 向量資料庫
texts = [item['title'] for item in embedding_data]
embeddings = [item['embedding'] for item in embedding_data]
faiss_db = FAISS.from_embeddings(list(zip(texts, embeddings)), embedding_model)

# 5. Prompt 模板
prompt_template = """
你是一位 maimai 遊戲專家，根據以下的攻略資料，詳細回答玩家的問題。
請提供具體的建議，並盡可能解釋相關概念。
如果資料中無法找到答案，也可以根據你已知的資訊推測，並說明。

攻略資料：
{context}

問題：
{question}

請詳細說明你的回答：
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# 6. RAG 查詢流程
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_db.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# 7. Gradio UI
def rag_answer(user_query, history):
    # 擷取引用的資料 context
    retrieved_docs = faiss_db.similarity_search(user_query, k=5)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # 把 context 傳入 prompt
    result = qa_chain.invoke({"query": user_query, "context": context})
    bot_reply = result['result']

    # 更新對話紀錄
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": bot_reply + "\n\n🔍 引用資料:\n" + context})
    return "", history, history


gr_app = gr.Blocks()
with gr.Blocks() as gr_app:
    gr.Markdown("# 🎮 maimai RAG 問答機器人")
    chatbot = gr.Chatbot(type='messages')
    user_input = gr.Textbox(label="請輸入你的問題", placeholder="例如：KOP是什麼？")
    submit_btn = gr.Button("送出問題")
    state = gr.State([])

    def disable_input():
        return gr.update(interactive=False)

    def enable_input():
        return gr.update(interactive=True)

    submit_btn.click(disable_input, None, [user_input]) \
        .then(rag_answer, [user_input, state], [user_input, chatbot, state]) \
        .then(enable_input, None, [user_input])

    user_input.submit(rag_answer, [user_input, state], [user_input, chatbot, state])

if __name__ == "__main__":
    gr_app.launch(server_name="0.0.0.0", server_port=8000)
