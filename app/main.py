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
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# 3. 載入 embedding 結果 (JSON)
with open("data/embedding_results.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

# 4. 建立 FAISS 向量資料庫
texts = [item['file'] for item in embedding_data]
embeddings = [item['embedding'] for item in embedding_data]
faiss_db = FAISS.from_embeddings(list(zip(texts, embeddings)), embedding_model)

# 5. Prompt 模板
prompt_template = """
你是一位 maimai 遊戲專家，根據以下的攻略資料，回答玩家的問題。
如果你不知道答案，請說「這部分我不確定，但我可以幫你查詢更多資料。」。

攻略資料：
{context}

問題：
{question}

請用簡潔清楚的方式回答：
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
    result = qa_chain.invoke({"query": user_query})
    bot_reply = result['result']
    history.append((user_query, bot_reply))
    return history, history

gr_app = gr.Blocks()
with gr_app:
    gr.Markdown("# 🎮 maimai RAG 問答機器人")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(label="請輸入你的問題", placeholder="例如：什麼是五星？")
    submit_btn = gr.Button("送出問題")
    state = gr.State([])

    submit_btn.click(fn=rag_answer, inputs=[user_input, state], outputs=[chatbot, state])
    user_input.submit(fn=rag_answer, inputs=[user_input, state], outputs=[chatbot, state])

if __name__ == "__main__":
    gr_app.launch(server_name="0.0.0.0", server_port=8000)
