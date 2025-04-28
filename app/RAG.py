import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 1. 載入 .env 並設定 API KEY
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 2. 初始化 embedding 模型與 LLM (GPT-4o-mini)
embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# 3. 載入 embedding 結果 (JSON)
with open("embedding_results.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

# 4. 使用 content 作為 FAISS 檢索資料
texts = [item['content'] for item in embedding_data]
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

請詳細且完整地回答玩家的問題，提供具體建議與說明：
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

# 7. 互動式 QA 查詢
print("\n我是 maimai LLM QA 機器人！請輸入問題，輸入 'exit' 離開\n")

while True:
    query = input("問題： ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("再見！👋")
        break

    answer = qa_chain.invoke(query)
    print(answer['result'])
    print("---")
