import os
from dotenv import load_dotenv
import yaml
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings

# 讀取 .env 檔案
load_dotenv()

# 環境變數自動載入 OPENAI_API_KEY
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("未找到 OPENAI_API_KEY，請確認 .env 設定正確。")

# 初始化 LangChain 的 Embedding 接口
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# 遞迴找出 "HTML資料" 資料夾中所有 YAML 檔案
yaml_files = []
for root, dirs, files in os.walk("HTML資料"):
    for file in files:
        if file.endswith(".yaml") or file.endswith(".yml"):
            yaml_files.append(os.path.join(root, file))

# 讀取 YAML 內容並轉文字
yaml_texts = []
for file_path in yaml_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        text = yaml.dump(data, allow_unicode=True)
        yaml_texts.append((file_path, text))

# 做 embedding
embeddings_result = []

for file_path, text in tqdm(yaml_texts, desc="Embedding YAML files"):
    try:
        embedding = embedding_model.embed_query(text)
        embeddings_result.append({"file": file_path, "embedding": embedding})
    except Exception as e:
        print(f"Error embedding {file_path}: {e}")

# 儲存結果
import json
with open("embedding_results.json", "w", encoding="utf-8") as f:
    json.dump(embeddings_result, f, ensure_ascii=False, indent=2)

print("Embedding 完成！共處理", len(embeddings_result), "個檔案")
