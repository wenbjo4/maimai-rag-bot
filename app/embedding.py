import os
import json
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings

# 載入 .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 初始化 Embedding 模型
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# 找出所有 YAML 檔案
yaml_files = []
for root, dirs, files in os.walk("data/HTML資料"):
    for file in files:
        if file.endswith(".yaml") or file.endswith(".yml"):
            yaml_files.append(os.path.join(root, file))

# 讀取 YAML 並以 title 為單位做 embedding
embeddings_result = []

for file_path in yaml_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        for entry in data:
            title = entry.get("title", "No Title")
            content = entry.get("content", "")
            text_to_embed = f"{title}\n{content}"

            try:
                embedding = embedding_model.embed_query(text_to_embed)
                embeddings_result.append({
                    "title": title,
                    "embedding": embedding
                })
            except Exception as e:
                print(f"Error embedding title '{title}' in {file_path}: {e}")

# 儲存結果
with open("data/embedding_results.json", "w", encoding="utf-8") as f:
    json.dump(embeddings_result, f, ensure_ascii=False, indent=2)

print("Embedding 完成！共處理", len(embeddings_result), "個 entries")
