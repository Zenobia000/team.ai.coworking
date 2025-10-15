# MLOps Pipeline 整合 LLMOps：實戰指南

## 目錄
1. [傳統 MLOps Pipeline 現況](#1-傳統-mlops-pipeline-現況)
2. [LLMOps 新增能力對照](#2-llmops-新增能力對照)
3. [整合架構設計](#3-整合架構設計)
4. [實作步驟](#4-實作步驟)
5. [工具鏈選擇](#5-工具鏈選擇)
6. [監控與治理](#6-監控與治理)

---

## 1. 傳統 MLOps Pipeline 現況

### 典型的 MLOps 生命週期

```
數據收集 → 特徵工程 → 模型訓練 → 模型評估 → 模型部署 → 監控 → 重訓練
   ↓           ↓           ↓          ↓          ↓        ↓        ↓
 DVC       Transform   Training   Metrics   Serving   Drift    Trigger
          Pipeline     Job       Report              Detection
```

### 傳統 MLOps 的核心組件

| 階段 | 工具/技術 | 輸入 | 輸出 |
|------|----------|------|------|
| **數據版本控制** | DVC, Delta Lake | 原始數據 | 版本化數據集 |
| **特徵工程** | Feature Store (Feast) | 數據集 | 特徵向量 |
| **實驗追蹤** | MLflow, Weights & Biases | 訓練參數 | 實驗結果 |
| **模型訓練** | TensorFlow, PyTorch | 特徵 + 標籤 | 模型權重 |
| **模型註冊** | Model Registry | 訓練好的模型 | 版本化模型 |
| **模型部署** | TF Serving, Triton | 模型 + 配置 | 推論端點 |
| **監控** | Prometheus, Grafana | 推論日誌 | 漂移警報 |

---

## 2. LLMOps 新增能力對照

### LLMOps vs MLOps：關鍵差異

| 維度 | 傳統 MLOps | LLMOps | 為什麼不同？ |
|------|-----------|--------|------------|
| **模型來源** | 從零訓練 | 使用基礎模型 + 微調 | LLM 預訓練成本 $10M+，企業負擔不起 |
| **"程式碼"** | Python 訓練腳本 | Prompt 提示詞 | Prompt 是 LLM 的"程式碼"，需版本控制 |
| **數據需求** | 大量標註數據 | 少量範例 + RAG 知識庫 | Few-shot Learning + 外部知識檢索 |
| **輸出類型** | 數值/分類 | 自然語言/結構化 | 需要輸出解析、防護欄、驗證 |
| **評估方式** | 準確率/F1/AUC | LLM-as-Judge + 人類評分 | 生成質量難以自動量化 |
| **推論成本** | 固定 GPU 成本 | 按 Token 計費 | 需要成本優化策略（快取/壓縮/路由） |
| **部署策略** | 單模型服務 | 多模型路由 + Agent 編排 | 複雜任務需要多步驟、工具調用 |

### LLMOps 必須新增的 7 個管線階段

```
[傳統 MLOps Pipeline]
    +
[Prompt 工程] → [RAG 建置] → [LLM 微調] → [防護欄] → [Agent 編排] → [成本優化] → [質量評估]
    ↓              ↓            ↓           ↓           ↓             ↓            ↓
 Prompt       Vector DB    LoRA/PEFT   Guardrails  LangGraph    Cache/Route   LLM Judge
 Registry                                                        + Compress
```

---

## 3. 整合架構設計

### 3.1 統一 MLOps + LLMOps 管線架構

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          統一數據層 (Unified Data Layer)                  │
│  - 結構化數據 (Feature Store) + 非結構化數據 (文檔/圖片/程式碼)             │
│  - DVC (ML 數據) + Vector DB (LLM 知識庫)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        雙軌訓練管線 (Dual Training)                        │
│  ┌──────────────────────┐              ┌──────────────────────────┐     │
│  │  傳統 ML 訓練         │              │  LLM 微調/Prompt 優化     │     │
│  │  - 特徵工程           │              │  - Prompt 版本控制        │     │
│  │  - 模型訓練           │              │  - Few-shot 範例庫       │     │
│  │  - 超參數調優         │              │  - LoRA/QLoRA 微調       │     │
│  │  - 交叉驗證           │              │  - RAG 管線建置          │     │
│  └──────────────────────┘              └──────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        統一模型註冊表 (Unified Registry)                   │
│  - ML Models (TensorFlow/PyTorch)  +  LLM Models (GGUF/SafeTensors)     │
│  - Prompt Templates + RAG Configs + Agent Workflows                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      統一推論閘道 (Unified Inference Gateway)              │
│  ┌──────────────────────┐              ┌──────────────────────────┐     │
│  │  ML 模型服務          │              │  LLM 服務                 │     │
│  │  - Triton/TF Serving │◄─────路由────►│  - vLLM/TGI/API         │     │
│  │  - ONNX Runtime      │              │  - 智能快取               │     │
│  │  - <10ms 延遲        │              │  - Agent 編排             │     │
│  └──────────────────────┘              └──────────────────────────┘     │
│                                  ↓                                       │
│                         結果融合層 (Ensemble)                             │
│                    ML 精準預測 + LLM 生成/解釋                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       統一監控與治理 (Unified Governance)                  │
│  - 模型漂移 + LLM 輸出質量監控                                             │
│  - 成本追蹤 (GPU 成本 + Token 成本)                                        │
│  - 防護欄 (安全性/合規性)                                                 │
│  - A/B 測試框架 (ML + LLM 統一)                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 關鍵設計原則

#### 原則 1：分層解耦 (Layered Decoupling)
```
應用層 (Product Layer)
    ↓ 使用標準 API
能力層 (Capability Layer) ← ML + LLM 能力在此統一封裝
    ↓ 調用底層服務
基礎設施層 (Infrastructure Layer) ← K8s / GPU / Storage
```

#### 原則 2：熱插拔模型 (Hot-swappable Models)
- ML 模型和 LLM 都註冊到統一 Registry
- 使用流量分割（Traffic Split）實現 A/B 測試
- Canary 部署：5% 流量 → 驗證 → 100% 切換

#### 原則 3：成本可歸因 (Cost Attribution)
```python
# 每次推論記錄成本
{
  "request_id": "abc123",
  "model_type": "llm",  # or "ml"
  "model_name": "gpt-4",
  "tokens": 1500,
  "cost_usd": 0.045,
  "latency_ms": 1200,
  "project": "edu-lms",
  "user_id": "user_456"
}
```

---

## 4. 實作步驟

### Phase 1：評估與準備 (Week 1-2)

#### 步驟 1.1：盤點現有 MLOps 資產
```bash
# 檢查清單
✓ 使用的實驗追蹤工具？(MLflow / W&B / ClearML)
✓ 模型部署方式？(K8s / Docker / Serverless)
✓ 監控工具？(Prometheus / Datadog)
✓ CI/CD 管線？(GitHub Actions / GitLab CI / Argo)
✓ Feature Store？(Feast / Tecton / 自建)
```

#### 步驟 1.2：選擇 LLMOps 工具鏈
| 需求 | 推薦工具 | 整合難度 | 備註 |
|------|---------|---------|------|
| **Prompt 管理** | LangSmith / PromptLayer | 低 | API 包裝即可 |
| **向量資料庫** | Qdrant / Weaviate | 中 | 需要額外部署 |
| **LLM 推論** | vLLM / LiteLLM | 中 | vLLM 需 GPU，LiteLLM 只是路由 |
| **Agent 框架** | LangGraph / AutoGen | 高 | 需要學習新範式 |
| **防護欄** | NeMo Guardrails | 中 | 配置驅動 |

---

### Phase 2：最小可行管線 (Week 3-6)

#### 目標：在不破壞現有系統的前提下，建立 LLMOps 原型

#### 步驟 2.1：建立 Prompt 版本控制

**使用 Git + YAML 管理 Prompt**

```yaml
# prompts/edu_quiz_generator_v1.yaml
version: "1.0"
description: "教育題目生成器 - 基礎版"
model: "gpt-4o-mini"
temperature: 0.7
max_tokens: 2000

system_prompt: |
  你是一位資深教育專家，擅長根據教材內容生成高質量的考題。

  要求：
  - 題目難度適中，符合目標年級
  - 包含單選、多選、簡答三種題型
  - 每題附帶詳細解析

user_prompt_template: |
  請根據以下教材內容生成 {num_questions} 道考題：

  教材主題：{topic}
  目標年級：{grade_level}

  教材內容：
  {content}

  請以 JSON 格式輸出，包含題目、選項、答案、解析。

few_shot_examples:
  - user: "教材主題：Python 迴圈，目標年級：高一"
    assistant: |
      [
        {
          "question": "下列哪個迴圈會無限執行？",
          "options": ["for i in range(5)", "while True", "for item in []"],
          "answer": "B",
          "explanation": "while True 會創建無限迴圈..."
        }
      ]
```

**整合到 MLflow**

```python
# prompt_registry.py
import mlflow
import yaml

class PromptRegistry:
    """Prompt 版本控制（類似 Model Registry）"""

    def log_prompt(self, prompt_config_path, version, metrics=None):
        """記錄 Prompt 版本"""
        with mlflow.start_run(run_name=f"prompt_v{version}"):
            # 讀取 Prompt 配置
            with open(prompt_config_path) as f:
                config = yaml.safe_load(f)

            # 記錄到 MLflow
            mlflow.log_dict(config, "prompt_config.yaml")
            mlflow.log_param("model", config["model"])
            mlflow.log_param("temperature", config["temperature"])

            # 記錄評估指標（如果有）
            if metrics:
                mlflow.log_metrics(metrics)

            # 標記為生產用 Prompt
            mlflow.set_tag("stage", "production")

    def load_prompt(self, version="latest"):
        """載入 Prompt 配置"""
        # 從 MLflow 載入最新或指定版本
        client = mlflow.tracking.MlflowClient()
        # ... 實作載入邏輯
```

---

#### 步驟 2.2：建立 RAG 管線

**架構：向量庫 + 檢索 + 重排**

```python
# rag_pipeline.py
from typing import List, Dict
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

class RAGPipeline:
    """檢索增強生成管線"""

    def __init__(self, collection_name="knowledge_base"):
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.collection_name = collection_name
        self._init_collection()

    def _init_collection(self):
        """初始化向量集合"""
        try:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI Embedding 維度
                    distance=Distance.COSINE
                )
            )
        except Exception:
            pass  # 集合已存在

    def index_documents(self, documents: List[Dict]):
        """索引文檔到向量庫"""
        points = []
        for i, doc in enumerate(documents):
            # 生成 Embedding
            embedding = self._get_embedding(doc["content"])

            points.append(PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {})
                }
            ))

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"✓ 已索引 {len(documents)} 份文檔")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """檢索相關文檔"""
        # 查詢向量化
        query_embedding = self._get_embedding(query)

        # 向量檢索
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        # 格式化結果
        docs = []
        for result in results:
            docs.append({
                "content": result.payload["content"],
                "score": result.score,
                "metadata": result.payload.get("metadata", {})
            })

        return docs

    def _get_embedding(self, text: str) -> List[float]:
        """生成文本 Embedding"""
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

# 使用範例
if __name__ == "__main__":
    rag = RAGPipeline()

    # 索引教材文檔
    documents = [
        {
            "content": "Python 的 for 迴圈用於遍歷序列...",
            "metadata": {"topic": "Python 迴圈", "grade": "高一"}
        },
        # ... 更多文檔
    ]
    rag.index_documents(documents)

    # 檢索相關內容
    query = "如何使用 Python 迴圈？"
    docs = rag.retrieve(query, top_k=3)
    print(docs)
```

---

#### 步驟 2.3：整合到現有 MLOps 管線

**使用 ClearML/MLflow 統一管理**

```python
# unified_pipeline.py
from clearml import Task, Model
from rag_pipeline import RAGPipeline
from prompt_registry import PromptRegistry
import openai

class UnifiedMLLMPipeline:
    """統一 ML + LLM 管線"""

    def __init__(self, project_name="edu-quiz-system"):
        self.task = Task.init(project_name=project_name, task_name="llm_generation")
        self.rag = RAGPipeline()
        self.prompt_registry = PromptRegistry()

    def generate_quiz(self, topic: str, num_questions: int = 5):
        """生成考題（整合 RAG + LLM）"""

        # 步驟 1：從 RAG 檢索相關教材
        self.task.get_logger().report_text("步驟 1：檢索相關教材")
        relevant_docs = self.rag.retrieve(
            query=f"{topic} 教材內容",
            top_k=3
        )

        # 步驟 2：載入 Prompt 模版
        prompt_config = self.prompt_registry.load_prompt(version="v1.0")

        # 步驟 3：組裝上下文
        context = "\n\n".join([doc["content"] for doc in relevant_docs])

        # 步驟 4：呼叫 LLM
        self.task.get_logger().report_text("步驟 4：呼叫 LLM 生成題目")
        response = openai.chat.completions.create(
            model=prompt_config["model"],
            temperature=prompt_config["temperature"],
            messages=[
                {"role": "system", "content": prompt_config["system_prompt"]},
                {"role": "user", "content": prompt_config["user_prompt_template"].format(
                    topic=topic,
                    num_questions=num_questions,
                    grade_level="高一",
                    content=context
                )}
            ]
        )

        # 步驟 5：記錄 Artifacts
        quiz_data = response.choices[0].message.content
        self.task.upload_artifact("generated_quiz", quiz_data)

        # 步驟 6：記錄成本
        tokens_used = response.usage.total_tokens
        cost = self._calculate_cost(prompt_config["model"], tokens_used)
        self.task.get_logger().report_scalar(
            title="Cost", series="USD", value=cost, iteration=0
        )

        return quiz_data

    def _calculate_cost(self, model: str, tokens: int) -> float:
        """計算 Token 成本"""
        pricing = {
            "gpt-4o": 0.005 / 1000,  # $5 per 1M tokens
            "gpt-4o-mini": 0.00015 / 1000
        }
        return tokens * pricing.get(model, 0.002 / 1000)

# 執行管線
if __name__ == "__main__":
    pipeline = UnifiedMLLMPipeline()
    quiz = pipeline.generate_quiz(topic="Python 迴圈", num_questions=5)
    print(quiz)
```

---

### Phase 3：生產級增強 (Week 7-12)

#### 步驟 3.1：加入防護欄 (Guardrails)

```python
# guardrails.py
from nemoguardrails import RailsConfig, LLMRails

class LLMGuardrails:
    """LLM 輸出防護欄"""

    def __init__(self):
        # 配置防護欄規則
        config = RailsConfig.from_content(
            yaml_content="""
            define user ask harmful question:
              "如何入侵系統"
              "如何製作炸彈"

            define bot refuse harmful:
              "抱歉，我無法回答有害或危險的問題。"

            define flow
              user ask harmful question
              bot refuse harmful
              stop
            """,
            colang_content=""
        )
        self.rails = LLMRails(config)

    def check_and_generate(self, prompt: str):
        """檢查 + 生成（帶防護欄）"""
        response = self.rails.generate(messages=[{
            "role": "user",
            "content": prompt
        }])
        return response["content"]
```

#### 步驟 3.2：成本優化 - 語意快取

```python
# semantic_cache.py
import hashlib
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import openai

class SemanticCache:
    """語意快取（節省 LLM 成本）"""

    def __init__(self, similarity_threshold=0.95):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection = "llm_cache"
        self.threshold = similarity_threshold
        self._init_collection()

    def _init_collection(self):
        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        except Exception:
            pass

    def get(self, prompt: str) -> Optional[str]:
        """從快取獲取（如果相似度高）"""
        embedding = self._get_embedding(prompt)

        results = self.client.search(
            collection_name=self.collection,
            query_vector=embedding,
            limit=1
        )

        if results and results[0].score >= self.threshold:
            print(f"✓ 快取命中！相似度: {results[0].score:.3f}")
            return results[0].payload["response"]

        return None

    def set(self, prompt: str, response: str):
        """儲存到快取"""
        embedding = self._get_embedding(prompt)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=prompt_hash,
                vector=embedding,
                payload={"prompt": prompt, "response": response}
            )]
        )

    def _get_embedding(self, text: str):
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

# 使用範例
cache = SemanticCache()

# 首次查詢（未命中，呼叫 LLM）
response = cache.get("什麼是 Python 迴圈？")
if response is None:
    response = openai.chat.completions.create(...)  # 實際呼叫
    cache.set("什麼是 Python 迴圈？", response)

# 第二次查詢（命中快取，節省成本）
response = cache.get("Python 迴圈是什麼？")  # 語意相似，命中！
```

---

## 5. 工具鏈選擇

### 5.1 推薦組合（按預算）

#### 方案 A：最小成本（API Only）
```yaml
Prompt 管理: Git + YAML
向量資料庫: Qdrant (自部署)
LLM: OpenAI API / Anthropic API
實驗追蹤: MLflow (開源)
監控: Prometheus + Grafana
成本追蹤: 自建 (記錄到 PostgreSQL)

月成本: ~$500-2K (主要是 LLM API)
```

#### 方案 B：平衡方案
```yaml
Prompt 管理: LangSmith ($99/月)
向量資料庫: Weaviate Cloud
LLM: 混合 (簡單用自部署 7B/13B, 複雜用 API)
實驗追蹤: ClearML (自部署)
監控: Datadog ($15/host/月)
Agent 框架: LangGraph
防護欄: NeMo Guardrails

月成本: ~$2K-5K
```

#### 方案 C：企業級
```yaml
Prompt 管理: LangSmith Team ($500/月)
向量資料庫: Pinecone / Weaviate Enterprise
LLM: 自部署 LLM (vLLM) + API 備援
實驗追蹤: ClearML Server (自部署)
監控: Datadog APM + Custom Dashboards
治理: Arize AI / WhyLabs
CI/CD: Argo Workflows
K8s: EKS / GKE with GPU nodes

月成本: $10K-50K (主要是 GPU)
```

### 5.2 技術棧對照表

| 能力 | 開源方案 | 商業方案 | 我們推薦 |
|------|---------|---------|---------|
| **Prompt 管理** | Git + YAML | LangSmith, PromptLayer | 初期用 Git，規模化後用 LangSmith |
| **向量資料庫** | Qdrant, Milvus | Pinecone, Weaviate Cloud | Qdrant (效能好+免費) |
| **LLM 推論** | vLLM, TGI | OpenAI, Anthropic API | **混合**：70% vLLM + 30% API |
| **Embedding** | sentence-transformers | OpenAI, Cohere | OpenAI (質量最好) |
| **Agent 框架** | LangGraph, AutoGen | LangChain Cloud | LangGraph (靈活+免費) |
| **防護欄** | NeMo Guardrails | Azure Content Safety | NeMo (可自定義規則) |
| **實驗追蹤** | MLflow | ClearML, W&B | ClearML (功能完整) |
| **監控** | Prometheus | Datadog, New Relic | 初期 Prometheus，後期 Datadog |

---

## 6. 監控與治理

### 6.1 LLMOps 特有的監控指標

#### 質量指標 (Quality Metrics)

```python
# llm_metrics.py
from typing import Dict, List
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

class LLMQualityMonitor:
    """LLM 輸出質量監控"""

    def evaluate_rag_response(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, float]:
        """評估 RAG 回答質量"""

        # 使用 RAGAs 框架
        result = evaluate(
            data={
                "question": [question],
                "answer": [answer],
                "contexts": [contexts]
            },
            metrics=[faithfulness, answer_relevancy]
        )

        return {
            "faithfulness": result["faithfulness"],      # 忠實度（不幻覺）
            "answer_relevancy": result["answer_relevancy"]  # 相關性
        }

    def detect_hallucination(self, response: str, context: str) -> bool:
        """檢測幻覺（簡化版）"""
        # 使用 NLI 模型檢查 response 是否能從 context 推導
        # 實際實作需要載入 BERT-based NLI 模型
        pass
```

#### 成本指標 (Cost Metrics)

```python
# cost_tracking.py
from datetime import datetime
from typing import Dict
import psycopg2

class LLMCostTracker:
    """LLM 成本追蹤"""

    def __init__(self, db_connection_string):
        self.conn = psycopg2.connect(db_connection_string)
        self._init_table()

    def _init_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS llm_cost_log (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    project TEXT,
                    model TEXT,
                    prompt_tokens INT,
                    completion_tokens INT,
                    total_tokens INT,
                    cost_usd DECIMAL(10, 6),
                    latency_ms INT,
                    user_id TEXT
                )
            """)
            self.conn.commit()

    def log_request(self, data: Dict):
        """記錄每次 LLM 請求"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO llm_cost_log
                (timestamp, project, model, prompt_tokens, completion_tokens,
                 total_tokens, cost_usd, latency_ms, user_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                datetime.now(),
                data["project"],
                data["model"],
                data["prompt_tokens"],
                data["completion_tokens"],
                data["total_tokens"],
                data["cost_usd"],
                data["latency_ms"],
                data.get("user_id", "anonymous")
            ))
            self.conn.commit()

    def get_daily_cost(self, project: str) -> float:
        """獲取每日成本"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT SUM(cost_usd)
                FROM llm_cost_log
                WHERE project = %s
                AND DATE(timestamp) = CURRENT_DATE
            """, (project,))
            result = cur.fetchone()
            return float(result[0] or 0)
```

#### 延遲指標 (Latency Metrics)

```python
# 使用 Prometheus 追蹤延遲
from prometheus_client import Histogram
import time

llm_latency = Histogram(
    'llm_request_duration_seconds',
    'LLM request latency',
    ['model', 'project']
)

@llm_latency.labels(model='gpt-4o', project='edu-quiz').time()
def generate_with_monitoring(prompt: str):
    # LLM 呼叫
    response = openai.chat.completions.create(...)
    return response
```

### 6.2 統一監控儀表板

使用 Grafana 整合 ML + LLM 指標：

```yaml
# grafana_dashboard.yaml
dashboard:
  title: "Unified ML + LLM Monitoring"
  panels:
    - title: "模型推論 QPS"
      targets:
        - expr: 'rate(ml_inference_total[5m])'      # 傳統 ML
        - expr: 'rate(llm_request_total[5m])'       # LLM

    - title: "推論延遲 (p95)"
      targets:
        - expr: 'histogram_quantile(0.95, ml_latency)'
        - expr: 'histogram_quantile(0.95, llm_latency)'

    - title: "每日成本"
      targets:
        - expr: 'sum(ml_gpu_cost_usd)'              # ML GPU 成本
        - expr: 'sum(llm_token_cost_usd)'           # LLM Token 成本

    - title: "LLM 質量指標"
      targets:
        - expr: 'avg(llm_faithfulness_score)'       # 忠實度
        - expr: 'avg(llm_relevancy_score)'          # 相關性
        - expr: 'rate(llm_guardrail_blocked[5m])'   # 防護欄攔截率
```

---

## 7. 實戰案例：教育出題系統

### 完整管線實作

```python
# edu_quiz_system.py
from clearml import Task
from rag_pipeline import RAGPipeline
from guardrails import LLMGuardrails
from semantic_cache import SemanticCache
from cost_tracking import LLMCostTracker
from llm_metrics import LLMQualityMonitor
import openai
import time

class EduQuizSystem:
    """教育出題系統 - 完整 LLMOps 管線"""

    def __init__(self):
        # 初始化組件
        self.task = Task.init(project_name="edu-quiz", task_name="quiz_generation")
        self.rag = RAGPipeline(collection_name="edu_materials")
        self.guardrails = LLMGuardrails()
        self.cache = SemanticCache()
        self.cost_tracker = LLMCostTracker("postgresql://localhost/llmops")
        self.quality_monitor = LLMQualityMonitor()

    def generate_quiz(self, topic: str, grade: str, num_questions: int = 5):
        """生成考題 - 完整流程"""
        start_time = time.time()

        # === 步驟 1：檢查快取 ===
        cache_key = f"{topic}_{grade}_{num_questions}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.task.get_logger().report_text("✓ 快取命中，跳過 LLM 呼叫")
            return cached_result

        # === 步驟 2：RAG 檢索 ===
        self.task.get_logger().report_text("步驟 2：RAG 檢索教材")
        contexts = self.rag.retrieve(
            query=f"{topic} {grade} 教材",
            top_k=3
        )
        context_text = "\n\n".join([c["content"] for c in contexts])

        # === 步驟 3：組裝 Prompt ===
        prompt = f"""請根據以下教材生成 {num_questions} 道適合{grade}的{topic}考題。

教材內容：
{context_text}

輸出格式：JSON，包含 question, options, answer, explanation"""

        # === 步驟 4：防護欄檢查 ===
        prompt = self.guardrails.check_and_generate(prompt)

        # === 步驟 5：呼叫 LLM ===
        self.task.get_logger().report_text("步驟 5：呼叫 LLM")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "你是出題專家"},
                {"role": "user", "content": prompt}
            ]
        )

        quiz_result = response.choices[0].message.content

        # === 步驟 6：質量評估 ===
        quality_scores = self.quality_monitor.evaluate_rag_response(
            question=f"生成{topic}的{num_questions}道題目",
            answer=quiz_result,
            contexts=[c["content"] for c in contexts]
        )

        self.task.get_logger().report_scalar(
            "faithfulness", "score", quality_scores["faithfulness"], 0
        )

        # === 步驟 7：成本追蹤 ===
        latency_ms = int((time.time() - start_time) * 1000)
        tokens = response.usage.total_tokens
        cost = tokens * 0.00015 / 1000  # GPT-4o-mini 定價

        self.cost_tracker.log_request({
            "project": "edu-quiz",
            "model": "gpt-4o-mini",
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": tokens,
            "cost_usd": cost,
            "latency_ms": latency_ms
        })

        # === 步驟 8：儲存快取 ===
        self.cache.set(cache_key, quiz_result)

        # === 步驟 9：記錄 Artifact ===
        self.task.upload_artifact("quiz_output", quiz_result)

        return quiz_result

# 執行
if __name__ == "__main__":
    system = EduQuizSystem()

    # 索引教材（一次性）
    system.rag.index_documents([
        {"content": "Python for 迴圈用於遍歷序列...", "metadata": {"grade": "高一"}},
        # ... 更多教材
    ])

    # 生成考題
    quiz = system.generate_quiz(
        topic="Python 迴圈",
        grade="高一",
        num_questions=5
    )

    print(quiz)
```

---

## 8. 總結與下一步

### 8.1 整合檢查清單

- [ ] **Phase 1 完成** (Week 1-2)
  - [ ] Prompt 版本控制（Git + YAML）
  - [ ] MLflow/ClearML 支援 Prompt 記錄
  - [ ] 向量資料庫部署（Qdrant）

- [ ] **Phase 2 完成** (Week 3-6)
  - [ ] RAG 管線可運行
  - [ ] 統一推論閘道（ML + LLM 路由）
  - [ ] 成本追蹤系統
  - [ ] 語意快取上線

- [ ] **Phase 3 完成** (Week 7-12)
  - [ ] 防護欄整合
  - [ ] Agent 編排框架
  - [ ] 統一監控儀表板
  - [ ] A/B 測試框架（ML + LLM）
  - [ ] 首個生產用例上線

### 8.2 關鍵成功指標

| 指標 | 目標 | 當前 | 備註 |
|------|------|------|------|
| **LLM 快取命中率** | >40% | 0% | 教育場景重複查詢多 |
| **平均 Token 成本** | <$0.0008/1K | - | 透過快取+壓縮降低 |
| **RAG 檢索召回率** | >0.90 | - | nDCG@10 指標 |
| **答案忠實度** | >95% | - | 防止幻覺 |
| **p95 延遲** | <2s | - | 首 Token <800ms |
| **防護欄攔截率** | >0% | - | 確保安全 |

### 8.3 常見陷阱與建議

#### ❌ 陷阱 1：過早優化
**問題**：一開始就自部署 LLM，結果維運成本高於 API
**建議**：先用 API 驗證，QPS >100 再考慮自部署

#### ❌ 陷阱 2：忽略成本追蹤
**問題**：LLM Token 成本暴增，不知道哪個功能在燒錢
**建議**：**從第一天就記錄每次請求的成本**

#### ❌ 陷阱 3：沒有防護欄
**問題**：LLM 輸出有害內容、洩露隱私
**建議**：生產環境必須有輸入/輸出防護欄

#### ❌ 陷阱 4：RAG 檢索質量差
**問題**：檢索到無關文檔，LLM 生成錯誤答案
**建議**：使用 Reranker + 定期評估檢索質量

#### ✅ 最佳實踐

1. **統一追蹤**：ML 和 LLM 都記錄到同一個實驗追蹤系統
2. **成本可歸因**：每次推論都記錄 project/user，方便成本分攤
3. **快取優先**：語意快取可節省 80% 成本（教育/客服場景）
4. **漸進式遷移**：不要一次性重寫，先在非關鍵路徑試 LLM
5. **質量監控**：LLM 輸出質量需要持續監控（不像 ML 有固定指標）

---

## 附錄：參考資源

### 開源專案
- **LangChain**：https://github.com/langchain-ai/langchain
- **vLLM**：https://github.com/vllm-project/vllm
- **Qdrant**：https://github.com/qdrant/qdrant
- **NeMo Guardrails**：https://github.com/NVIDIA/NeMo-Guardrails

### 學習資源
- **LLMOps 最佳實踐**：https://huyenchip.com/2023/04/11/llm-engineering.html
- **Prompt Engineering Guide**：https://www.promptingguide.ai/
- **RAG 評估框架**：https://github.com/explodinggradients/ragas

---

**最後更新**：2025-10-15
**版本**：1.0
