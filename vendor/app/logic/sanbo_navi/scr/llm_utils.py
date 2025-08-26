# logic
from logic.sanbo_navi.scr.prompts import (
    build_category_suggestion_prompt,
    build_answer_prompt,
)

# --- GPT/LLM 呼び出し クラス版 ---
from abc import ABC, abstractmethod


# --- 抽象基底クラス ---
class LLMClientBase(ABC):
    def __init__(self, client):
        self.client = client

    @abstractmethod
    def call(self, system_prompt, user_prompt):
        pass


# --- OpenAI 用クラス ---
class OpenAIClient(LLMClientBase):
    def __init__(self, client, model_name="gpt-4o"):
        super().__init__(client)
        self.model_name = model_name

    def call(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()


# --- Anthropic 用クラス ---
class AnthropicClient(LLMClientBase):
    def __init__(self, client, model_name="claude-3-opus-20240229"):
        super().__init__(client)
        self.model_name = model_name

    def call(self, system_prompt, user_prompt):
        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content.strip()


# --- Huggingface 用クラス ---
class HuggingfaceClient(LLMClientBase):
    def call(self, system_prompt, user_prompt):
        result = self.client(
            f"{system_prompt}\n\n{user_prompt}", max_length=512, do_sample=True
        )
        return result[0]["generated_text"].strip()


# --- カテゴリ推定 ---
def suggest_category(query_input, llm_client: LLMClientBase):
    prompt = build_category_suggestion_prompt(query_input)
    system_prompt = (
        "あなたは日本の産業廃棄物処理のカテゴリ分類に詳しい専門アシスタントです。"
    )
    return llm_client.call(system_prompt, prompt)


# --- 回答生成 ---
def generate_answer(query, selected_category, vectorstore, llm_client: LLMClientBase):
    docs = vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=30)

    if selected_category:
        docs = [
            doc for doc in docs if selected_category in doc.metadata.get("category", [])
        ]

    context = "\n".join([doc.page_content for doc in docs])
    sources = [
        (doc.metadata.get("source", "不明"), doc.metadata.get("page", "不明"))
        for doc in docs
    ]

    prompt = build_answer_prompt(context, query)
    system_prompt = "あなたは日本の産業廃棄物処理文書をもとに回答する専門AIです。"

    answer = llm_client.call(system_prompt, prompt)
    return answer, sources
