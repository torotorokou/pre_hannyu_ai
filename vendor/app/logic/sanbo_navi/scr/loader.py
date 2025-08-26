import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import yaml
import json


def get_resource_paths() -> dict:
    return {
        "PDF_PATH": "/work/app/logic/sanbo_navi/local_data/master/SOLVEST.pdf",
        "JSON_PATH": "/work/app/logic/sanbo_navi/local_data/master/structured_SOLVEST_output_final.json",
        "FAISS_PATH": "/work/app/logic/sanbo_navi/local_data/master/vectorstore/solvest_faiss_corrected",
        "ENV_PATH": "/work/app/logic/sanbo_navi/config/.env",
        "YAML_PATH": "/work/app/logic/sanbo_navi/local_data/master/category_question_templates.yaml",
    }


# --- 設定ファイルの読み込み ---
def load_config():
    FAISS_PATH = get_resource_paths().get("FAISS_PATH")
    PDF_PATH = get_resource_paths().get("PDF_PATH")
    JSON_PATH = get_resource_paths().get("JSON_PATH")
    return FAISS_PATH, PDF_PATH, JSON_PATH


@st.cache_data
def load_question_templates():
    yaml_path = get_resource_paths().get("YAML_PATH")
    with open(yaml_path, encoding="utf-8") as f:
        templates = yaml.safe_load(f)
    return templates


# --- JSONからカテゴリ・サブカテゴリを取得 ---
@st.cache_data
def load_json_data(json_path):
    # JSONファイルを読み込んでデータとして返す
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


@st.cache_data
def extract_categories_and_titles(data):
    # JSONからカテゴリとサブカテゴリを抽出
    categories = set()
    subcategories = {}
    for section in data:
        cats = section.get("category", [])
        if isinstance(cats, str):
            cats = [cats]
        for cat in cats:
            categories.add(cat)
            subcategories.setdefault(cat, set()).add(section.get("title"))
    categories = sorted(categories)
    for k in subcategories:
        subcategories[k] = sorted(subcategories[k])
    return categories, subcategories
