# app.py

import streamlit as st
from sentence_transformers import SentenceTransformer, util

# 初始化模型
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# 法律情境知識庫（擴充後）
law_knowledge_base = [
    {
        "情境說明": "公司無預警資遣員工",
        "對應法條": ["勞基法第16條", "勞基法第20條"],
        "建議": "若遭遇無預警資遣，雇主可能違反《勞基法》，可向勞工局申訴。"
    },
    {
    "情境說明": "公司在高溫環境中不開冷氣，辦公室悶熱，導致員工頭暈、無法集中精神甚至中暑",
    "對應法條": ["職業安全衛生法第6條", "職業安全衛生法第13條", "勞基法第22條"],
    "建議": "雇主應提供適當通風、冷卻或溫度控制設備，確保勞工健康。可記錄環境溫度與員工不適狀況，向勞檢單位或勞工局申訴。"
    },
    {
        "情境說明": "公司長期未給予加班費",
        "對應法條": ["勞基法第24條", "勞基法第32條"],
        "建議": "雇主應依法給付加班費，否則屬違法行為，可提報勞工局。"
    },
    {
        "情境說明": "公司未提供勞健保",
        "對應法條": ["勞工保險條例", "全民健康保險法"],
        "建議": "雇主依法應為員工投保勞健保，可向勞保局或健保署檢舉。"
    },
    {
        "情境說明": "雇主要求勞工從事原職務以外的清潔打掃工作",
        "對應法條": ["勞基法第10條", "勞基法第11條"],
        "建議": "若勞動契約中未約定清潔職務，雇主不得任意指派與原職務無關的工作。建議保留書面紀錄並向勞工局諮詢。"
    },
    {
        "情境說明": "雇主要求勞工負責接送其家人，處理非工作相關私事",
        "對應法條": ["民法", "勞基法第10條"],
        "建議": "若勞工未同意且工作內容與私人事務無關，雇主不得要求執行。建議主張工作界限並保留證據。"
    }
]

# 預先編碼情境
情境敘述們 = [項["情境說明"] for 項 in law_knowledge_base]
情境向量們 = model.encode(情境敘述們)

# Streamlit 介面
st.title("⚖️ AI法律情境諮詢平台")
st.markdown("請描述你遇到的問題，我們會試著分析是否涉及勞動法規並提供建議。")

user_input = st.text_area("📨 請輸入你的情境描述：", height=150)
if st.button("🔍 開始分析") and user_input:
    user_vector = model.encode(user_input)
    similarities = util.cos_sim(user_vector, 情境向量們)[0]
    top_indices = similarities.argsort(descending=True)[:3]

    st.subheader("📊 分析結果 (前三相近情境)")
    for idx in top_indices:
        match = law_knowledge_base[idx]
        score = float(similarities[idx])
        st.markdown(f"### 💡 相似情境 #{top_indices.tolist().index(idx)+1}")
        st.write(f"🔎 **判斷情境**：{match['情境說明']}")
        st.write(f"📜 **可能違反法條**：{', '.join(match['對應法條'])}")
        st.write(f"📌 **建議行動**：{match['建議']}")
        st.write(f"📐 **語意相似度分數**：{score:.2f}")
        st.markdown("---")

    st.info("📘【免責聲明】本系統所提供之內容僅供參考，非正式法律意見。建議您洽詢專業律師或當地勞工主管機關獲取具體協助。")

# 使用者回報新情境功能
st.markdown("---")
st.subheader("📩 回報你遇到的新狀況")
new_case = st.text_area("🆕 請填寫你想補充的法律情境（可匿名）")
if st.button("📤 提交回報") and new_case:
    with open("user_reports.txt", "a", encoding="utf-8") as f:
        f.write(new_case + "\n---\n")
    st.success("✅ 已收到你的回報，我們將持續擴充系統內容以協助更多人。")