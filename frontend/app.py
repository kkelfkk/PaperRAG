"""Streamlit interface for indexing, retrieval, and grounded paper Q&A."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.api_client import DEFAULT_API_URL, PaperRAGAPIClient, PaperRAGAPIError

STRATEGIES = {
    "Hybrid（推荐）": "hybrid",
    "Dense 向量检索": "dense",
    "BM25 关键词检索": "bm25",
    "Hybrid + Reranker": "hybrid_rerank",
    "多论文分解 + Hybrid + Reranker（跨论文推荐）": (
        "decomposed_hybrid_rerank"
    ),
}


@st.cache_resource
def get_client(base_url: str) -> PaperRAGAPIClient:
    return PaperRAGAPIClient(base_url)


def render_hits(payload: dict[str, Any]) -> None:
    hits = payload.get("hits", [])
    if not hits:
        st.warning("没有找到符合条件的证据。可以放宽过滤条件或换一种问法。")
        return
    st.caption(
        f"返回 {len(hits)} 条证据 · {payload.get('embedding_model', 'unknown')}"
    )
    for hit in hits:
        section = hit.get("section") or "未识别章节"
        label = (
            f"#{hit.get('rank')}  {hit.get('title', 'Untitled')} · "
            f"第 {hit.get('page_number')} 页 · {section}"
        )
        with st.expander(label, expanded=hit.get("rank") == 1):
            st.caption(
                f"score={float(hit.get('score', 0)):.4f} · "
                f"chunk_id={hit.get('chunk_id')}"
            )
            st.write(hit.get("text", ""))


def render_answer(payload: dict[str, Any]) -> None:
    if payload.get("abstained"):
        st.warning(payload.get("answer") or "现有证据不足，系统选择不回答。")
    else:
        st.markdown(payload.get("answer", ""))
    st.caption(
        f"模型：{payload.get('model', 'unknown')} · "
        f"使用证据：{payload.get('retrieved_count', 0)} 条"
    )
    citations = payload.get("citations", [])
    if citations:
        st.subheader("引用来源")
        for citation in citations:
            section = citation.get("section") or "未识别章节"
            st.markdown(
                f"- **[{citation.get('source_id')}] {citation.get('title')}** — "
                f"第 {citation.get('page_number')} 页，{section}"
            )


def main() -> None:
    st.set_page_config(page_title="PaperRAG", page_icon="📚", layout="wide")
    st.title("📚 PaperRAG")
    st.caption("面向学术论文的混合检索、重排与可验证引用问答")

    with st.sidebar:
        st.header("连接设置")
        base_url = st.text_input("FastAPI 地址", value=DEFAULT_API_URL)
        try:
            client = get_client(base_url)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        if st.button("检查 API", use_container_width=True):
            try:
                health = client.health()
                st.success(f"API 正常 · v{health.get('version')}")
            except (PaperRAGAPIError, ValueError) as exc:
                st.error(str(exc))

        st.divider()
        st.header("检索设置")
        strategy_label = st.selectbox("检索策略", options=list(STRATEGIES))
        strategy = STRATEGIES[strategy_label]
        if strategy == "decomposed_hybrid_rerank":
            st.caption("识别问题中的论文名称，分论文检索后合并证据。")
        top_k = st.slider("返回证据数 Top-K", min_value=1, max_value=20, value=5)
        document_id = st.text_input("限定 document_id（可选）").strip() or None
        section = st.text_input("限定章节（可选，精确匹配）").strip() or None
        use_page_range = st.checkbox("限定页码范围")
        page_from: int | None = None
        page_to: int | None = None
        page_range_valid = True
        if use_page_range:
            page_columns = st.columns(2)
            page_from = int(
                page_columns[0].number_input("起始页", min_value=1, value=1)
            )
            page_to = int(
                page_columns[1].number_input(
                    "结束页",
                    min_value=1,
                    value=10,
                )
            )
            page_range_valid = page_from <= page_to
            if not page_range_valid:
                st.error("起始页不能大于结束页。")

    index_tab, query_tab = st.tabs(["上传论文", "检索与问答"])

    with index_tab:
        st.subheader("上传并建立索引")
        uploaded = st.file_uploader("选择文本型 PDF", type=["pdf"])
        settings = st.expander("切块参数")
        with settings:
            max_chars = st.number_input(
                "最大 chunk 字符数", min_value=64, max_value=20000, value=1200
            )
            overlap = st.number_input(
                "重叠字符数", min_value=0, max_value=10000, value=200
            )
            recreate = st.checkbox("清空现有索引后再导入")
        if st.button("开始索引", type="primary", disabled=uploaded is None):
            assert uploaded is not None
            try:
                with st.spinner("正在解析、切块并生成向量……"):
                    report = client.index_pdf(
                        uploaded.name,
                        uploaded.getvalue(),
                        recreate=recreate,
                        max_chunk_chars=int(max_chars),
                        overlap_chars=int(overlap),
                    )
                st.session_state["last_document_id"] = report.get("document_id")
                st.success(
                    f"索引完成：{report.get('indexed_chunks')} 个 chunks，"
                    f"document_id = {report.get('document_id')}"
                )
                st.json(report)
            except (PaperRAGAPIError, ValueError) as exc:
                st.error(str(exc))

    with query_tab:
        st.subheader("向论文提问")
        last_document = st.session_state.get("last_document_id")
        if last_document and not document_id:
            st.info(
                f"刚刚索引的 document_id：{last_document}。"
                "如需只搜索该论文，请复制到侧边栏。"
            )
        mode = st.radio("运行模式", ["生成带引用回答", "只查看检索证据"], horizontal=True)
        query = st.text_area(
            "问题",
            placeholder="例如：论文提出的主要方法是什么？它解决了什么问题？",
            height=100,
        )
        if st.button(
            "运行 PaperRAG",
            type="primary",
            disabled=not query.strip() or not page_range_valid,
        ):
            common = {
                "top_k": top_k,
                "strategy": strategy,
                "document_id": document_id,
                "section": section,
                "page_from": page_from,
                "page_to": page_to,
            }
            try:
                with st.spinner("正在检索证据……"):
                    if mode == "只查看检索证据":
                        result = client.search(query, **common)
                    else:
                        result = client.ask(query, **common)
                if mode == "只查看检索证据":
                    render_hits(result)
                else:
                    render_answer(result)
            except (PaperRAGAPIError, ValueError) as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
