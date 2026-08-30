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


def _store_documents(client: PaperRAGAPIClient, base_url: str) -> list[dict[str, Any]]:
    payload = client.list_documents()
    documents = payload.get("documents", [])
    if not isinstance(documents, list):
        raise PaperRAGAPIError("论文列表格式不正确。")
    st.session_state["indexed_documents"] = documents
    st.session_state["documents_base_url"] = base_url
    return documents


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
                f"chunk_id={hit.get('chunk_id')} · "
                f"document_id={hit.get('document_id')}"
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
                _store_documents(client, base_url)
                st.success(f"API 正常 · v{health.get('version')}")
            except (PaperRAGAPIError, ValueError) as exc:
                st.error(str(exc))

        st.divider()
        st.header("论文库")
        if st.session_state.get("documents_base_url") != base_url:
            st.session_state["indexed_documents"] = []
        if st.button("加载 / 刷新论文列表", use_container_width=True):
            try:
                _store_documents(client, base_url)
            except (PaperRAGAPIError, ValueError) as exc:
                st.error(str(exc))
        documents = st.session_state.get("indexed_documents", [])
        document_lookup = {
            str(document.get("document_id")): document for document in documents
        }
        document_options = [""] + list(document_lookup)

        def document_label(document_id: str) -> str:
            if not document_id:
                return "全部已索引论文"
            document = document_lookup[document_id]
            return f"{document.get('title', 'Untitled')} · {document_id[:8]}"

        selected_document_id = st.selectbox(
            "检索范围",
            options=document_options,
            format_func=document_label,
        )
        document_id = selected_document_id or None
        if documents:
            with st.expander(f"已索引 {len(documents)} 篇论文"):
                for document in documents:
                    st.markdown(f"**{document.get('title', 'Untitled')}**")
                    st.caption(
                        f"{document.get('source_file')} · "
                        f"{document.get('chunk_count')} chunks · "
                        f"{document.get('indexed_pages')} 页 · "
                        f"ID: {document.get('document_id')}"
                    )
        else:
            st.caption("点击上方按钮读取已建立索引的论文。")

        st.divider()
        st.header("检索设置")
        strategy_label = st.selectbox("检索策略", options=list(STRATEGIES))
        strategy = STRATEGIES[strategy_label]
        if strategy == "decomposed_hybrid_rerank":
            st.caption("识别问题中的论文名称，分论文检索后合并证据。")
        top_k = st.slider("返回证据数 Top-K", min_value=1, max_value=20, value=5)
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
        if success_message := st.session_state.pop("index_success", None):
            st.success(success_message)
        if refresh_error := st.session_state.pop("document_refresh_error", None):
            st.warning(refresh_error)
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
                st.session_state["index_success"] = (
                    f"索引完成：{report.get('indexed_chunks')} 个 chunks，"
                    f"document_id = {report.get('document_id')}"
                )
                try:
                    _store_documents(client, base_url)
                except (PaperRAGAPIError, ValueError) as exc:
                    st.session_state["document_refresh_error"] = (
                        f"索引成功，但刷新论文列表失败：{exc}"
                    )
                st.rerun()
            except (PaperRAGAPIError, ValueError) as exc:
                st.error(str(exc))

    with query_tab:
        st.subheader("向论文提问")
        if document_id:
            st.info(f"当前只检索：{document_label(document_id)}")
        else:
            st.info("当前检索全部已索引论文。")
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
