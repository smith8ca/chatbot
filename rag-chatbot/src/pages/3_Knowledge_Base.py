"""
Knowledge Base Page

Lists the documents currently stored in the Chroma collection with basic metadata
and provides management actions.
"""

import os

import streamlit as st
from chatbot.chromadb_client import ChromaDBClient
from dotenv import load_dotenv

load_dotenv()


def main():
    st.set_page_config(page_title="Knowledge Base", layout="wide")
    st.title("Knowledge Base")
    st.caption("Documents currently stored in the vector database")

    try:
        db_client = ChromaDBClient(
            collection_name=os.getenv("CHROMA_COLLECTION", "rag_documents"),
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        )

        # Controls
        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            limit = st.number_input(
                "Limit", min_value=1, max_value=500, value=100, step=10
            )
        with col_b:
            offset = st.number_input("Offset", min_value=0, value=0, step=10)

        st.divider()

        # Bulk actions
        with st.expander("Bulk Actions"):
            filename_for_delete = st.text_input("Filename to delete all versions")
            if st.button("Delete All Versions for Filename") and filename_for_delete:
                deleted_count = db_client.delete_by_filename(filename_for_delete)
                if deleted_count > 0:
                    st.success(
                        f"Requested deletion of {deleted_count} entries for '{filename_for_delete}'."
                    )
                    st.rerun()
                else:
                    st.warning(
                        "No entries deleted (filename may not exist or delete not supported)."
                    )

        docs = db_client.list_documents(limit=int(limit), offset=int(offset))
        if not docs:
            st.info("No documents found in the knowledge base.")
            return

        # Table-like view
        for item in docs:
            display_id = item.get("id") or "(unknown id)"
            with st.expander(f"ID: {display_id}"):
                meta = item.get("metadata", {})
                filename = meta.get("filename", "(unknown)")
                file_type = meta.get("file_type", "")
                length = meta.get("text_length", 0)
                stored_at = meta.get("stored_at", "")
                processed_at = meta.get("processed_at", "")

                c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
                c1.write(f"**Filename**: {filename}")
                c2.write(f"**Type**: {file_type}")
                c3.write(f"**Chars**: {length}")
                c4.write(f"**Processed**: {processed_at}")

                st.caption(f"Stored At: {stored_at}")

                with st.expander("Preview (first 500 chars)"):
                    preview = (item.get("document") or "")[:500]
                    if preview:
                        st.code(preview)
                    else:
                        st.text("(no content)")

                # Management actions
                if item.get("id"):
                    col_del, col_sp = st.columns([1, 5])
                    with col_del:
                        if st.button("🗑️ Delete", key=f"del_{item['id']}"):
                            ok = db_client.delete_document(item["id"])
                            if ok:
                                st.success("Document deleted")
                                st.rerun()
                            else:
                                st.error("Failed to delete document")
                else:
                    st.warning(
                        "Cannot delete: document ID unavailable in this listing."
                    )

    except Exception as e:
        st.error(f"Failed to load knowledge base: {e}")


if __name__ == "__main__":
    main()
