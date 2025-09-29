"""
Feedback Analytics Page

This Streamlit multipage renders both session analytics and persistent analytics
for chatbot feedback, with export and clear controls.
"""

import os
from datetime import datetime

import streamlit as st
from chatbot.feedback_manager import FeedbackManager
from dotenv import load_dotenv

load_dotenv()


def main():
    st.set_page_config(page_title="Feedback Analytics", layout="wide")
    st.title("Feedback Analytics")
    st.caption("Analyze feedback for the current session and across sessions.")

    feedback_manager = FeedbackManager()

    # Session analytics
    st.header("Session Feedback")
    messages = st.session_state.get("messages", [])
    session_stats = feedback_manager.compute_session_feedback(messages, recent_limit=10)

    if session_stats.get("total_responses", 0) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Responses", session_stats["total_responses"])
        c2.metric("👍 Positive", session_stats["positive_feedback"])
        c3.metric("👎 Negative", session_stats["negative_feedback"])
        c4.metric("No Feedback", session_stats["no_feedback"])

        st.metric("Satisfaction Rate", f"{session_stats['satisfaction_rate']:.1f}%")

        if session_stats.get("recent"):
            st.subheader("Recent Feedback (Session)")
            for item in session_stats["recent"]:
                icon = "⏳"
                if item.get("feedback") == "positive":
                    icon = "👍"
                elif item.get("feedback") == "negative":
                    icon = "👎"
                st.text(f"{icon} {item.get('content_preview', '')}")
    else:
        st.info("No session responses yet.")

    st.divider()

    # Persistent analytics
    st.header("Persistent Feedback")
    stats = feedback_manager.get_feedback_stats()

    if stats.get("total_feedback", 0) > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Feedback Entries", stats["total_feedback"])
        c2.metric("Satisfaction", f"{stats['satisfaction_rate']}%")
        c3.metric("Avg Response Length", stats["average_response_length"])

        c1, c2 = st.columns(2)
        c1.metric("👍 Positive", stats["positive_feedback"])
        c2.metric("👎 Negative", stats["negative_feedback"])

        st.subheader("Manage Feedback Data")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export Feedback Data"):
                export_file = (
                    f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                if feedback_manager.export_feedback(export_file):
                    st.success(f"Feedback exported to {export_file}")
                else:
                    st.error("Failed to export feedback data")
        with col2:
            if st.button("🗑️ Clear All Feedback", type="secondary"):
                if feedback_manager.clear_feedback():
                    st.success("All feedback data cleared")
                    st.rerun()
                else:
                    st.error("Failed to clear feedback data")
    else:
        st.info("No persistent feedback data yet.")


if __name__ == "__main__":
    main()
