"""
Feedback Manager for RAG Chatbot

This module handles storing and managing user feedback on chatbot responses.
It provides functionality to save feedback data, generate analytics, and export
feedback for analysis.

Features:
    - Store feedback with timestamps
    - Generate feedback statistics
    - Export feedback data
    - Track response quality over time

Author: Charles A. Smith
Version: 1.0.0
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FeedbackManager:
    """Manages user feedback on chatbot responses using SQLite or JSON."""

    def __init__(
        self,
        storage_backend: Optional[str] = None,
        db_path: Optional[str] = None,
        json_path: Optional[str] = None,
    ):
        """
        Initialize the feedback manager.

        Args:
            storage_backend: 'sqlite' (default) or 'json'. If None, read FEEDBACK_BACKEND env var.
            db_path: Path to the SQLite database file. If None, FEEDBACK_DB_PATH or 'feedback_data.sqlite3'.
            json_path: Path to the JSON file. If None, FEEDBACK_JSON_PATH or 'feedback_data.json'.
        """
        # Determine backend and paths
        backend_env = os.getenv("FEEDBACK_BACKEND", "sqlite").strip().lower()
        self.backend = (storage_backend or backend_env) if backend_env in {"sqlite", "json"} else "sqlite"

        self.db_path = Path(db_path or os.getenv("FEEDBACK_DB_PATH", "feedback_data.sqlite3"))
        self.json_path = Path(json_path or os.getenv("FEEDBACK_JSON_PATH", "feedback_data.json"))

        if self.backend == "sqlite":
            self._ensure_database()
            # Migrate any legacy JSON if present and table empty
            self._maybe_migrate_from_json()
            logger.info(f"FeedbackManager initialized with SQLite database: {self.db_path}")
        else:
            # Ensure JSON file exists
            if not self.json_path.exists():
                try:
                    with open(self.json_path, "w", encoding="utf-8") as f:
                        json.dump([], f)
                except Exception as e:
                    logger.error(f"Error creating JSON feedback file: {e}")
            logger.info(f"FeedbackManager initialized with JSON file: {self.json_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_database(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message_id TEXT,
                        timestamp TEXT,
                        user_query TEXT,
                        response TEXT,
                        feedback TEXT CHECK(feedback IN ('positive','negative')),
                        response_length INTEGER,
                        query_length INTEGER
                    )
                    """
                )
        except Exception as e:
            logger.error(f"Error ensuring database: {e}")

    def _table_is_empty(self) -> bool:
        try:
            with self._connect() as conn:
                cur = conn.execute("SELECT COUNT(1) AS c FROM feedback")
                row = cur.fetchone()
                return (row["c"] if row else 0) == 0
        except Exception as e:
            logger.error(f"Error checking table emptiness: {e}")
            return True

    def _maybe_migrate_from_json(self) -> None:
        try:
            if not self.json_path.exists():
                return
            if not self._table_is_empty():
                return

            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list) or not data:
                    return

            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT INTO feedback (message_id, timestamp, user_query, response, feedback, response_length, query_length)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            entry.get("message_id"),
                            entry.get("timestamp") or datetime.now().isoformat(),
                            entry.get("user_query", ""),
                            entry.get("response", ""),
                            entry.get("feedback"),
                            int(entry.get("response_length", len(entry.get("response", "")))),
                            int(entry.get("query_length", len(entry.get("user_query", "")))),
                        )
                        for entry in data
                    ],
                )

            logger.info(f"Migrated {len(data)} legacy feedback entries from JSON to SQLite")
        except Exception as e:
            logger.error(f"Error migrating from JSON: {e}")

    def add_feedback(
        self,
        message_id: str,
        user_query: str,
        response: str,
        feedback: str,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Add feedback for a chatbot response.

        Args:
            message_id: Unique identifier for the message
            user_query: The user's original question
            response: The chatbot's response
            feedback: 'positive' or 'negative'
            timestamp: When the feedback was given (defaults to now)

        Returns:
            True if feedback was saved successfully, False otherwise
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()

            if self.backend == "sqlite":
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO feedback (message_id, timestamp, user_query, response, feedback, response_length, query_length)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            message_id,
                            timestamp.isoformat(),
                            user_query,
                            response,
                            feedback,
                            len(response),
                            len(user_query),
                        ),
                    )
            else:
                # JSON backend
                entry = {
                    "message_id": message_id,
                    "timestamp": timestamp.isoformat(),
                    "user_query": user_query,
                    "response": response,
                    "feedback": feedback,
                    "response_length": len(response),
                    "query_length": len(user_query),
                }
                existing = self._json_load()
                existing.append(entry)
                self._json_save(existing)

            logger.info(f"Added {feedback} feedback for message {message_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return False

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive feedback statistics.

        Returns:
            Dictionary containing various feedback metrics
        """
        try:
            if self.backend == "sqlite":
                with self._connect() as conn:
                    cur = conn.execute(
                        "SELECT COUNT(1) AS total FROM feedback"
                    )
                    total = cur.fetchone()["total"]

                    if total == 0:
                        return {
                            "total_feedback": 0,
                            "positive_feedback": 0,
                            "negative_feedback": 0,
                            "satisfaction_rate": 0.0,
                            "average_response_length": 0,
                            "average_query_length": 0,
                        }

                    pos = conn.execute(
                        "SELECT COUNT(1) AS c FROM feedback WHERE feedback='positive'"
                    ).fetchone()["c"]
                    neg = conn.execute(
                        "SELECT COUNT(1) AS c FROM feedback WHERE feedback='negative'"
                    ).fetchone()["c"]

                    satisfaction_rate = (
                        (pos / (pos + neg)) * 100 if (pos + neg) > 0 else 0.0
                    )

                    avg_resp = conn.execute(
                        "SELECT AVG(response_length) AS a FROM feedback"
                    ).fetchone()["a"]
                    avg_query = conn.execute(
                        "SELECT AVG(query_length) AS a FROM feedback"
                    ).fetchone()["a"]

                    return {
                        "total_feedback": int(total),
                        "positive_feedback": int(pos),
                        "negative_feedback": int(neg),
                        "satisfaction_rate": round(float(satisfaction_rate), 1),
                        "average_response_length": round(float(avg_resp), 1) if avg_resp is not None else 0,
                        "average_query_length": round(float(avg_query), 1) if avg_query is not None else 0,
                    }
            else:
                data = self._json_load()
                if not data:
                    return {
                        "total_feedback": 0,
                        "positive_feedback": 0,
                        "negative_feedback": 0,
                        "satisfaction_rate": 0.0,
                        "average_response_length": 0,
                        "average_query_length": 0,
                    }

                total = len(data)
                pos = len([f for f in data if f.get("feedback") == "positive"])
                neg = len([f for f in data if f.get("feedback") == "negative"])
                satisfaction_rate = (pos / (pos + neg) * 100) if (pos + neg) > 0 else 0.0
                avg_resp = sum(f.get("response_length", 0) for f in data) / total
                avg_query = sum(f.get("query_length", 0) for f in data) / total
                return {
                    "total_feedback": total,
                    "positive_feedback": pos,
                    "negative_feedback": neg,
                    "satisfaction_rate": round(satisfaction_rate, 1),
                    "average_response_length": round(avg_resp, 1),
                    "average_query_length": round(avg_query, 1),
                }
        except Exception as e:
            logger.error(f"Error calculating feedback stats: {e}")
            return {}

    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent feedback entries.

        Args:
            limit: Maximum number of recent entries to return

        Returns:
            List of recent feedback entries
        """
        try:
            if self.backend == "sqlite":
                with self._connect() as conn:
                    cur = conn.execute(
                        """
                        SELECT message_id, timestamp, user_query, response, feedback, response_length, query_length
                        FROM feedback
                        ORDER BY datetime(timestamp) DESC
                        LIMIT ?
                        """,
                        (int(limit),),
                    )
                    rows = cur.fetchall()
                    return [dict(row) for row in rows]
            else:
                data = self._json_load()
                # sort by timestamp desc
                data_sorted = sorted(
                    data, key=lambda x: x.get("timestamp", ""), reverse=True
                )
                return data_sorted[:limit]
        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}")
            return []

    def export_feedback(self, export_file: str) -> bool:
        """
        Export feedback data to a file.

        Args:
            export_file: Path to export file

        Returns:
            True if export was successful, False otherwise
        """
        try:
            export_path = Path(export_file)
            if self.backend == "sqlite":
                with self._connect() as conn:
                    rows = conn.execute(
                        """
                        SELECT message_id, timestamp, user_query, response, feedback, response_length, query_length
                        FROM feedback ORDER BY datetime(timestamp) ASC
                        """
                    ).fetchall()
                    data = [dict(r) for r in rows]
            else:
                data = self._json_load()

            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_entries": len(data),
                "feedback_data": data,
            }

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Exported {len(data)} feedback entries to {export_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}")
            return False

    def clear_feedback(self) -> bool:
        """
        Clear all feedback data.

        Returns:
            True if data was cleared successfully, False otherwise
        """
        try:
            if self.backend == "sqlite":
                with self._connect() as conn:
                    conn.execute("DELETE FROM feedback")
            else:
                self._json_save([])
            logger.info("All feedback data cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing feedback: {e}")
            return False

    # JSON helpers
    def _json_load(self) -> List[Dict[str, Any]]:
        try:
            if self.json_path.exists():
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            return []
        except Exception as e:
            logger.error(f"Error reading JSON feedback: {e}")
            return []

    def _json_save(self, data: List[Dict[str, Any]]) -> None:
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error writing JSON feedback: {e}")

    def compute_session_feedback(self, messages: List[Dict[str, Any]], recent_limit: int = 5) -> Dict[str, Any]:
        """
        Compute feedback analytics from the current in-memory chat session messages.

        Args:
            messages: The Streamlit session messages list
            recent_limit: Number of recent assistant messages to include

        Returns:
            Dictionary with session feedback metrics and recent entries
        """
        try:
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            total_responses = len(assistant_messages)

            positive_feedback = len(
                [m for m in assistant_messages if m.get("feedback") == "positive"]
            )
            negative_feedback = len(
                [m for m in assistant_messages if m.get("feedback") == "negative"]
            )
            no_feedback = total_responses - positive_feedback - negative_feedback

            satisfaction_rate = (
                (positive_feedback / (positive_feedback + negative_feedback)) * 100
                if (positive_feedback + negative_feedback) > 0
                else 0.0
            )

            recent_messages = assistant_messages[-recent_limit:]
            recent_summaries: List[Dict[str, Any]] = []
            for msg in recent_messages:
                feedback = msg.get("feedback")
                preview = msg.get("content", "")
                if len(preview) > 50:
                    preview = preview[:50] + "..."
                recent_summaries.append(
                    {
                        "feedback": feedback,  # 'positive' | 'negative' | None
                        "content_preview": preview,
                    }
                )

            return {
                "total_responses": total_responses,
                "positive_feedback": positive_feedback,
                "negative_feedback": negative_feedback,
                "no_feedback": no_feedback,
                "satisfaction_rate": round(satisfaction_rate, 1),
                "recent": recent_summaries,
            }
        except Exception as e:
            logger.error(f"Error computing session feedback: {e}")
            return {
                "total_responses": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "no_feedback": 0,
                "satisfaction_rate": 0.0,
                "recent": [],
            }