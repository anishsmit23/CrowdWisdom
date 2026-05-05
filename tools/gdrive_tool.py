"""CrewAI custom tool — fetches CWT product data from Google Drive.

Reads a Google Drive file (Google Doc or binary) and returns its
text content.  Falls back to a hardcoded placeholder (FR-18) if
the Drive API is unreachable or mis-configured, ensuring the
downstream agents always have *something* to work with.
"""

from __future__ import annotations

from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from cwt_ads_agent.config import config
from cwt_ads_agent.utils.logger import get_logger
from cwt_ads_agent.utils.retry import AgentError, retry_with_backoff

_log = get_logger(__name__)

# ------------------------------------------------------------------ #
# FR-18: guaranteed fallback so the pipeline never stalls on Drive
# ------------------------------------------------------------------ #
_PLACEHOLDER = (
    "CWT PLACEHOLDER: CrowdWisdomTrading provides AI-powered trading signals "
    "with 87% accuracy. Members get real-time alerts, portfolio analysis, "
    "and access to a community of 12,000+ traders. Plans start at $49/month. "
    "Free 7-day trial available."
)

# ------------------------------------------------------------------ #
# Scopes
# ------------------------------------------------------------------ #
_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# ------------------------------------------------------------------ #
# Tool input schema
# ------------------------------------------------------------------ #
class _GDriveInput(BaseModel):
    """Input schema for GDriveTool."""

    file_id: str = Field(
        default="",
        description=(
            "Google Drive file ID to fetch. "
            "Leave empty to use the default from GDRIVE_FILE_ID env-var."
        ),
    )


# ------------------------------------------------------------------ #
# Tool implementation
# ------------------------------------------------------------------ #
class GDriveTool(BaseTool):
    """Fetches CWT product data from Google Drive."""

    name: str = "Google Drive Reader"
    description: str = "Fetches CWT product data from Google Drive"
    args_schema: Type[BaseModel] = _GDriveInput

    # In-memory cache keyed by file_id → content string.
    _cache: dict[str, str] = {}

    def _run(self, file_id: str = "") -> str:
        """Fetch file content from Google Drive.

        Parameters
        ----------
        file_id:
            Drive file ID.  Falls back to ``config.gdrive_file_id``.

        Returns
        -------
        str
            Plain-text file content, or the FR-18 placeholder on
            any unrecoverable Drive error.
        """
        resolved_id = file_id.strip() if file_id else config.gdrive_file_id
        if not resolved_id:
            _log.warning("No GDRIVE_FILE_ID configured — returning placeholder (FR-18)")
            return _PLACEHOLDER

        # --- cache hit ---
        if resolved_id in self._cache:
            _log.info("GDriveTool cache hit for file_id=%s", resolved_id)
            return self._cache[resolved_id]

        # --- fetch with retry ---
        try:
            content = retry_with_backoff(
                fn=self._fetch_file,
                max_retries=2,
                backoff=[3, 6],
                error_context={"tool": self.name, "file_id": resolved_id},
                file_id=resolved_id,
            )
        except AgentError:
            _log.warning(
                "All Drive retries exhausted for file_id=%s — "
                "returning placeholder (FR-18)",
                resolved_id,
            )
            return _PLACEHOLDER

        self._cache[resolved_id] = content
        return content

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_credentials(creds_path: str):
        """Load or refresh OAuth2 credentials.

        Looks for a cached ``token.json`` first; falls back to the
        service-account / OAuth flow defined by *creds_path*.
        """
        from pathlib import Path

        try:
            from google.oauth2.service_account import Credentials as SACredentials
        except ImportError as exc:
            raise AgentError(
                "google-auth is not installed",
                context={"dependency": "google-auth"},
            ) from exc

        token_path = Path(creds_path).parent / "token.json"

        creds = None

        # 1. Try cached token
        if token_path.exists():
            try:
                from google.oauth2.credentials import Credentials

                creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)
            except Exception:  # noqa: BLE001
                creds = None

        # 2. Refresh if expired
        if creds and creds.expired and creds.refresh_token:
            try:
                from google.auth.transport.requests import Request

                creds.refresh(Request())
            except Exception:  # noqa: BLE001
                creds = None

        # 3. Fresh flow — service account or installed-app
        if not creds or not creds.valid:
            creds_file = Path(creds_path)
            if not creds_file.exists():
                raise AgentError(
                    f"Credentials file not found: {creds_path}",
                    context={"tool": "Google Drive Reader", "path": creds_path},
                )

            import json

            with open(creds_file) as f:
                creds_data = json.load(f)

            if creds_data.get("type") == "service_account":
                creds = SACredentials.from_service_account_file(
                    str(creds_file), scopes=_SCOPES
                )
            else:
                from google_auth_oauthlib.flow import InstalledAppFlow

                flow = InstalledAppFlow.from_client_secrets_file(
                    str(creds_file), _SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Persist for next time
            if hasattr(creds, "to_json"):
                token_path.write_text(creds.to_json())

        return creds

    @staticmethod
    def _fetch_file(*, file_id: str) -> str:
        """Download file content from Drive (export or get_media)."""
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise AgentError(
                "google-api-python-client is not installed",
                context={"dependency": "google-api-python-client"},
            ) from exc

        creds = GDriveTool._load_credentials(config.gdrive_credentials_path)
        service = build("drive", "v3", credentials=creds)

        # Attempt 1: export as text/plain (works for Google Docs)
        try:
            _log.info("Attempting files().export() for file_id=%s", file_id)
            response = (
                service.files()
                .export(fileId=file_id, mimeType="text/plain")
                .execute()
            )
            content = (
                response.decode("utf-8") if isinstance(response, bytes) else response
            )
            _log.info("Export succeeded — %d chars", len(content))
            return content
        except Exception as export_exc:  # noqa: BLE001
            _log.info(
                "Export failed (%s), falling back to get_media()", export_exc
            )

        # Attempt 2: raw binary download (non-Google-Doc files)
        try:
            _log.info("Attempting files().get_media() for file_id=%s", file_id)
            response = (
                service.files().get_media(fileId=file_id).execute()
            )
            content = (
                response.decode("utf-8", errors="replace")
                if isinstance(response, bytes)
                else str(response)
            )
            _log.info("get_media succeeded — %d chars", len(content))
            return content
        except Exception as media_exc:
            _log.warning("get_media also failed: %s", media_exc)
            raise
