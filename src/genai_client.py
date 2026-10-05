"""Client factory for Google GenAI SDK supporting both Google AI Studio
and Google Cloud Vertex AI (Gemini Enterprise Agent Platform).
"""
import os
import logging
from typing import Optional, Dict, Any, Tuple

from dotenv import load_dotenv
from google import genai
from google.api_core import exceptions
from google.genai import errors


def resolve_genai_config(
    use_vertexai: Optional[bool] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    credentials_path: Optional[str] = None,
    api_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Resolves whether to use Google Cloud Vertex AI or Google AI Studio, along with
    project, location, credentials_path, and api_key from CLI flags, config dict,
    and environment variables.
    
    Returns:
        (use_vertexai, project, location, credentials_path, api_key)
    """
    load_dotenv()
    gcp_cfg = (config or {}).get("google_cloud", {})

    # 1. Determine use_vertexai
    is_vertex = None
    if use_vertexai is not None:
        is_vertex = bool(use_vertexai)
    elif "use_vertexai" in gcp_cfg:
        is_vertex = bool(gcp_cfg.get("use_vertexai"))
    elif "vertexai" in (config or {}):
        is_vertex = bool((config or {}).get("vertexai"))
    else:
        # Check environment variables
        env_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or os.getenv("GOOGLE_GENAI_USE_ENTERPRISE") or os.getenv("VERTEXAI")
        if env_vertex is not None:
            is_vertex = env_vertex.lower() in ("true", "1", "yes", "t")

    # 2. Project
    resolved_project = (
        project
        or gcp_cfg.get("project")
        or (config or {}).get("project")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("PROJECT_ID")
    )

    # 3. Location
    resolved_location = (
        location
        or gcp_cfg.get("location")
        or (config or {}).get("location")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("CLOUD_ML_REGION")
    )

    # 4. Credentials path
    resolved_creds_path = (
        credentials_path
        or gcp_cfg.get("credentials")
        or (config or {}).get("credentials")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )

    # 5. API key
    resolved_api_key = (
        api_key
        or gcp_cfg.get("api_key")
        or (config or {}).get("api_key")
        or os.getenv("VERTEX_API_KEY") if is_vertex else None
    )
    if not resolved_api_key:
        resolved_api_key = (
            api_key
            or (config or {}).get("api_key")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

    # If is_vertex is still None, infer from project / creds vs API key
    if is_vertex is None:
        if (resolved_project or resolved_creds_path) and not os.getenv("GEMINI_API_KEY"):
            is_vertex = True
        else:
            is_vertex = False

    if is_vertex and not resolved_location:
        resolved_location = "us-central1"

    return is_vertex, resolved_project, resolved_location, resolved_creds_path, resolved_api_key


def get_genai_client(
    use_vertexai: Optional[bool] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    credentials_path: Optional[str] = None,
    credentials: Optional[Any] = None,
    api_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> genai.Client:
    """Instantiates a genai.Client configured for either Google Cloud Vertex AI
    (Gemini Enterprise Agent Platform) or Gemini Developer API (Google AI Studio).

    Args:
        use_vertexai: Force Vertex AI (True) or AI Studio (False). If None, inferred.
        project: Google Cloud project ID (for Vertex AI).
        location: Google Cloud region (default "us-central1" for Vertex AI).
        credentials_path: Path to Service Account JSON file.
        credentials: An existing google.auth.credentials.Credentials object.
        api_key: Direct API key (for AI Studio or Vertex AI Express Mode).
        config: Optional configuration dictionary (e.g. from config.yaml).

    Returns:
        Configured genai.Client instance.
    """
    is_vertex, res_project, res_location, res_creds_path, res_api_key = resolve_genai_config(
        use_vertexai=use_vertexai,
        project=project,
        location=location,
        credentials_path=credentials_path,
        api_key=api_key,
        config=config,
    )

    if is_vertex:
        auth_credentials = credentials
        if not auth_credentials and res_creds_path:
            if not os.path.exists(res_creds_path):
                raise FileNotFoundError(f"Service account credentials file not found: {res_creds_path}")
            from google.oauth2 import service_account
            auth_credentials = service_account.Credentials.from_service_account_file(res_creds_path)

        logging.info(
            f"Initializing Google GenAI client with Google Cloud Vertex AI "
            f"(Gemini Enterprise Agent Platform) [project={res_project or 'auto'}, location={res_location}]"
        )
        try:
            return genai.Client(
                vertexai=True,
                project=res_project,
                location=res_location,
                credentials=auth_credentials,
                api_key=res_api_key if not auth_credentials and not res_project else None,
            )
        except Exception as e:
            logging.error(
                "Failed to initialize Vertex AI client. Ensure your Google Cloud Project ID is set "
                "(GOOGLE_CLOUD_PROJECT or --project) and Application Default Credentials are configured "
                "(gcloud auth application-default login or GOOGLE_APPLICATION_CREDENTIALS)."
            )
            raise e

    # AI Studio / Gemini Developer API path
    if not res_api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set and no Google Cloud Vertex AI configuration provided.\n"
            "- For Google AI Studio: set GEMINI_API_KEY in .env or your shell.\n"
            "- For Google Cloud ($300 free trial / Vertex AI): set GOOGLE_GENAI_USE_VERTEXAI=true and "
            "GOOGLE_CLOUD_PROJECT=<project_id> in .env (or run with --vertexai --project <id>)."
        )

    logging.info("Initializing Google GenAI client with Gemini Developer API (Google AI Studio)")
    return genai.Client(api_key=res_api_key)
