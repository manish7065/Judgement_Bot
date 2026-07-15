import os
import time
from typing import Any, List, Optional, Sequence, Tuple

import chromadb

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0


def _load_dotenv_if_needed() -> None:
    """Load .env if no Chroma configuration has already been supplied."""
    if os.getenv("CHROMA_ENDPOINTS") or (
        os.getenv("CHROMA_HOST") and os.getenv("CHROMA_PORT")
    ):
        return

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        return


def _parse_chroma_endpoint(endpoint: str) -> Tuple[str, int]:
    """Parse a host:port endpoint string from CHROMA_ENDPOINTS."""
    if ":" not in endpoint:
        raise ValueError(
            "Invalid CHROMA_ENDPOINTS entry. Expected 'host:port' format."
        )

    host, port_str = endpoint.split(":", 1)
    try:
        port = int(port_str.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid Chroma port in endpoint '{endpoint}'. Must be an integer."
        ) from exc

    return host.strip(), port


def get_chroma_endpoints() -> List[Tuple[str, int]]:
    """Return a list of Chroma HTTP endpoints from environment variables."""
    _load_dotenv_if_needed()

    endpoints_raw = os.getenv("CHROMA_ENDPOINTS")
    if endpoints_raw:
        endpoints = [
            _parse_chroma_endpoint(item.strip())
            for item in endpoints_raw.split(",")
            if item.strip()
        ]
        if endpoints:
            return endpoints

    host = os.getenv("CHROMA_HOST")
    port = os.getenv("CHROMA_PORT")
    if host and port:
        try:
            return [(host.strip(), int(port.strip()))]
        except ValueError as exc:
            raise ValueError("CHROMA_PORT must be an integer.") from exc

    raise RuntimeError(
        "Missing Chroma configuration. Set CHROMA_ENDPOINTS or CHROMA_HOST and CHROMA_PORT."
    )


def create_chromadb_client(
    endpoints: Optional[Sequence[Tuple[str, int]]] = None,
    max_retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    local_fallback_path: Optional[str] = None,
) -> Any:
    """Create a Chroma client with retry logic and optional local fallback."""
    endpoints = list(endpoints) if endpoints is not None else get_chroma_endpoints()
    last_error = None

    for host, port in endpoints:
        for attempt in range(1, max_retries + 1):
            try:
                client = chromadb.HttpClient(host=host, port=port)
                client.list_collections()
                return client
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    time.sleep(backoff_seconds * (2 ** (attempt - 1)))

    fallback_path = local_fallback_path or os.getenv("CHROMA_LOCAL_FALLBACK_PATH")
    if fallback_path:
        try:
            return chromadb.PersistentClient(path=fallback_path)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to connect to any Chroma endpoint or fallback persistent store."
    ) from last_error


def create_chromadb_collection(
    collection_name: str,
    client: Optional[Any] = None,
    local_fallback_path: Optional[str] = None,
) -> Any:
    """Create or return an existing Chroma collection."""
    client = client or create_chromadb_client(local_fallback_path=local_fallback_path)
    return client.get_or_create_collection(name=collection_name)

