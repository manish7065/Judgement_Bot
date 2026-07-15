"""
This script tests different methods of connecting to the Turso database and retrieving data.
It demonstrates the recommended libsql-client approach for CRUD operations,
and also provides the raw HTTP approach as a reference.

The libsql-client library handles the Turso HTTP pipeline protocol internally,
giving you a standard DB-API-like interface — no manual JSON payload construction needed.
"""
import os
import json
import httpx

from dotenv import load_dotenv
load_dotenv()


# ──────────────────────────────────────────────
# Recommended approach: libsql-client
# ──────────────────────────────────────────────

def test_with_libsql_client() -> bool:
    """
    Test connection using the libsql-client library.
    This is the recommended approach as it handles:
    - HTTP pipeline protocol automatically
    - Connection pooling
    - Auth token in headers
    - Result parsing into Row objects
    """
    try:
        import libsql_client

        db = libsql_client.create_client_sync(
            url=os.getenv("TURSO_DB_URL"),
            auth_token=os.getenv("TURSO_DB_TOKEN"),
        )

        result = db.execute("SELECT 1")
        print(f"Connection successful via libsql-client. Result: {result.rows}")
        return True

    except ImportError:
        print("libsql-client not installed. Run: pip install libsql-client")
        return False
    except Exception as e:
        print(f"libsql-client error: {e}")
        return False


def crud_example_with_libsql_client():
    """
    Example of basic CRUD operations using libsql-client.
    No JSON payloads, no pipeline construction — just standard SQL.
    """
    import libsql_client

    db = libsql_client.create_client_sync(
        url=os.getenv("TURSO_DB_URL"),
        auth_token=os.getenv("TURSO_DB_TOKEN"),
    )

    print("\n--- CRUD Example (libsql-client) ---")

    # CREATE table
    db.execute("CREATE TABLE IF NOT EXISTS turso_test (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, value INTEGER)")
    print("✓ Table created/verified")

    # INSERT
    db.execute("INSERT INTO turso_test (name, value) VALUES (?, ?)", args=("alpha", 100))
    db.execute("INSERT INTO turso_test (name, value) VALUES (?, ?)", args=("beta", 200))
    print("✓ Rows inserted")

    # READ
    rows = db.execute("SELECT * FROM turso_test").rows
    print(f"✓ Read {len(rows)} row(s):")
    for row in rows:
        print(f"   id={row['id']}, name={row['name']}, value={row['value']}")

    # UPDATE
    db.execute("UPDATE turso_test SET value = ? WHERE name = ?", args=(999, "alpha"))
    print("✓ Row updated")

    # READ after update
    row = db.execute("SELECT * FROM turso_test WHERE name = 'alpha'").rows[0]
    print(f"   Verified: name={row['name']}, value={row['value']} (should be 999)")

    # DELETE
    db.execute("DELETE FROM turso_test WHERE name = ?", args=("beta",))
    remaining = db.execute("SELECT count(*) as cnt FROM turso_test").rows[0]["cnt"]
    print(f"✓ Row deleted. Remaining rows: {remaining}")

    # Cleanup
    db.execute("DROP TABLE turso_test")
    print("✓ Cleanup complete (table dropped)")


# ──────────────────────────────────────────────
# Raw HTTP approach (reference)
# ──────────────────────────────────────────────

TURSO_DB_URL = os.getenv("TURSO_DB_URL")
TURSO_DB_TOKEN = os.getenv("TURSO_DB_TOKEN")

headers = {
    "Authorization": f"Bearer {TURSO_DB_TOKEN}",
    "Content-Type": "application/json"
}

timeout = httpx.Timeout(
    connect=30.0,
    read=120.0,
    write=120.0,
    pool=120.0
)

client = httpx.Client(timeout=timeout)


def test_with_http() -> bool:
    """
    Test connection using raw HTTP (Turso pipeline API directly).
    This is mainly for reference — use libsql-client for production.
    """
    try:
        pipeline_url = f"{TURSO_DB_URL}/v2/pipeline"
        payload = {
            "requests": [
                {"type": "execute", "stmt": {"sql": "SELECT 1;"}}
            ]
        }

        response = client.post(pipeline_url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        # Check if the query succeeded — "type": "ok" is at the top level of each result item
        if result.get("results") and result["results"][0].get("type") == "ok":
            print(f"Connection successful via HTTP. Result: {result['results'][0]['response']['result']['rows']}")
            return True
        else:
            print(f"Unexpected response format: {json.dumps(result, indent=2)}")
            return False

    except httpx.RequestError as e:
        print(f"HTTP request error: {e}")
        return False
    except httpx.HTTPStatusError as e:
        print(f"HTTP status error {e.response.status_code}: {e.response.text}")
        return False
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Response parsing error: {e}")
        return False


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing Turso DB Connection ===\n")

    # Method 1: libsql-client (recommended)
    print("[Method 1] libsql-client:")
    success_1 = test_with_libsql_client()
    print()

    # Method 2: Raw HTTP (reference)
    print("[Method 2] Raw HTTP:")
    success_2 = test_with_http()
    print()

    if success_1:
        crud_example_with_libsql_client()
        print()

    overall = success_1 or success_2
    print(f"{'='*40}")
    print(f"Overall: Connection test {'PASSED' if overall else 'FAILED'}")