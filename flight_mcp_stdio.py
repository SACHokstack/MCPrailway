import os
from typing import Annotated

import httpx
from fastmcp import FastMCP
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR
from pydantic import Field

# Minimal stdio-based MCP server exposing flight tracking tools.
# This is intended to be launched directly by Cline via mcp settings (stdio transport).
mcp = FastMCP("Flight Tracker MCP")

API_BASE = "https://api.aviationstack.com/v1"
AVIATIONSTACK_API_KEY = os.environ.get("AVIATIONSTACK_API_KEY")


def _require_api_key():
    if not AVIATIONSTACK_API_KEY:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message="AVIATIONSTACK_API_KEY is not set. Please configure it in your MCP settings."
            )
        )


@mcp.tool(description="Look up real-time flight details by IATA flight number (e.g., AA100)")
async def get_flight_info(
    flight_number: Annotated[str, Field(description="IATA flight number (e.g., 'AA100')")]
) -> str:
    """Fetches real-time flight details from AviationStack API."""
    _require_api_key()
    flight_number = flight_number.upper().strip()

    url = f"{API_BASE}/flights"
    params = {
        "access_key": AVIATIONSTACK_API_KEY,
        "flight_iata": flight_number
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch flight data: {str(e)}"))

    if not data.get("data"):
        return f"✈️ No data found for flight {flight_number}."

    flight = data["data"][0]
    airline = flight.get("airline", {}).get("name", "Unknown Airline")
    departure = flight.get("departure", {}).get("airport", "Unknown Departure")
    arrival = flight.get("arrival", {}).get("airport", "Unknown Arrival")
    status = flight.get("flight_status", "Unknown Status")
    dep_time = flight.get("departure", {}).get("estimated", "N/A")
    arr_time = flight.get("arrival", {}).get("estimated", "N/A")

    status_emoji = {
        "scheduled": "🕐",
        "active": "✈️",
        "landed": "🛬",
        "cancelled": "❌",
        "delayed": "⏰",
        "diverted": "🔄",
        "unknown": "❓",
    }.get(str(status).lower(), "📋")

    return (
        f"✈️ Flight {flight_number} ({airline})\n"
        f"{status_emoji} Status: {status.title() if isinstance(status, str) else status}\n"
        f"🛫 From: {departure} → 🛬 To: {arrival}\n"
        f"📅 Departure (Estimated): {dep_time}\n"
        f"📅 Arrival (Estimated): {arr_time}"
    )


@mcp.tool(name="track_flight", description="Track a flight by IATA number with formatted status and times.")
async def track_flight(
    flight_number: Annotated[str, Field(description="IATA flight number to track (e.g., 'AA100', 'DL123', 'UA456')")]
) -> str:
    """Track a flight and get real-time status information."""
    _require_api_key()
    flight_number = flight_number.upper().strip()

    url = f"{API_BASE}/flights"
    params = {
        "access_key": AVIATIONSTACK_API_KEY,
        "flight_iata": flight_number
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        return f"❌ Flight Tracking Error: {str(e)}"

    if not data.get("data"):
        return (
            f"✈️ Flight Tracking Result\n\n"
            f"No active flight data found for {flight_number}.\n\n"
            f"Possible reasons:\n- Not scheduled today\n- Number format incorrect\n- Flight already completed"
        )

    flight = data["data"][0]

    airline = flight.get("airline", {}).get("name", "Unknown Airline")
    departure_info = flight.get("departure", {}) or {}
    arrival_info = flight.get("arrival", {}) or {}
    dep_airport = departure_info.get("airport", "Unknown Departure")
    arr_airport = arrival_info.get("airport", "Unknown Arrival")
    status = (flight.get("flight_status") or "unknown").lower()

    dep_scheduled = departure_info.get("scheduled", "N/A")
    dep_estimated = departure_info.get("estimated", dep_scheduled)
    arr_scheduled = arrival_info.get("scheduled", "N/A")
    arr_estimated = arrival_info.get("estimated", arr_scheduled)

    status_emoji = {
        "scheduled": "🕐",
        "active": "✈️",
        "landed": "🛬",
        "cancelled": "❌",
        "delayed": "⏰",
        "diverted": "🔄",
        "unknown": "❓",
    }.get(status, "📋")

    result = []
    result.append(f"✈️ Flight Tracking: {flight_number}")
    result.append(f"🏢 Airline: {airline}")
    result.append(f"{status_emoji} Status: {status.title()}")
    result.append("")
    result.append("🛫 Departure")
    result.append(f"   📍 Airport: {dep_airport}")
    result.append(f"   📅 Scheduled: {dep_scheduled}")
    result.append(f"   ⏱️ Estimated: {dep_estimated}")
    result.append("")
    result.append("🛬 Arrival")
    result.append(f"   📍 Airport: {arr_airport}")
    result.append(f"   📅 Scheduled: {arr_scheduled}")
    result.append(f"   ⏱️ Estimated: {arr_estimated}")

    return "\n".join(result)


# Simple health/validation tool so clients can test connectivity
@mcp.tool(description="Return a static value to validate connectivity.")
async def validate() -> str:
    return "ok"


# --- Tool: Taddy Podcast API ---
TADDY_ENDPOINT = os.environ.get("TADDY_ENDPOINT") or "https://api.taddy.org"
TADDY_USER_ID = os.environ.get("TADDY_USER_ID") or "3210"
TADDY_API_KEY = os.environ.get("TADDY_API_KEY") or "946ebcc8e8196d9065f45827c180850683e01dc132554b4e6d341b19a6e718e2065662fdcbf07a7dc8ebca7459a987a1d2"


async def _taddy_request(query: str) -> dict:
    """
    Sends a GraphQL request to the Taddy API.
    """
    if not TADDY_API_KEY or not TADDY_USER_ID:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message="TADDY_API_KEY or TADDY_USER_ID is not configured. Set them via MCP environment."
            )
        )

    headers = {
        "Content-Type": "application/json",
        "X-USER-ID": str(TADDY_USER_ID),
        "X-API-KEY": TADDY_API_KEY,
    }
    payload = {"query": query}

    async with httpx.AsyncClient() as client:
        resp = await client.post(TADDY_ENDPOINT, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()


@mcp.tool(name="podcast_query", description="Run an arbitrary GraphQL query against the Taddy podcast API")
async def podcast_query(
    query: Annotated[str, Field(description="GraphQL query string to send to Taddy, e.g., { getPodcastSeries(name:\"This American Life\") { uuid name } }")]
) -> str:
    """
    Executes a raw GraphQL query on Taddy and returns the JSON response.
    """
    import json
    try:
        result = await _taddy_request(query)
        # If GraphQL errors are returned, surface them clearly
        if isinstance(result, dict) and result.get("errors"):
            return f"❌ Taddy GraphQL errors:\n{json.dumps(result['errors'], indent=2, ensure_ascii=False)}"
        return json.dumps(result, indent=2, ensure_ascii=False)
    except httpx.HTTPError as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Taddy API request failed: {str(e)}"))


@mcp.tool(name="get_podcast_series", description="Fetch podcast series details by name from Taddy")
async def get_podcast_series(
    name: Annotated[str, Field(description="Podcast series name, e.g., 'This American Life'")]
) -> str:
    """
    Convenience tool wrapping Taddy's getPodcastSeries query.
    """
    # Escape double quotes in the series name for GraphQL safety
    safe_name = name.replace('"', '\\"')
    query = f'{{ getPodcastSeries(name:"{safe_name}") {{ uuid name }} }}'
    try:
        result = await _taddy_request(query)
        data = result.get("data") or {}
        series = data.get("getPodcastSeries")

        if series is None:
            return f"🔎 No podcast series found for name: {name}"

        # Handle both single object or list responses gracefully
        def fmt(item: dict) -> str:
            return f"- {item.get('name', 'Unknown')} (uuid: {item.get('uuid', 'n/a')})"

        if isinstance(series, list):
            if not series:
                return f"🔎 No podcast series found for name: {name}"
            lines = [fmt(item) for item in series if isinstance(item, dict)]
            return "🎙️ Taddy Podcast Series Results:\n" + "\n".join(lines)
        elif isinstance(series, dict):
            return "🎙️ Taddy Podcast Series Result:\n" + fmt(series)
        else:
            return f"Unexpected response shape from Taddy: {series!r}"

    except httpx.HTTPError as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Taddy API request failed: {str(e)}"))

if __name__ == "__main__":
    # Expose as stdio transport so Cline can launch it via MCP settings.
    mcp.run()
