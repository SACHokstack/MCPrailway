import asyncio
from typing import Annotated
import os
from threading import Thread
from flask import Flask, request, Response, stream_with_context
from dotenv import load_dotenv
import finnhub
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl
import markdownify
import httpx
import readabilipy
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
import akinator
import sys
sys.path.append(os.path.dirname(__file__))  # Add current directory to path

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder (now smart!) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    """
    Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
    """
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


# Image inputs and sending images

MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io

    from PIL import Image

    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))
    

# --- Tool: Stock Price ---
# Setup Finnhub client
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

StockPriceDescription = RichToolDescription(
    description="Live stock price tool: fetches the latest market price for a given stock symbol.",
    use_when="Use this when you need the current price of a stock using its ticker symbol.",
    side_effects="Returns the latest market price in USD from Finnhub.",
)

@mcp.tool(description=StockPriceDescription.model_dump_json())
async def get_live_stock_price(
    symbol: Annotated[str, Field(description="The stock ticker symbol, e.g., AAPL for Apple, TSLA for Tesla")]
) -> str:
    """
    Fetches the current live price of the given stock symbol from Finnhub.
    """
    try:
        quote = finnhub_client.quote(symbol)
        current_price = quote.get("c")  # 'c' = current price in Finnhub API

        if current_price is None:
            raise McpError(
                ErrorData(code=INVALID_PARAMS, message=f"Could not fetch live price for symbol: {symbol}")
            )

        return f"💹 **{symbol}** — Current Price: **${current_price}**"

    except Exception as e:
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Error fetching stock price: {str(e)}")
        )


api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=510)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description='Query Arxiv Papers')

ArxivDescription = RichToolDescription(
    description="Arxiv paper search: query scientific papers from arXiv by topic, keywords, or authors.",
    use_when=(
    "Use this ONLY for queries requesting academic research papers, scholarly articles, "
    "scientific publications, or abstracts — especially when the request mentions 'paper', 'papers', "
    "'research', 'study', 'NLP research', 'transformers research', or 'arxiv'. "
    "DO NOT use web search for these queries. "
    "Use when a game is mentioned in the user query. "
    "Always use this tool instead of general search."),
    side_effects="Returns relevant paper titles, abstracts, and links.",
)

@mcp.tool(name="academic_paper_search", description=ArxivDescription.model_dump_json())
async def query_arxiv(
    query: Annotated[str, Field(description="Your search query for arXiv, e.g., 'transformers in NLP'")]
) -> str:
    """
    Queries the arXiv API for papers matching the given query and returns the results.
    """
    try:
        result = arxiv_tool.run(query)  # synchronous call inside async function
        return f"📚 **Arxiv Results for:** _{query}_\n\n{result}"
    except Exception as e:
        return f"Error fetching Arxiv papers: {e}"
    



import akinator

# Global game state
aki_instance = None

@mcp.tool(description="Play Akinator - the character guessing game")
async def play_akinator(
    user_answer: Annotated[str, Field(description="Your answer: start/y/n/i/p/pn/b/exit")]
) -> str:
    global aki_instance
    
    user_input = user_answer.strip().lower()
    
    # Exit game
    if user_input == "exit":
        aki_instance = None
        return "👋 Game exited. Thanks for playing!"
    
    # Start new game
    if user_input == "start":
        aki_instance = akinator.Akinator()
        aki_instance.start_game()
        return f"🎮 Game Started!\n\nQuestion: {str(aki_instance)}\n\nAnswers: [y]es/[n]o/[i] don't know/[p]robably/[pn] probably not/[b]ack/[exit]"
    
    # No active game
    if aki_instance is None:
        return "No game active. Type 'start' to begin or 'exit' to quit."
    
    # Game finished
    if aki_instance.finished:
        result = (f"🏆 Game Over!\n\n"
                 f"Proposition: {aki_instance.name_proposition}\n"
                 f"Description: {aki_instance.description_proposition}\n"
                 f"Photo: {aki_instance.photo}\n"
                 f"Final Message: {aki_instance.question}")
        aki_instance = None
        return result
    
    # Go back
    if user_input == "b":
        try:
            aki_instance.back()
            return f"↩️ Went back.\n\nQuestion: {str(aki_instance)}"
        except akinator.CantGoBackAnyFurther:
            return "⚠️ You can't go back any further!\n\nQuestion: {str(aki_instance)}"
    
    # Answer question
    try:
        aki_instance.answer(user_input)
        if aki_instance.finished:
            result = (f"🏆 Game Over!\n\n"
                     f"Proposition: {aki_instance.name_proposition}\n"
                     f"Description: {aki_instance.description_proposition}\n"
                     f"Photo: {aki_instance.photo}\n"
                     f"Final Message: {aki_instance.question}")
            aki_instance = None
            return result
        else:
            return f"Question: {str(aki_instance)}"
    except akinator.InvalidChoiceError:
        return f"⚠️ Invalid answer. Please try again.\n\nQuestion: {str(aki_instance)}\n\nAnswers: [y]es/[n]o/[i] don't know/[p]robably/[pn] probably not/[b]ack/[exit]"
    

# Traffic tool with inline implementation
import requests
import time
from typing import Optional, Dict

@mcp.tool(description="Get traffic updates between two locations")
async def get_traffic_update(
    query: Annotated[str, Field(description="Traffic query: 'from A to B' or 'A|B'")]
) -> str:
    
    def geocode(location: str) -> Optional[Dict]:
        """Simple geocoding function"""
        try:
            time.sleep(1)  # Rate limiting
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': location,
                'format': 'json',
                'limit': 1
            }
            headers = {'User-Agent': 'TrafficBot/1.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            
            if data:
                return {
                    'lat': float(data[0]['lat']),
                    'lon': float(data[0]['lon'])
                }
            return None
        except:
            return None
    
    def get_route(origin_coords: Dict, dest_coords: Dict) -> Optional[Dict]:
        """Simple routing function"""
        try:
            origin = f"{origin_coords['lon']},{origin_coords['lat']}"
            dest = f"{dest_coords['lon']},{dest_coords['lat']}"
            
            url = f"http://router.project-osrm.org/route/v1/driving/{origin};{dest}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data['code'] == 'Ok' and data['routes']:
                route = data['routes'][0]
                return {
                    'distance': route['distance'],
                    'duration': route['duration']
                }
            return None
        except:
            return None
    
    try:
        # Parse query
        if " to " in query.lower():
            parts = query.lower().split(" to ")
            origin = parts[0].replace("from ", "").strip()
            destination = parts[1].strip()
        elif "|" in query:
            parts = query.split("|")
            origin = parts[0].strip()
            destination = parts[1].strip()
        else:
            return "❌ Use format: 'from A to B' or 'A|B'"
        
        # Get coordinates
        origin_coords = geocode(origin)
        if not origin_coords:
            return f"❌ Location not found: {origin}"
        
        dest_coords = geocode(destination)
        if not dest_coords:
            return f"❌ Location not found: {destination}"
        
        # Get route
        route = get_route(origin_coords, dest_coords)
        if not route:
            return "❌ Could not calculate route"
        
        # Format response
        distance_km = route['distance'] / 1000
        duration_min = route['duration'] / 60
        
        # Simple traffic estimate
        expected_speed = 40  # km/h
        expected_time = (distance_km / expected_speed) * 60
        
        if duration_min <= expected_time * 1.2:
            traffic = "🟢 Light traffic"
        elif duration_min <= expected_time * 1.6:
            traffic = "🟡 Moderate traffic"
        else:
            traffic = "🔴 Heavy traffic"
        
        return f"""🚗 Traffic Update

📍 {origin.title()} → {destination.title()}
📏 Distance: {distance_km:.1f} km
⏱️ Duration: {duration_min:.0f} minutes
{traffic}"""
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- Original MCP main function ---
async def mcp_main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

# --- Flask app wrapper ---
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "✅ MCP Server is running on Railway!"

@flask_app.route("/health")
def health():
    return {"status": "healthy", "service": "MCP Server"}

# --- Reverse proxy to expose internal MCP server on public port without redirects ---
@flask_app.route("/mcp", methods=["GET", "POST", "OPTIONS"])
@flask_app.route("/mcp/", defaults={"subpath": ""}, methods=["GET", "POST", "OPTIONS"])
@flask_app.route("/mcp/<path:subpath>", methods=["GET", "POST", "OPTIONS"])
def mcp_proxy(subpath: str | None = None):
    """
    Proxies requests from public /mcp[...] to the internal FastMCP server on 127.0.0.1:8086/mcp/[...]
    - Avoids 307 redirects to http by terminating HTTPS at Flask and forwarding internally.
    - Preserves streaming responses (e.g., event streams) via generator.
    """
    import requests  # local import to avoid any top-level import ordering issues

    target_base = "http://127.0.0.1:8086/mcp/"
    target_url = target_base + (subpath or "")

    # Forward headers except hop-by-hop and host-specific
    excluded = {"host", "content-length", "connection", "accept-encoding"}
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}

    # Forward request to internal MCP server
    resp = requests.request(
        method=request.method,
        url=target_url,
        params=request.args,
        data=request.get_data(),
        headers=fwd_headers,
        stream=True,
    )

    # Build proxied response, excluding hop-by-hop headers
    excluded_resp = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    response_headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in excluded_resp]

    def generate():
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    return Response(stream_with_context(generate()), status=resp.status_code, headers=response_headers)

# --- Run MCP server in background ---
def run_mcp():
    asyncio.run(mcp_main())  # Calls your existing main()

# --- Entry point ---
if __name__ == "__main__":
    # Start MCP server in separate thread
    Thread(target=run_mcp, daemon=True).start()
    
    # Run Flask web server (Railway will use this)
    port = int(os.environ.get("PORT", 8000))
    flask_app.run(host="0.0.0.0", port=port)
