from typing import Optional
from urllib.parse import urlparse

import requests
import yfinance as yf
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# FastMCP server to download data from different APIs
app = FastMCP("FastMCP Server to download data")


class YFinanceRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g. AAPL")
    period: Optional[str] = Field(
        "1mo", description="Data period, e.g. 1d, 5d, 1mo, 1y"
    )
    interval: Optional[str] = Field("1d", description="Data interval, e.g. 1m, 1h, 1d")
    start: Optional[str] = Field(None, description="Start date YYYY-MM-DD")
    end: Optional[str] = Field(None, description="End date YYYY-MM-DD")


class ProxyRequest(BaseModel):
    url: str
    method: Optional[str] = "GET"
    params: Optional[dict] = None
    headers: Optional[dict] = None
    json_field: Optional[dict] = None
    timeout: Optional[float] = 10.0


# Restrict proxy to known data provider hosts to avoid becoming an open proxy
ALLOWED_HOSTS = {
    "query1.finance.yahoo.com",
    "finance.yahoo.com",
    "www.alphavantage.co",
    "alphavantage.co",
    "finnhub.io",
    "api.tiingo.com",
    "api.coincap.io",
    "api.coingecko.com",
}


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


@app.tool()
def download_yfinance(req: YFinanceRequest, file_name: str) -> dict[str, str]:
    """
    Download historical market data from Yahoo Finance and save to CSV.

    param req: YFinanceRequest object with parameters for data retrieval
    param file_name: Name of the CSV file to save data (must end with .csv
    return: dict with path to stored data file or error message
    """
    if not file_name.endswith(".csv"):
        return {"error": "file_name MUST end with .csv"}

    file_path = f"data/{safe_filename(file_name)}"
    try:
        ticker = yf.Ticker(req.symbol)
        # prefer history with explicit start/end if provided, otherwise period
        if req.start or req.end:
            df = ticker.history(interval=req.interval, start=req.start, end=req.end)
        else:
            df = ticker.history(period=req.period, interval=req.interval)

        if df.empty:
            return {"error": "No data returned for given parameters."}

        df.to_csv(file_path)
        return {"data_file_stored": file_path}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def proxy_request(req: ProxyRequest, file_name: str) -> dict[str, str]:
    """
    Make a proxy HTTP request to download data and save to CSV.

    param req: ProxyRequest object with parameters for the HTTP request
    param file_name: Name of the CSV file to save data (must end with .csv)
    return: dict with path to stored data file or error message
    """
    parsed = urlparse(req.url)
    hostname = parsed.hostname or ""

    if hostname not in ALLOWED_HOSTS:
        return {"error": f"Host {hostname} not allowed"}

    if not file_name.endswith(".csv"):
        return {"error": "file_name MUST end with .csv"}

    file_path = f"data/{safe_filename(file_name)}"

    try:
        resp = requests.request(
            method=req.method.upper(),
            url=req.url,
            params=req.params,
            headers=req.headers,
            json=req.json_field,
            timeout=req.timeout,
            stream=True,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"error": str(e)}

    with open(file_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return {"file_path": file_path}


@app.tool()
def allowed_hosts() -> dict[str, list[str]]:
    """
    Return the list of allowed hosts for proxy requests.

    return: dict with list of allowed hosts
    """
    return {"allowed_hosts_for_proxy_request": sorted(list(ALLOWED_HOSTS))}


if __name__ == "__main__":
    app.run(transport="stdio")
