from typing import Optional

import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field


class YFinanceRequest(BaseModel):
    share_name: str = Field(
        ..., description="Name of the share to download data for, e.g. 'E.ON'"
    )
    period: Optional[str] = Field(
        "1mo", description="Data period, e.g. 1d, 5d, 1mo, 1y"
    )
    interval: Optional[str] = Field("1d", description="Data interval, e.g. 1m, 1h, 1d")
    start: Optional[str] = Field(None, description="Start date YYYY-MM-DD")
    end: Optional[str] = Field(None, description="End date YYYY-MM-DD")


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def resolve_ticker(symbol: str) -> str:
    s = yf.Search(symbol)
    if not s.quotes:
        raise ValueError(f"No ticker found for symbol: {symbol}")
    return s.quotes[0]["symbol"]


def download_yfinance(reqs: list[YFinanceRequest], file_name: str) -> str:
    """
    Download historical market data from Yahoo Finance and save to CSV.

    Output format:
        Date,
        EOAN.DE_Open,
        EOAN.DE_High,
        EOAN.DE_Low,
        EOAN.DE_Close,
        EOAN.DE_Volume,
        AAPL_Open,
        AAPL_High,
        ...

    Each row corresponds to a datetime.
    """

    if not file_name.endswith(".csv"):
        raise ValueError("file_name MUST end with .csv")

    file_path = f"data/{safe_filename(file_name)}"

    dfs = []

    for req in reqs:
        ticker_symbol = resolve_ticker(req.share_name)

        ticker = yf.Ticker(ticker_symbol)

        if req.start or req.end:
            df = ticker.history(
                interval=req.interval,
                start=req.start,
                end=req.end,
            )
        else:
            df = ticker.history(
                period=req.period,
                interval=req.interval,
            )

        if df.empty:
            raise ValueError(
                f"No data returned for '{req.share_name}' ({ticker_symbol})"
            )

        # Remove timezone information to ensure alignment
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        # Prefix all columns with ticker symbol
        df = df.add_prefix(f"{ticker_symbol}_")

        dfs.append(df)

    # Outer join keeps all timestamps from all securities
    combined_df = pd.concat(dfs, axis=1, join="outer")

    combined_df.index.name = "Date"

    # Sort chronologically
    combined_df.sort_index(inplace=True)

    # Optional: remove rows where every column is NaN
    combined_df.dropna(how="all", inplace=True)

    combined_df.to_csv(file_path)

    return file_path


def handle_data_download(
    reqs: list[YFinanceRequest],
    file_name: str,
) -> str:
    """
    Handle a data download request, either from Yahoo Finance or via proxy.

    param reqs: List of YFinanceRequest objects with parameters for the request
    param file_name: Name of the CSV file to save data (must end with .csv)
    return: Path to the stored data file
    """
    if all(isinstance(req, YFinanceRequest) for req in reqs):
        return download_yfinance(reqs, file_name)
    else:
        raise ValueError("Invalid request type")
