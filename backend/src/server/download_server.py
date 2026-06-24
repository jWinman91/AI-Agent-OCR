from pathlib import Path

import pandas as pd
import yfinance as yf
from src.utils.data_models import (
    DataDownloadResponse,
    DataDownloadResult,
    YFinanceRequest,
)


def safe_filename(name: Path) -> Path:
    return Path(str(name).replace("/", "_").replace("\\", "_"))


def resolve_ticker(symbol: str) -> str:
    s = yf.Search(symbol)
    if not s.quotes:
        raise ValueError(f"No ticker found for symbol: {symbol}")
    return s.quotes[0]["symbol"]


def download_yfinance(reqs: list[YFinanceRequest], file_name: Path) -> Path:
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

    if not file_name.suffix == ".csv":
        raise ValueError("file_name MUST end with .csv")

    file_path = Path("data") / safe_filename(file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)

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


def handle_data_download(download_response: DataDownloadResponse) -> DataDownloadResult:
    """
    Handle a data download request, either from Yahoo Finance or via proxy.

    :param download_response: DataDownloadResponse object containing the download
    request details.
    :return: DataDownloadResult object containing the path to the downloaded data file
    """
    reqs = download_response.download_request
    file_name = download_response.file_name

    if all(isinstance(req, YFinanceRequest) for req in reqs):
        try:
            data_file_path = download_yfinance(reqs, file_name)
            return DataDownloadResult(
                data_file_path=data_file_path,
            )
        except Exception as e:
            return DataDownloadResult(
                data_file_path=None,
                error_message=str(e),
            )
    else:
        return DataDownloadResult(
            data_file_path=None,
            error_message=(
                "Invalid request format. Expected a list of YFinanceRequest objects.",
            ),
        )
