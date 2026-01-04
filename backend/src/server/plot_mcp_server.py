# FastMCP Server with Python and R Plot Execution
import os
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp_server = FastMCP("FastMCP Server to analayse and visualise data")

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class CodeRequest(BaseModel):
    file_path: str
    code: str


class InstallRequest(BaseModel):
    library_name: str


# Route 1: Execute Python Code
@mcp_server.tool()
def run_python(req: CodeRequest) -> dict[str, str]:
    """
    Executes Python code to run an analysis of a dataframe (df) and/or generate a plot.
    If the path to the dataframe is provided, the dataframe will be loaded and made
    available in the code execution context as the variable `df`.
    If no file exists at the provided path, an error message is returned.
    Numerical results can be saved to a CSV file under the `df_file_path`.
    Plots can be generated using Matplotlib or other libraries
    and saved as PNG files under the `plot_path`.

    :param req: CodeRequest object containing
      - file path: path to a CSV or Excel file, containing the data to be analysed.
      - code: Python code to be execute for analysis.
    :return: Dictionary with the path to the generated plot and/or dataframe or
    an error message.
    """
    if not os.path.exists(req.file_path):
        return {"error": f"File {req.file_path} does not exist."}

    df = (
        pd.read_csv(req.file_path)
        if req.file_path.endswith(".csv")
        else pd.read_excel(req.file_path)
    )
    context = {"df": df, "plot_path": "", "df_file_path": ""}

    try:
        exec(req.code, context)
        return {
            "plot_path": context["plot_path"],
            "df_file_path": context["df_file_path"],
        }
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def get_df_infos_python(req: CodeRequest) -> dict[str, Any]:
    """
    Executes python code that extracts metadata from pandas DataFrame `df`.
    It is important that the python code stores all calculated metadata in the
    `res` dictionary. This `res` dictionary must be JSON serializable.

    :param req: CodeRequest object containing the code to be executed and the file path.
    :return: Dictionary containing metadata about the `df`,
             including the first few rows.
    """
    df = (
        pd.read_csv(req.file_path)
        if req.file_path.endswith(".csv")
        else pd.read_excel(req.file_path)
    )
    try:
        context = {"df": df, "res": {}}
        exec(req.code, context)
        res = context["res"]
        return res
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def install_python_library(library_name: str) -> dict[str, str]:
    """
    Installs a Python library using pip.

    :param library_name: Name of the Python library to install.
    :return: Dictionary with success or error message.
    """
    import subprocess

    try:
        subprocess.check_call(["pip", "install", library_name])
        return {"message": f"Successfully installed {library_name}"}
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp_server.run(transport="stdio")
