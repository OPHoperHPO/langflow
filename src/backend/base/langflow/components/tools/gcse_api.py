import httpx

from typing import Any

from langchain.tools import StructuredTool
from loguru import logger
from pydantic import BaseModel, Field

from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.field_typing import Tool
from langflow.inputs import IntInput, MultilineInput, SecretStrInput, StrInput
from langflow.schema import Data
from typing import List, Optional, Dict, Any

__all__ = ["GCSEAPIComponent"]


class GCSEAPIComponent(LCToolComponent):
    display_name = "GCSE Search API"
    description = "Call GCSE Search API with result limiting"
    name = "GCSEAPI"

    inputs = [
        SecretStrInput(name="gcse_api_key", display_name="GCSE API Key", required=True),
        SecretStrInput(name="gcse_endpoint_url", display_name="GCSE Endpoint Url", required=True),
        MultilineInput(
            name="input_value",
            display_name="Input",
        ),
        IntInput(name="page", display_name="Curr page", value=1, advanced=True),
        IntInput(name="max_results", display_name="Max Results", value=5, advanced=True),
        StrInput(name="hl", display_name="Language code", value="ru", advanced=True),
        StrInput(name="search_type", display_name="Search type", value=None, advanced=True),
    ]

    class GCSEAPISchema(BaseModel):
        query: str = Field(..., description="The search query")
        page: int = Field(default=1, description="Page in the search results to look up")
        hl: str = Field("ru", description="Language code for the search results")
        max_results: int = Field(5, description="Maximum number of results to return")
        search_type: Optional[str] = Field(None, description="Type of search results to return")

    def build_tool(self) -> Tool:
        class GoogleSearchError(BaseException):
                """Custom exception class to handle errors related to Google Custom Search API"""
            
                pass
            
            
        class GoogleCSEAPI:
                """
                A class to interact with the Google Custom Search Engine (CSE) API using httpx.
            
                Attributes:
                    http_client (httpx.AsyncClient): The HTTP client used to perform asynchronous requests to the CSE API.
                """
            
                def __init__(self, api_url: str, api_token: str) -> None:
                    """
                    Initializes the GoogleCSEAPI object with a base URL and an authentication token.
            
                    Args:
                        api_url (str): The base URL for the Google CSE API.
                        api_token (str): The authentication token used to authenticate API requests.
                    """
                    self.api_url = api_url
                    self.api_token = api_token
            
                async def search(
                    self,
                    query: str,
                    num: int = 10,
                    start: int = 0,
                    hl: str = "ru",
                    search_type: Optional[str] = None,
                ) -> dict[str, str]:
                    """
                    Performs a search request to the Google CSE API.
            
                    Args:
                        query (str): The search query string.
                        num (int, optional): The number of results to return. Defaults to 10.
                        start (int, optional): The index of the first result to return (for pagination). Defaults to 0.
                        hl (str, optional): The language code for the search results (e.g., 'ru' for Russian). Defaults to 'ru'.
                        search_type (str, optional): The type of search results to return (e.g., 'image'). Defaults to None.
            
                    Returns:
                        ServerResponse: The server response object containing the search results and pagination information.
            
                    Raises:
                        GoogleSearchError: If the request to the API fails or returns a non-200 status code.
            
                    The `ServerResponse` object returned contains the following structure:
            
                    - `cursor` (Cursor): An object for managing pagination across search results.
                        - `currentPageIndex` (int): The current page index.
                        - `estimatedResultCount` (Optional[str]): The estimated number of results.
                        - `moreResultsUrl` (str): URL to fetch additional results.
                        - `pages` (Optional[List[Page]]): List of page objects for navigation (optional).
                        - `resultCount` (Optional[str]): The total number of results.
                        - `searchResultTime` (str): Time taken for the search.
            
                    - `findMoreOnGoogle` (Optional[FindMoreOnGoogle]): URL for searching for more results in Google (optional).
                        - `url` (str): The URL for additional search results.
            
                    - `context` (Optional[Context]): Contextual information about the search.
                        - `title` (str): The title of the search.
                        - `total_results` (str): The total number of results found.
            
                    - `results` (Optional[List[Result]]): A list of search result objects (optional).
                        - `content` (str): The description of the result with formatting.
                        - `contentNoFormatting` (str): The description of the result without formatting.
                        - `title` (str): The result's title with formatting.
                        - `titleNoFormatting` (str): The result's title without formatting.
                        - `unescapedUrl` (str): The original unescaped URL of the result.
                        - `url` (str): The result URL.
                        - `visibleUrl` (str): The URL as displayed to the user.
                        - `originalContextUrl` (Optional[str]): The original context URL (optional).
                        - Image-related fields (optional): `height`, `width`, `tbUrl`, `tbMedUrl`, `tbLargeUrl`, etc.
                        - `imageId` (Optional[str]): The ID of the image (if applicable).
                        - `breadcrumbUrl` (Optional[BreadcrumbUrl]): Object with breadcrumb URL information (optional).
                        - `fileFormat` (Optional[str]): The file format of the image (if applicable).
                        - `clicktrackUrl` (Optional[str]): URL for click tracking (if applicable).
                        - `formattedUrl` (Optional[str]): Formatted URL.
                        - `richSnippet` (Optional[RichSnippet]): Extended information about the result, such as images, meta tags, and more.
                    """
                    params = {
                        "q": query,
                        "num": num,
                        "start": start,
                        "hl": hl,
                    }

                    if search_type == "image":
                        params["searchType"] = search_type

                    async with httpx.AsyncClient(
                        base_url=self.api_url,
                        params={"authKey": self.api_token},
                        transport=self.transport,
                        limits=self.limits,
                    ) as client:
                        response = await client.get("/", params=params)
                        response.raise_for_status()

                    return response.json()
            
                def search_sync(
                    self,
                    query: str,
                    num: int = 10,
                    start: int = 0,
                    hl: str = "ru",
                    search_type: Optional[str] = None,
                ) -> dict[str, str]:
                    """
                    Performs a search request to the Google CSE API.
            
                    Args:
                        query (str): The search query string.
                        num (int, optional): The number of results to return. Defaults to 10.
                        start (int, optional): The index of the first result to return (for pagination). Defaults to 0.
                        hl (str, optional): The language code for the search results (e.g., 'ru' for Russian). Defaults to 'ru'.
                        search_type (str, optional): The type of search results to return (e.g., 'image'). Defaults to None.
            
                    Returns:
                        ServerResponse: The server response object containing the search results and pagination information.
            
                    Raises:
                        GoogleSearchError: If the request to the API fails or returns a non-200 status code.
            
                    The `ServerResponse` object returned contains the same structure as described in the `search` method.
                    """
                    params = {
                        "q": query,
                        "num": num,
                        "start": start,
                        "hl": hl,
                    }

                    if search_type == "image":
                        params["searchType"] = search_type

                    with httpx.Client(
                        base_url=self.api_url,
                        params={"authKey": self.api_token},
                        transport=httpx.HTTPTransport(retries=8),
                        limits=self.limits,
                    ) as client:
                        response = client.get("/", params=params)
                        response.raise_for_status()

                    return response.json()


        wrapper = GoogleCSEAPI(
                api_url=self.gcse_endpoint_url,
                api_token=self.gcse_api_key,
            )

        def search_func(
            query: str, page: int = 1, max_results: int = 5, hl: str = "ru", search_type: Optional[str] = None
        ) -> list[dict[str, Any]]:

            raw_results = wrapper.search_sync(
                query, num=max_results, start=max((page - 1) * max_results, 0), hl=hl, search_type=search_type
            )
            return raw_results["results"] if raw_results["results"] else []

        tool = StructuredTool.from_function(
            name="gcse_search_api",
            description="Search for recent results using GCSE with result limiting",
            func=search_func,
            args_schema=self.GCSEAPISchema,
        )

        self.status = "GCSEAPI Tool created"
        return tool

    def run_model(self) -> list[Data]:
        tool = self.build_tool()
        try:
            results = tool.run(
                {
                    "query": self.input_value,
                    "page": self.page,
                    "max_results": self.max_results,
                    "hl": self.hl,
                    "search_type": self.search_type,
                }
            )

            data_list = [Data(data=result, text=result.get("content", "")) for result in results]

        except Exception as e:  # noqa: BLE001
            logger.opt(exception=True).debug("Error running GCSE ")
            self.status = f"Error: {e}"
            return [Data(data={"error": str(e)}, text=str(e))]

        self.status = data_list
        return data_list
