from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Callable, List, Optional

from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader
from snowflake.snowpark.session import Session

logger = logging.getLogger(__name__)


class SnowflakeStageFileLoader(UnstructuredBaseLoader):
    """Load from `Snowflake Stage` file."""

    def __init__(
        self,
        staged_file_path: str,
        *,
        session: Session,
        mode: str = "single",
        post_processors: Optional[List[Callable]] = None,
        **unstructured_kwargs: Any,
    ):
        """Initialize with bucket and key name.

        :param mode: Mode in which to read the file. Valid options are: single,
            paged and elements.
        :param post_processors: Post processing functions to be applied to
            extracted elements.
        :param **unstructured_kwargs: Arbitrary additional kwargs to pass in when
            calling `partition`
        """
        super().__init__(mode, post_processors, **unstructured_kwargs)
        self.staged_file_path = staged_file_path
        self.session = session
        self._meta = None

    def _get_elements(self) -> List:
        """Get elements."""
        from unstructured.partition.auto import partition

        local_file_name = os.path.basename(self.staged_file_path)

        self._meta = (
            self.session.sql(f"LIST '{self.staged_file_path}'").collect()[0].as_dict()
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{local_file_name}"

            os.makedirs(temp_dir, exist_ok=True)

            self.session.file.get(self.staged_file_path, temp_dir)

            return partition(filename=file_path, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        return {"source": f"{self.staged_file_path}", **self._meta}
