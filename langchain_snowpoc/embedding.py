import logging
from typing import Any, List, Mapping, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from snowflake.snowpark.session import Session

logger = logging.getLogger(__name__)


class SnowflakeEmbeddings(BaseModel, Embeddings):
    """Snowflake runs large language models.

    Example:
        .. code-block:: python

            from langchain_snowpoc.embeddings import SnowflakeEmbeddings
            sf_emb = SnowflakeEmbeddings(
                model="e5-base-v2",
            )
            r1 = sf_emb.embed_documents(
                [
                    "Alpha is the first letter of Greek alphabet",
                    "Beta is the second letter of Greek alphabet",
                ]
            )
            r2 = sf_emb.embed_query(
                "What is the second letter of Greek alphabet"
            )

    """

    session: Session = None
    """Snowflake sesison to use"""

    model: str = "e5-base-v2"
    """Model name to use."""

    show_progress: bool = False
    """Whether to show a tqdm progress bar. Must have `tqdm` installed."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model": self.model, "session": self.session},
            **self._default_params,
        }

    def _process_emb_response(self, input: str) -> List[float]:
        """Process a response from Snowflake.

        Args:
            response: The response from Snowflake.

        Returns:
            The response as a dictionary.
        """
        q = f"SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('{self.model}', $${input}$$) as EMBEDDING"

        return self.session.sql(q).collect()[0].EMBEDDING

    def _embed(self, input: List[str]) -> List[List[float]]:
        if self.show_progress:
            try:
                from tqdm import tqdm

                iter_ = tqdm(input, desc="SnowflakeEmbeddings")
            except ImportError:
                logger.warning(
                    "Unable to show progress bar because tqdm could not be imported. "
                    "Please install with `pip install tqdm`."
                )
                iter_ = input
        else:
            iter_ = input
        return [self._process_emb_response(prompt) for prompt in iter_]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Snowflake's embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        instruction_pairs = [text for text in texts]
        embeddings = self._embed(instruction_pairs)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using Snowfake embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self._embed([text])[0]
        return embedding
