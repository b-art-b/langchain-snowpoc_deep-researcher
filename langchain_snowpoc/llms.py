import logging
from typing import Any, List, Mapping, Optional
import json

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from snowflake.snowpark.session import Session
from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

logger = logging.getLogger(__name__)


class SQLCortex(BaseChatModel):
    session: Session = None
    model: str = "llama3.1-405b"
    options: dict = {"temperature": 0.7, "max_tokens": 2048}

    @property
    def _llm_type(self) -> str:
        return "sqlcortex"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        formatted_messages = []
        for message in messages:
            msg_dict = {"role": "", "content": message.content}
            if isinstance(message, SystemMessage):
                msg_dict["role"] = "system"
            elif isinstance(message, HumanMessage):
                msg_dict["role"] = "user"
            elif isinstance(message, AIMessage):
                msg_dict["role"] = "assistant"
            else:
                msg_dict["role"] = "user"
            formatted_messages.append(msg_dict)

        message_json = json.dumps(formatted_messages)
        message_dollar = f"$$[{message_json[1:-1]}]$$"
        options_json = json.dumps(self.options)

        sql_stmt = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{self.model}',
            parse_json({message_dollar}),
            parse_json('{options_json}')
        ) as COMPLETION
        """
        completion = self.session.sql(sql_stmt).collect()[0].COMPLETION

        message = AIMessage(content=completion)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"session": self.session, "model": self.model, "options": self.options}
