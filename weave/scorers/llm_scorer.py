from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Optional, Union

from pydantic import Field, field_validator

import weave
from weave.scorers.base_scorer import Scorer
from weave.scorers.llm_utils import (
    _LLM_CLIENTS,
    _LLM_CLIENTS_NAMES,
    instructor_client,
    set_device,
)

if TYPE_CHECKING:
    from torch import Tensor


class LLMScorer(Scorer):
    """Score model outputs using a Large Language Model (LLM).

    This scorer leverages LLMs to evaluate and score model outputs. It provides a flexible
    way to use different LLM providers for scoring purposes.

    Attributes:
        client: An instantiated LLM client with valid API credentials
        model_id: The specific model identifier to use for scoring
    """

    client: _LLM_CLIENTS = Field(
        description="The LLM client to use, has to be instantiated with an api_key"
    )
    model_id: str = Field(description="The model to use")

    @field_validator("client")
    def validate_client(cls, v: _LLM_CLIENTS) -> _LLM_CLIENTS:
        client_type_name = type(v).__name__
        if client_type_name not in _LLM_CLIENTS_NAMES:
            raise ValueError(
                f"Invalid client type. Expected one of {_LLM_CLIENTS_NAMES}, got {client_type_name}"
            )
        return v


class InstructorLLMScorer(Scorer):
    """Score a model using an LLM with structured outputs.

    This scorer extends the base LLM scoring capability by adding temperature and
    token control for more precise scoring behavior. It automatically wraps the
    provided client with [instructor](https://github.com/instructor-ai/instructor)
    functionality for structured outputs.

    Attributes:
        client: An instantiated LLM client with valid API credentials
        model_id: The specific model identifier to use for scoring
        temperature: Controls randomness in the LLM's responses (0.0 to 1.0)
        max_tokens: Maximum number of tokens allowed in the LLM's response
    """

    client: _LLM_CLIENTS = Field(
        description="The LLM client to use, has to be instantiated with an api_key"
    )
    model_id: str = Field(description="The model to use")
    temperature: float = Field(
        ..., description="The temperature to use for the response"
    )
    max_tokens: int = Field(
        ..., description="The maximum number of tokens in the response"
    )

    @field_validator("client")
    def validate_client(cls, v: _LLM_CLIENTS) -> _LLM_CLIENTS:
        client_type_name = type(v).__name__
        if client_type_name not in _LLM_CLIENTS_NAMES:
            raise ValueError(
                f"Invalid client type. Expected one of {_LLM_CLIENTS_NAMES}, got {client_type_name}"
            )
        return instructor_client(v)


class HuggingFacePipelineScorer(Scorer):
    """
    Base class for using Hugging Face pipelines for moderation scoring.

    This class simplifies the use of Hugging Face pipelines by handling the initialization and providing a common interface for scoring.

    Args:
        task (str): The pipeline task type (e.g., `"text-classification"`).
        model_name_or_path (str): The name or path of the model to use.
        device (str): The device to use for inference. Defaults to `"cpu"`.
        pipeline_kwargs (dict[str, Any]): Additional keyword arguments for the pipeline. Defaults to `{}`.

    Returns:
        list[dict[str, Any]]: The pipeline's output after processing the input text.

    Example:
        >>> from weave.scorers.moderation_scorer import PipelineScorer
        >>> scorer = PipelineScorer(
        ...     task="text-classification",
        ...     model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"
        ... )
        >>> output = scorer.pipe("This is a great movie!")
        >>> print(output)
        [{'label': 'POSITIVE', 'score': 0.9998}]
    """

    task: str = Field(
        description="The task to use for the pipeline, for example 'text-classification'"
    )
    model_name_or_path: str = Field(default="", description="The path to the model")
    device: str = Field(default="auto", description="The device to use for the model")
    pipeline_kwargs: dict[str, Any] = Field(default_factory=dict)
    pipeline: Optional[Any] = None

    def model_post_init(self, __context: Any) -> None:
        self.device = set_device(self.device)
        try:
            if find_spec("transformers") is None:
                print(
                    "The `transformers` package is required to use PipelineScorer, please run `pip install transformers`"
                )
        except ImportError:
            print(
                "The `transformers` package is required to use PipelineScorer, please run `pip install transformers`"
            )
        if self.pipeline is None:
            self.set_pipeline()

    def load_pipeline(self) -> None:
        raise NotImplementedError(
            "Subclasses must implement the `load_pipeline` method."
        )

    def pipe(self, prompt: str) -> list[dict[str, Any]]:
        return self.pipeline(prompt)[0]

    @weave.op
    def score(self, *, output: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class HuggingFaceScorer(Scorer):
    """Score model outputs using a Hugging Face model."""

    model_name_or_path: str = Field(default="", description="The path to the model")
    device: str = Field(default="auto", description="The device to use for the model")
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None

    def model_post_init(self, __context: Any = None) -> None:
        """Template method for post-initialization."""
        self.device = set_device(self.device)
        if self.model is None:
            self.load_model()
        else:
            print("Using user-provided model.")

        if self.tokenizer is None:
            self.load_tokenizer()
        else:
            print("Using user-provided tokenizer.")

    def load_model(self) -> None:
        raise NotImplementedError("Subclasses must implement the `load_model` method.")

    def load_tokenizer(self) -> None:
        raise NotImplementedError(
            "Subclasses must implement the `load_tokenizer` method."
        )

    @weave.op
    def score(self, *, output: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class RollingWindowScorer(HuggingFaceScorer):
    """
    Base Scorer class that handles rolling window processing for long inputs.

    Args:
        max_tokens: Maximum number of tokens per window.
        overlap: Number of overlapping tokens between consecutive windows.
        device: The device to use for inference.
        aggregation_method: The method to aggregate predictions ("max" or "average").
    """

    max_tokens: int = 512  # Default maximum tokens per window
    overlap: int = 50
    aggregation_method: str = "max"  # New class attribute for aggregation method

    def tokenize_input(self, prompt: str) -> "Tensor":
        """
        Tokenize the input prompt without truncation.

        Args:
            prompt: The input text to tokenize.

        Returns:
            A tensor of tokenized input IDs.
        """
        return self.tokenizer(
            prompt, return_tensors="pt", truncation=False
        ).input_ids.to(self.device)

    def predict_chunk(self, input_ids: "Tensor") -> list[Union[int, float]]:
        raise NotImplementedError("Subclasses must implement predict_chunk method.")

    def aggregate_predictions(
        self, all_predictions: list[list[Union[int, float]]]
    ) -> list[float]:
        """
        Aggregate predictions using the specified class attribute method.

        Args:
            all_predictions: List of prediction lists from chunks.

        Returns:
            Aggregated prediction scores per category.
        """
        if not all_predictions:
            return []

        num_categories = len(all_predictions[0])
        aggregated = []

        for i in range(num_categories):
            category_scores = [pred[i] for pred in all_predictions]
            if self.aggregation_method == "max":
                aggregated.append(max(category_scores))
            elif self.aggregation_method == "average":
                aggregated.append(sum(category_scores) / len(category_scores))
            else:
                raise ValueError(
                    f"Unsupported aggregation method: {self.aggregation_method}"
                )

        return aggregated

    def predict_long(self, input_ids: "Tensor") -> list[float]:
        """
        Handle prediction for long inputs by processing in overlapping windows.

        Args:
            input_ids: Tokenized input IDs.

        Returns:
            A list of aggregated prediction scores for each category.
        """
        total_tokens: int = input_ids.size(1)

        if total_tokens <= self.max_tokens:
            return self.predict_chunk(input_ids)

        all_predictions: list[list[float]] = []
        stride: int = self.max_tokens - self.overlap

        for i in range(0, total_tokens - self.overlap, stride):
            chunk_input_ids = input_ids[:, i : i + self.max_tokens]
            chunk_predictions = self.predict_chunk(chunk_input_ids)
            all_predictions.append(chunk_predictions)
        # Aggregate predictions using the specified aggregation method
        final_predictions: list[float] = self.aggregate_predictions(all_predictions)

        return final_predictions

    def predict(self, prompt: str) -> list[float]:
        """
        Predict scores for the input prompt, handling long inputs if necessary.

        Args:
            prompt (str): The input text to evaluate.

        Returns:
            list[float]: A list of prediction scores for each category.

        Example:
            >>> scorer = RollingWindowScorer()
            >>> predictions = scorer.predict("Some long input text...")
            >>> print(predictions)
            [0.5, 0.3, 0.0, 0.2, 0.7]
        """
        input_ids: Tensor = self.tokenize_input(prompt)
        return self.predict_long(input_ids)
