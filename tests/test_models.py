"""
Unit tests for models.py - Pydantic models validation.
"""

import pytest
from pydantic import ValidationError

from models import ChatGPTImageGeneration


class TestChatGPTImageGeneration:
    """Tests for ChatGPTImageGeneration model."""

    def test_valid_model(self):
        """Should create model with valid data."""
        data = {
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "id": "gen_123",
            "conversation_id": "conv_123",
            "message_id": "msg_123",
            "asset_pointer": "asset_123",
            "url": "https://example.com/image.png",
            "prompt": "A test prompt",
        }
        model = ChatGPTImageGeneration(**data)
        assert model.id == "gen_123"
        assert model.prompt == "A test prompt"

    def test_missing_required_field(self):
        """Should raise ValidationError for missing required fields."""
        data = {
            "id": "gen_123",
            # Missing other required fields
        }
        with pytest.raises(ValidationError):
            ChatGPTImageGeneration(**data)

    def test_extra_fields_forbidden(self):
        """Should raise ValidationError for extra fields."""
        data = {
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "id": "gen_123",
            "conversation_id": "conv_123",
            "message_id": "msg_123",
            "asset_pointer": "asset_123",
            "url": "https://example.com/image.png",
            "prompt": "A test prompt",
            "extra_field": "not allowed",
        }
        with pytest.raises(ValidationError, match="extra_field"):
            ChatGPTImageGeneration(**data)

    def test_empty_prompt_allowed(self):
        """Empty prompt should be allowed (has default)."""
        data = {
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "id": "gen_123",
            "conversation_id": "conv_123",
            "message_id": "msg_123",
            "asset_pointer": "asset_123",
            "url": "https://example.com/image.png",
        }
        model = ChatGPTImageGeneration(**data)
        assert model.prompt == ""

    def test_wrong_type(self):
        """Should raise ValidationError for wrong types."""
        data = {
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "id": 123,  # Should be string
            "conversation_id": "conv_123",
            "message_id": "msg_123",
            "asset_pointer": "asset_123",
            "url": "https://example.com/image.png",
            "prompt": "A test prompt",
        }
        with pytest.raises(ValidationError):
            ChatGPTImageGeneration(**data)

class TestImageGenerationProtocol:
    """Tests for ImageGeneration protocol compatibility."""

    def test_chatgpt_has_required_attributes(self):
        """ChatGPT generation should have id and prompt."""
        data = {
            "created_at": "2024-01-15T10:30:00.000000+00:00",
            "id": "gen_123",
            "conversation_id": "conv_123",
            "message_id": "msg_123",
            "asset_pointer": "asset_123",
            "url": "https://example.com/image.png",
            "prompt": "Test",
        }
        gen = ChatGPTImageGeneration(**data)
        assert hasattr(gen, "id")
        assert hasattr(gen, "prompt")
        assert gen.id == "gen_123"
        assert gen.prompt == "Test"

    def test_chatgpt_model_compatible(self):
        """ChatGPT model should match the image-generation protocol."""
        chatgpt = ChatGPTImageGeneration(
            created_at="2024-01-15T10:30:00.000000+00:00",
            id="gen_1",
            conversation_id="conv",
            message_id="msg",
            asset_pointer="asset",
            url="http://example.com",
            prompt="test",
        )
        assert hasattr(chatgpt, "id")
        assert hasattr(chatgpt, "prompt")
        assert isinstance(chatgpt.id, str)
        assert isinstance(chatgpt.prompt, str)
