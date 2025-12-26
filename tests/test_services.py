"""Tests for service layer."""

import pytest
from unittest.mock import patch, MagicMock

from mirage.schemas.brand import BrandData, BrandColors, BrandTypography


class TestFirecrawlService:
    """Tests for FirecrawlService."""

    def test_requires_api_key(self):
        """Test that service requires API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="FIRECRAWL_API_KEY is required"):
                from mirage.services.firecrawl import FirecrawlService
                FirecrawlService()

    def test_accepts_api_key_parameter(self):
        """Test that API key can be passed as parameter."""
        from mirage.services.firecrawl import FirecrawlService

        service = FirecrawlService(api_key="test-key")
        assert service.api_key == "test-key"

    def test_reads_api_key_from_env(self):
        """Test that API key is read from environment."""
        with patch.dict("os.environ", {"FIRECRAWL_API_KEY": "env-key"}):
            from mirage.services.firecrawl import FirecrawlService

            service = FirecrawlService()
            assert service.api_key == "env-key"


class TestGeminiService:
    """Tests for GeminiService."""

    def test_requires_api_key(self):
        """Test that service requires API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY is required"):
                from mirage.services.gemini import GeminiService
                GeminiService()

    def test_accepts_api_key_parameter(self):
        """Test that API key can be passed as parameter."""
        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel"):
                from mirage.services.gemini import GeminiService

                service = GeminiService(api_key="test-key")
                assert service.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_calculate_color_similarity(self):
        """Test color similarity calculation."""
        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel"):
                from mirage.services.gemini import GeminiService

                service = GeminiService(api_key="test-key")

                # Same color should have similarity 1.0
                similarity = await service.calculate_color_similarity("#FF0000", "#FF0000")
                assert similarity == 1.0

                # Very different colors should have low similarity
                similarity = await service.calculate_color_similarity("#000000", "#FFFFFF")
                assert similarity < 0.5

                # Similar colors should have high similarity
                similarity = await service.calculate_color_similarity("#FF0000", "#FF1111")
                assert similarity > 0.9


class TestBrandDataSchema:
    """Tests for BrandData schema."""

    def test_brand_data_serialization(self):
        """Test that BrandData can be serialized to dict."""
        brand = BrandData(
            url="https://example.com",
            colors=BrandColors(primary="#FF0000"),
            typography=BrandTypography(headings="Arial", body="Helvetica"),
        )

        data = brand.model_dump()
        assert data["url"] == "https://example.com"
        assert data["colors"]["primary"] == "#FF0000"
        assert data["typography"]["headings"] == "Arial"

    def test_brand_data_validation(self):
        """Test that BrandData validates input."""
        # Valid data should work
        brand = BrandData.model_validate({
            "url": "https://example.com",
            "colors": {"primary": "#FF0000"},
            "typography": {"headings": "Arial", "body": "Helvetica"},
        })
        assert brand.url == "https://example.com"


class TestVisionService:
    """Tests for VisionService."""

    def test_requires_api_key(self):
        """Test that service requires API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
                from mirage.services.vision import VisionService
                VisionService()

    def test_accepts_api_key_parameter(self):
        """Test that API key can be passed as parameter."""
        with patch("anthropic.AsyncAnthropic"):
            from mirage.services.vision import VisionService

            service = VisionService(api_key="test-key")
            assert service.api_key == "test-key"

    def test_build_extraction_prompt(self):
        """Test that extraction prompt is well-formed."""
        with patch("anthropic.AsyncAnthropic"):
            from mirage.services.vision import VisionService

            service = VisionService(api_key="test-key")
            prompt = service._build_extraction_prompt()

            assert "colors" in prompt
            assert "typography" in prompt
            assert "buttons" in prompt
            assert "JSON" in prompt
            assert "HEXCODE" in prompt

    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response."""
        with patch("anthropic.AsyncAnthropic"):
            from mirage.services.vision import VisionService

            service = VisionService(api_key="test-key")

            response = '''
            {
                "colors": {
                    "primary": "#FF5A5F",
                    "secondary": "#00A699",
                    "background": "#FFFFFF",
                    "text": "#484848"
                },
                "typography": {
                    "headings": "Circular",
                    "body": "Circular"
                },
                "buttons": {
                    "primary": {
                        "bg": "#FF5A5F",
                        "text": "#FFFFFF",
                        "border_radius": "8px"
                    }
                }
            }
            '''

            result = service._parse_response(
                response,
                "https://example.com",
                "https://screenshot.url/img.png"
            )

            assert result.colors.primary == "#FF5A5F"
            assert result.colors.secondary == "#00A699"
            assert result.typography.headings == "Circular"
            assert result.buttons.primary.bg == "#FF5A5F"
            assert result.url == "https://example.com"
            assert "https://screenshot.url/img.png" in result.screenshots

    def test_parse_response_with_markdown_code_block(self):
        """Test parsing response wrapped in markdown code blocks."""
        with patch("anthropic.AsyncAnthropic"):
            from mirage.services.vision import VisionService

            service = VisionService(api_key="test-key")

            response = '''```json
{
    "colors": {
        "primary": "#FF90E8",
        "background": "#FFFFFF",
        "text": "#000000"
    },
    "typography": {
        "headings": "Mabry",
        "body": "Mabry"
    },
    "buttons": {}
}
```'''

            result = service._parse_response(
                response,
                "https://gumroad.com",
                "https://screenshot.url/img.png"
            )

            assert result.colors.primary == "#FF90E8"
            assert result.typography.headings == "Mabry"

    def test_parse_response_invalid_json_returns_defaults(self):
        """Test that invalid JSON returns default brand data."""
        with patch("anthropic.AsyncAnthropic"):
            from mirage.services.vision import VisionService

            service = VisionService(api_key="test-key")

            result = service._parse_response(
                "not valid json at all",
                "https://example.com",
                "https://screenshot.url/img.png"
            )

            assert result.colors.primary == "#000000"
            assert result.url == "https://example.com"
            assert result.typography.headings == "sans-serif"

    def test_default_brand_data(self):
        """Test default brand data factory."""
        with patch("anthropic.AsyncAnthropic"):
            from mirage.services.vision import VisionService

            service = VisionService(api_key="test-key")

            result = service._default_brand_data(
                "https://example.com",
                "https://screenshot.url/img.png"
            )

            assert result.url == "https://example.com"
            assert result.colors.primary == "#000000"
            assert result.typography.headings == "sans-serif"
            assert result.typography.body == "sans-serif"
            assert "https://screenshot.url/img.png" in result.screenshots
