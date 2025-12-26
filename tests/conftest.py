"""Pytest fixtures for mirage-brandextract-mcp tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mirage.schemas.brand import (
    BrandData,
    BrandColors,
    BrandTypography,
    BrandSpacing,
    BrandButtons,
    ButtonStyle,
    GeneratedCode,
)


@pytest.fixture
def sample_brand_data() -> BrandData:
    """Create sample brand data for testing."""
    return BrandData(
        url="https://example.com",
        colors=BrandColors(
            primary="#FF5A5F",
            secondary="#00A699",
            accent="#FC642D",
            background="#FFFFFF",
            text="#484848",
            palette=["#FF5A5F", "#00A699", "#FC642D", "#FFFFFF", "#484848"],
        ),
        typography=BrandTypography(
            headings="Circular",
            body="Circular",
            weights=[400, 500, 700],
            base_size="16px",
            line_height="1.5",
        ),
        spacing=BrandSpacing(
            grid="8px",
            margins={"sm": "8px", "md": "16px", "lg": "24px"},
            padding={"sm": "8px", "md": "16px", "lg": "24px"},
        ),
        buttons=BrandButtons(
            primary=ButtonStyle(
                bg="#FF5A5F",
                text="#FFFFFF",
                border_radius="8px",
                padding="14px 24px",
            )
        ),
    )


@pytest.fixture
def sample_generated_code() -> GeneratedCode:
    """Create sample generated code for testing."""
    return GeneratedCode(
        html='<div class="hero"><h1>Welcome</h1><button>Get Started</button></div>',
        css=".hero { padding: 40px; } .hero h1 { color: var(--color-primary); }",
        preview_url="data:text/html;base64,AAAA",
        component_type="hero_section",
    )


@pytest.fixture
def mock_firecrawl_service(sample_brand_data):
    """Create a mock FirecrawlService."""
    mock = MagicMock()
    mock.extract_brand = AsyncMock(return_value=sample_brand_data)
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_gemini_service(sample_generated_code):
    """Create a mock GeminiService."""
    mock = MagicMock()
    mock.generate_replica = AsyncMock(return_value=sample_generated_code)
    mock.generate_from_template = AsyncMock(return_value=sample_generated_code)
    mock.calculate_color_similarity = AsyncMock(return_value=0.85)
    return mock


@pytest.fixture
def mock_vision_service(sample_brand_data):
    """Create a mock VisionService."""
    mock = MagicMock()
    mock.analyze_brand = AsyncMock(return_value=sample_brand_data)
    mock.close = AsyncMock()
    return mock
