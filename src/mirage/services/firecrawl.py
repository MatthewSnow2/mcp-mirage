"""Firecrawl API service for web scraping and brand extraction."""

import os
from typing import Optional

import httpx

from ..schemas.brand import (
    BrandData,
    BrandColors,
    BrandTypography,
    BrandSpacing,
    BrandButtons,
    ButtonStyle,
)


class FirecrawlService:
    """Async client for Firecrawl API."""

    BASE_URL = "https://api.firecrawl.dev/v1"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Firecrawl service.

        Args:
            api_key: Firecrawl API key. If not provided, reads from FIRECRAWL_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY is required")

        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def scrape(self, url: str, include_screenshot: bool = False) -> dict:
        """Scrape a URL using Firecrawl.

        Args:
            url: The URL to scrape
            include_screenshot: Whether to include a screenshot

        Returns:
            Raw scrape response from Firecrawl
        """
        payload = {
            "url": url,
            "formats": ["html", "markdown"],
        }

        if include_screenshot:
            payload["formats"].append("screenshot")

        response = await self.client.post("/scrape", json=payload)
        response.raise_for_status()
        return response.json()

    async def extract_brand(self, url: str, include_screenshots: bool = False) -> BrandData:
        """Extract brand identity from a website.

        Args:
            url: The website URL to analyze
            include_screenshots: Whether to capture screenshots

        Returns:
            Structured brand data
        """
        # Use Firecrawl's dedicated branding format
        payload = {
            "url": url,
            "formats": ["branding"],
        }

        if include_screenshots:
            payload["formats"].append("screenshot")

        response = await self.client.post("/scrape", json=payload)
        response.raise_for_status()
        data = response.json()

        # Parse the branding data from Firecrawl's dedicated endpoint
        branding = data.get("data", {}).get("branding", {})
        screenshots = []

        if include_screenshots and "screenshot" in data.get("data", {}):
            screenshots.append(data["data"]["screenshot"])

        # Extract colors from branding response
        colors_data = branding.get("colors", {})
        colors = BrandColors(
            primary=colors_data.get("primary", "#000000"),
            secondary=colors_data.get("secondary"),
            accent=colors_data.get("accent"),
            background=colors_data.get("background"),
            text=colors_data.get("textPrimary"),  # Note: Firecrawl uses "textPrimary"
            palette=list(colors_data.values()) if colors_data else [],
        )

        # Extract typography from branding response
        typo_data = branding.get("typography", {})
        font_families = typo_data.get("fontFamilies", {})
        font_weights = typo_data.get("fontWeights", {})
        
        typography = BrandTypography(
            headings=font_families.get("heading", font_families.get("primary", "sans-serif")),
            body=font_families.get("primary", "sans-serif"),
            weights=list(font_weights.values()) if font_weights else [400, 600, 700],
            base_size=typo_data.get("fontSizes", {}).get("body", "16px"),
        )

        # Extract spacing from branding response
        spacing_data = branding.get("spacing", {})
        spacing = BrandSpacing(
            grid=f"{spacing_data.get('baseUnit', 8)}px",
        )
        if spacing_data.get("borderRadius"):
            spacing.gap = spacing_data.get("borderRadius")

        # Extract button styles from branding response
        components = branding.get("components", {})
        buttons = BrandButtons()
        
        if components.get("buttonPrimary"):
            btn = components["buttonPrimary"]
            buttons.primary = ButtonStyle(
                bg=btn.get("background", colors.primary),
                text=btn.get("textColor", "#ffffff"),
                border_radius=btn.get("borderRadius", "4px"),
                padding="12px 24px",  # Firecrawl doesn't always return padding
            )
        
        if components.get("buttonSecondary"):
            btn = components["buttonSecondary"]
            buttons.secondary = ButtonStyle(
                bg=btn.get("background", "transparent"),
                text=btn.get("textColor", colors.primary),
                border_radius=btn.get("borderRadius", "4px"),
                padding="12px 24px",
                border=btn.get("borderColor"),
            )

        # Extract images/logos
        images = branding.get("images", {})

        return BrandData(
            url=url,
            colors=colors,
            typography=typography,
            spacing=spacing,
            buttons=buttons,
            logo_url=images.get("logo") or branding.get("logo"),
            favicon_url=images.get("favicon"),
            screenshots=screenshots,
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()