"""Claude Vision API service for visual brand extraction."""

import json
import os
from typing import Optional

import anthropic

from ..schemas.brand import (
    BrandData,
    BrandColors,
    BrandTypography,
    BrandSpacing,
    BrandButtons,
    ButtonStyle,
)


class VisionService:
    """Async client for Claude Vision API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Vision service.

        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")

        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"

    async def analyze_brand(
        self,
        screenshot_url: str,
        source_url: str,
    ) -> BrandData:
        """Analyze a screenshot to extract brand identity.

        Args:
            screenshot_url: URL to the screenshot image
            source_url: Original website URL (for reference)

        Returns:
            Structured brand data extracted from visual analysis
        """
        prompt = self._build_extraction_prompt()

        message = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": screenshot_url,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        return self._parse_response(message.content[0].text, source_url, screenshot_url)

    def _build_extraction_prompt(self) -> str:
        """Build the prompt for brand extraction."""
        return """Analyze this website screenshot and extract the brand identity.

Return a JSON object with this exact structure:
{
    "colors": {
        "primary": "#HEXCODE",
        "secondary": "#HEXCODE or null",
        "accent": "#HEXCODE or null",
        "background": "#HEXCODE",
        "text": "#HEXCODE"
    },
    "typography": {
        "headings": "font family name",
        "body": "font family name",
        "weights": [400, 600, 700]
    },
    "buttons": {
        "primary": {
            "bg": "#HEXCODE",
            "text": "#HEXCODE",
            "border_radius": "Npx",
            "has_border": false
        },
        "secondary": null
    }
}

GUIDELINES:
1. For colors, extract the EXACT hex values you see visually. Focus on:
   - Primary: The main brand/accent color (buttons, links, key elements)
   - Secondary: A supporting color if present
   - Accent: Highlight/call-to-action color if different from primary
   - Background: Main page background
   - Text: Primary text color

2. For typography, identify font families from visual appearance.
   If you can't determine exact font, describe it (e.g., "sans-serif", "geometric sans-serif")

3. For buttons, analyze the most prominent button style visible.
   Note border radius (rounded corners) and whether it has a border.

4. Focus on the UI chrome (header, buttons, text) not decorative images.

5. If the page has light and dark sections, prioritize the main content area.

6. Return ONLY the JSON object, no additional text or markdown."""

    def _parse_response(
        self,
        response_text: str,
        source_url: str,
        screenshot_url: str,
    ) -> BrandData:
        """Parse Claude's response into BrandData."""
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            return self._default_brand_data(source_url, screenshot_url)

        # Build BrandColors
        colors_data = data.get("colors", {})
        colors = BrandColors(
            primary=colors_data.get("primary", "#000000"),
            secondary=colors_data.get("secondary"),
            accent=colors_data.get("accent"),
            background=colors_data.get("background", "#ffffff"),
            text=colors_data.get("text", "#000000"),
            palette=[
                c for c in [
                    colors_data.get("primary"),
                    colors_data.get("secondary"),
                    colors_data.get("accent"),
                    colors_data.get("background"),
                    colors_data.get("text"),
                ] if c
            ],
        )

        # Build BrandTypography
        typo_data = data.get("typography", {})
        typography = BrandTypography(
            headings=typo_data.get("headings", "sans-serif"),
            body=typo_data.get("body", "sans-serif"),
            weights=typo_data.get("weights", [400, 600, 700]),
        )

        # Build BrandButtons
        buttons = BrandButtons()
        buttons_data = data.get("buttons", {})

        if buttons_data.get("primary"):
            btn = buttons_data["primary"]
            buttons.primary = ButtonStyle(
                bg=btn.get("bg", colors.primary),
                text=btn.get("text", "#ffffff"),
                border_radius=btn.get("border_radius", "4px"),
                padding="12px 24px",
                border=btn.get("border") if btn.get("has_border") else None,
            )

        if buttons_data.get("secondary"):
            btn = buttons_data["secondary"]
            buttons.secondary = ButtonStyle(
                bg=btn.get("bg", "transparent"),
                text=btn.get("text", colors.primary),
                border_radius=btn.get("border_radius", "4px"),
                padding="12px 24px",
                border=btn.get("border") if btn.get("has_border") else None,
            )

        return BrandData(
            url=source_url,
            colors=colors,
            typography=typography,
            spacing=BrandSpacing(),
            buttons=buttons,
            screenshots=[screenshot_url],
        )

    def _default_brand_data(self, source_url: str, screenshot_url: str) -> BrandData:
        """Return default brand data when parsing fails."""
        return BrandData(
            url=source_url,
            colors=BrandColors(primary="#000000"),
            typography=BrandTypography(headings="sans-serif", body="sans-serif"),
            spacing=BrandSpacing(),
            buttons=BrandButtons(),
            screenshots=[screenshot_url],
        )

    async def close(self):
        """Close the client (no-op for anthropic client, but matches pattern)."""
        pass
