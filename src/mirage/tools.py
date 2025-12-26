"""MCP tool implementations for brand extraction and replication."""

from typing import Optional

from mcp.server.fastmcp import FastMCP

from .schemas.brand import BrandData, GeneratedCode, BrandComparison, ComparisonMetrics
from .services.firecrawl import FirecrawlService
from .services.gemini import GeminiService
from .services.vision import VisionService


def register_tools(mcp: FastMCP) -> None:
    """Register all brand extraction tools with the MCP server."""

    @mcp.tool()
    async def extract_brand(url: str, include_screenshots: bool = False) -> dict:
        """Extract complete brand identity from a website.

        Use when you need to analyze a website's visual design system including
        colors, typography, spacing, and button styles.

        Args:
            url: Target website URL to extract brand from
            include_screenshots: Include visual screenshots in the response

        Returns:
            Brand data including colors, typography, spacing, and buttons
        """
        firecrawl = FirecrawlService()
        try:
            brand_data = await firecrawl.extract_brand(url, include_screenshots)
            return brand_data.model_dump()
        finally:
            await firecrawl.close()

    @mcp.tool()
    async def generate_replica(
        brand_data: dict,
        component_type: str = "landing_page",
        customization: str = "",
    ) -> dict:
        """Generate HTML/CSS using extracted brand data.

        Use when you have brand data and want to create matching components.

        Args:
            brand_data: Output from extract_brand tool (dict with colors, typography, etc.)
            component_type: What to generate - one of: landing_page, email, button, card
            customization: Additional instructions for generation

        Returns:
            Generated HTML and CSS code with preview URL
        """
        # Parse brand data back into Pydantic model
        brand = BrandData.model_validate(brand_data)

        gemini = GeminiService()
        result = await gemini.generate_replica(brand, component_type, customization)
        return result.model_dump()

    @mcp.tool()
    async def replicate_website(
        url: str,
        component_type: str = "landing_page",
        customization: str = "",
    ) -> dict:
        """Complete workflow: extract brand + generate replica in one step.

        Use when you want to create a component matching a website without
        needing the intermediate brand data.

        Args:
            url: Target website URL
            component_type: What to generate - one of: landing_page, email, button, card
            customization: Additional instructions

        Returns:
            Both brand_data and generated HTML/CSS
        """
        firecrawl = FirecrawlService()
        try:
            brand_data = await firecrawl.extract_brand(url, include_screenshots=False)
        finally:
            await firecrawl.close()

        gemini = GeminiService()
        generated = await gemini.generate_replica(brand_data, component_type, customization)

        return {
            "brand_data": brand_data.model_dump(),
            "generated": generated.model_dump(),
        }

    @mcp.tool()
    async def compare_brands(url1: str, url2: str) -> dict:
        """Extract and compare brand identities from two websites.

        Use for competitive analysis, brand consistency checks, or understanding
        visual differences between two sites.

        Args:
            url1: First website URL
            url2: Second website URL

        Returns:
            Brand data for both sites with similarity scores and differences
        """
        firecrawl = FirecrawlService()
        gemini = GeminiService()

        try:
            # Extract both brands
            brand1 = await firecrawl.extract_brand(url1)
            brand2 = await firecrawl.extract_brand(url2)
        finally:
            await firecrawl.close()

        # Calculate color similarity
        color_similarity = await gemini.calculate_color_similarity(
            brand1.colors.primary,
            brand2.colors.primary,
        )

        # Check typography match
        typography_match = (
            brand1.typography.headings.lower() == brand2.typography.headings.lower()
            or brand1.typography.body.lower() == brand2.typography.body.lower()
        )

        # Find font overlap
        fonts1 = {brand1.typography.headings.lower(), brand1.typography.body.lower()}
        fonts2 = {brand2.typography.headings.lower(), brand2.typography.body.lower()}
        font_overlap = list(fonts1 & fonts2)

        # Identify differences
        differences = []
        if brand1.colors.primary != brand2.colors.primary:
            differences.append(f"Primary color: {brand1.colors.primary} vs {brand2.colors.primary}")
        if brand1.typography.headings.lower() != brand2.typography.headings.lower():
            differences.append(f"Heading font: {brand1.typography.headings} vs {brand2.typography.headings}")
        if brand1.typography.body.lower() != brand2.typography.body.lower():
            differences.append(f"Body font: {brand1.typography.body} vs {brand2.typography.body}")

        comparison = BrandComparison(
            site1=brand1,
            site2=brand2,
            comparison=ComparisonMetrics(
                color_similarity=color_similarity,
                typography_match=typography_match,
                font_overlap=font_overlap,
                differences=differences,
            ),
        )

        return comparison.model_dump()

    @mcp.tool()
    async def apply_brand_to_template(url: str, template_type: str) -> dict:
        """Extract brand and apply to common templates.

        Use when you want pre-built components styled with a specific brand.

        Args:
            url: Source website for branding
            template_type: Template to apply - one of: hero_section, pricing_table,
                          feature_grid, testimonial, cta

        Returns:
            Generated HTML and CSS for the template with applied branding
        """
        firecrawl = FirecrawlService()
        try:
            brand_data = await firecrawl.extract_brand(url)
        finally:
            await firecrawl.close()

        gemini = GeminiService()
        result = await gemini.generate_from_template(brand_data, template_type)

        return {
            "html": result.html,
            "css": result.css,
            "preview_url": result.preview_url,
            "template_type": template_type,
            "source_url": url,
        }

    @mcp.tool()
    async def extract_brand_visual(url: str) -> dict:
        """Extract brand identity using visual analysis (Claude Vision).

        More accurate and deterministic than text-based extraction, but slower
        and more expensive. Uses screenshot analysis to extract exact colors.

        Args:
            url: Target website URL to extract brand from

        Returns:
            Brand data including colors, typography, and button styles
        """
        firecrawl = FirecrawlService()
        vision = VisionService()
        try:
            # Get screenshot via Firecrawl
            scrape_result = await firecrawl.scrape(url, include_screenshot=True)
            screenshot_url = scrape_result.get("data", {}).get("screenshot")

            if not screenshot_url:
                raise ValueError("Failed to capture screenshot")

            # Analyze with Claude Vision
            brand_data = await vision.analyze_brand(screenshot_url, url)
            return brand_data.model_dump()
        finally:
            await firecrawl.close()
            await vision.close()

    @mcp.tool()
    async def replicate_website_visual(
        url: str,
        component_type: str = "landing_page",
        customization: str = "",
    ) -> dict:
        """Complete workflow using visual analysis: extract brand + generate replica.

        Uses Claude Vision for accurate brand extraction, then generates matching
        HTML/CSS components. More accurate than replicate_website but slower.

        Args:
            url: Target website URL
            component_type: What to generate - one of: landing_page, email, button, card
            customization: Additional instructions for generation

        Returns:
            Both brand_data and generated HTML/CSS
        """
        firecrawl = FirecrawlService()
        vision = VisionService()
        try:
            # Get screenshot via Firecrawl
            scrape_result = await firecrawl.scrape(url, include_screenshot=True)
            screenshot_url = scrape_result.get("data", {}).get("screenshot")

            if not screenshot_url:
                raise ValueError("Failed to capture screenshot")

            # Analyze with Claude Vision
            brand_data = await vision.analyze_brand(screenshot_url, url)
        finally:
            await firecrawl.close()
            await vision.close()

        gemini = GeminiService()
        generated = await gemini.generate_replica(brand_data, component_type, customization)

        return {
            "brand_data": brand_data.model_dump(),
            "generated": generated.model_dump(),
        }
