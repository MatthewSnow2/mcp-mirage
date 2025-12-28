# mcp-mirage

An MCP (Model Context Protocol) server that extracts brand identity from any website and recreates it programmatically.

## Features

- **Extract Brand** - Analyze any website to extract colors, typography, spacing, and button styles
- **Generate Replica** - Create HTML/CSS components using extracted brand data
- **Replicate Website** - One-step extraction and generation
- **Compare Brands** - Compare brand identities between two websites
- **Apply to Template** - Generate pre-built templates with extracted branding

## Requirements

- Python 3.11+
- Firecrawl API key ([get one here](https://firecrawl.dev))
- Google Generative AI API key ([get one here](https://aistudio.google.com/apikey))

## Installation

### Using uv (recommended)

```bash
cd mcp-mirage
uv pip install -e .
```

### Using pip

```bash
cd mcp-mirage
pip install -e .
```

## Configuration

1. Copy the environment template:

```bash
cp .env.example .env
```

2. Add your API keys to `.env`:

```
FIRECRAWL_API_KEY=your_firecrawl_key
GOOGLE_API_KEY=your_google_key
```

## Usage

### Running the Server

```bash
# With uv
uv run python server.py

# Or directly
python server.py
```

### Testing with MCP Inspector

```bash
mcp dev server.py
```

### Adding to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mirage-brandextract": {
      "command": "python",
      "args": ["/path/to/mirage-brandextract-mcp/server.py"],
      "env": {
        "FIRECRAWL_API_KEY": "your_key",
        "GOOGLE_API_KEY": "your_key"
      }
    }
  }
}
```

## Tools

### `extract_brand`

Extract complete brand identity from a website.

```python
# Parameters
url: str                      # Target website URL
include_screenshots: bool     # Include visual screenshots (default: False)

# Returns
{
    "colors": {
        "primary": "#FF5A5F",
        "secondary": "#00A699",
        "palette": [...]
    },
    "typography": {
        "headings": "Circular",
        "body": "Circular",
        "weights": [400, 500, 700]
    },
    "spacing": {...},
    "buttons": {...},
    "url": "original_url"
}
```

### `generate_replica`

Generate HTML/CSS using extracted brand data.

```python
# Parameters
brand_data: dict          # Output from extract_brand
component_type: str       # "landing_page", "email", "button", "card"
customization: str        # Additional instructions

# Returns
{
    "html": "<div>...</div>",
    "css": "/* styles */",
    "preview_url": "data:text/html;base64,..."
}
```

### `replicate_website`

Complete workflow: extract brand + generate replica in one step.

```python
# Parameters
url: str              # Target website URL
component_type: str   # What to generate
customization: str    # Additional instructions

# Returns
{
    "brand_data": {...},
    "generated": {
        "html": "...",
        "css": "..."
    }
}
```

### `compare_brands`

Compare brand identities from two websites.

```python
# Parameters
url1: str    # First website URL
url2: str    # Second website URL

# Returns
{
    "site1": {...brand_data...},
    "site2": {...brand_data...},
    "comparison": {
        "color_similarity": 0.85,
        "typography_match": true,
        "font_overlap": ["Inter"],
        "differences": [...]
    }
}
```

### `apply_brand_to_template`

Apply brand to pre-built templates.

```python
# Parameters
url: str              # Source website for branding
template_type: str    # "hero_section", "pricing_table", "feature_grid", "testimonial", "cta"

# Returns
{
    "html": "...",
    "css": "...",
    "preview_url": "...",
    "template_type": "hero_section"
}
```

## Example Use Cases

### 1. Extract Airbnb branding and create a hero section

```
extract_brand(url="https://airbnb.com")
generate_replica(brand_data=result, component_type="landing_page", customization="Focus on hero section")
```

### 2. Compare competitor websites

```
compare_brands(url1="https://airbnb.com", url2="https://vrbo.com")
```

### 3. Quick template generation

```
apply_brand_to_template(url="https://stripe.com", template_type="pricing_table")
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

## License

MIT

---

*Built autonomously by [GRIMLOCK](https://github.com/MatthewSnow2/grimlock) - Autonomous MCP Server Factory*
