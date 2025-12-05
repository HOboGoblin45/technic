# Technic Brand Guide (Quick Reference)

## Core
- Symbol: angled tile with inner stroke forming an upward blade/arrow (growth + precision).
- Wordmark: “TECHNIC” uppercase, tight letter-spacing; “Personal Quant” as secondary line.
- Voice: confident, concise, mentor-like; avoid hype.

## Palette
- Primary: `#b6ff3b` (neon growth green)
- Secondary: `#5eead4` (aqua accent)
- Text: `#e5e7eb`
- Surfaces: `#02040d` / `#0a1020`
- Positive: `#9ef01a`
- Negative: `#ff6b81`

Use one hero accent (primary) on CTAs/highlights; keep backgrounds neutral/dark for contrast.

## Typography
- Inter / SF Pro / system sans for headings and body.
- Bold for labels/headers; medium/regular for body.

## Assets
- Symbol SVG: `assets/brand/logo-symbol.svg`
- Lockup SVG: `assets/brand/logo-lockup.svg`
- PNG icons: generated in `assets/brand/png/` (64/128/180/256/512)
- Export script: `python scripts/export_icons.py` (Pillow-based; no native deps).

## Usage
- Preferred lockup: symbol + wordmark + tagline on dark background.
- Monochrome: all-white or all-black when color is constrained.
- App icon: symbol only, centered, on dark or neon background.
- Buttons/CTAs: neon gradient; avoid multiple accent colors.

## UI alignment
- The Streamlit theme (`inject_premium_theme`) already uses these tokens and the brand lockup.
- To add a light variant: duplicate the CSS block with light surfaces and swap text to dark; keep primary/secondary accents sparing.
