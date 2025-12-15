#!/usr/bin/env python3
"""Remove old helper methods from symbol_detail_page.dart"""

import re

# Read the file
with open('technic_app/lib/screens/symbol_detail/symbol_detail_page.dart', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and remove _buildMeritCard method
content = re.sub(
    r'  Widget _buildMeritCard\(SymbolDetail detail\).*?(?=\n  Widget _build|\n  List<Widget> _build|\n  Color _get|\n\})',
    '',
    content,
    flags=re.DOTALL
)

# Find and remove _buildMeritFlags method
content = re.sub(
    r'  List<Widget> _buildMeritFlags\(String flags\).*?(?=\n  Widget _build|\n  Color _get|\n\})',
    '',
    content,
    flags=re.DOTALL
)

# Find and remove _buildPriceChart method
content = re.sub(
    r'  Widget _buildPriceChart\(SymbolDetail detail\).*?(?=\n  Widget _build|\n  Color _get|\n\})',
    '',
    content,
    flags=re.DOTALL
)

# Find and remove _buildCandlestickChart method
content = re.sub(
    r'  Widget _buildCandlestickChart\(List<PricePoint> history\).*?(?=\n  Widget _build|\n  Color _get|\n\})',
    '',
    content,
    flags=re.DOTALL
)

# Find and remove _buildFactorBreakdown method
content = re.sub(
    r'  Widget _buildFactorBreakdown\(SymbolDetail detail\).*?(?=\n  Widget _build|\n  Color _get|\n\})',
    '',
    content,
    flags=re.DOTALL
)

# Find and remove _buildFactorBar method
content = re.sub(
    r'  Widget _buildFactorBar\(String label, double value\).*?(?=\n  Widget _build|\n  Color _get|\n\})',
    '',
    content,
    flags=re.DOTALL
)

# Find and remove _getMeritBandColor method
content = re.sub(
    r'  Color _getMeritBandColor\(String band\).*?(?=\n\})',
    '',
    content,
    flags=re.DOTALL
)

# Find and remove _SimpleLinePainter class
content = re.sub(
    r'/// Simple line chart painter\nclass _SimpleLinePainter extends CustomPainter.*',
    '',
    content,
    flags=re.DOTALL
)

# Write the cleaned content
with open('technic_app/lib/screens/symbol_detail/symbol_detail_page.dart', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Removed old helper methods from symbol_detail_page.dart")
print("✅ File is now using only the new widgets")
