/// Mock Data
/// 
/// Fallback data for offline mode and testing.
library;

import 'package:flutter/material.dart';
import '../models/market_mover.dart';
import '../models/scan_result.dart';
import '../models/idea.dart';
import '../models/scoreboard_slice.dart';
import '../models/copilot_message.dart';
import '../theme/app_colors.dart';
/// Default tickers for search hints and quick access
const defaultTickers = <String>[
  'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN',
  'GOOGL', 'META', 'NFLX', 'AMD', 'INTC',
  'JPM', 'GS', 'XOM', 'CVX', 'BA',
  'CAT', 'LMT', 'KO', 'PEP', 'WMT',
];

/// Mock market movers for fallback display
const mockMovers = <MarketMover>[
  MarketMover(
    'AAPL',
    '+1.3%',
    'Momentum higher',
    true,
    [0.2, 0.25, 0.28, 0.27],
  ),
  MarketMover(
    'NVDA',
    '+2.1%',
    'Breakout pushing',
    true,
    [0.4, 0.42, 0.45, 0.44],
  ),
  MarketMover(
    'TSLA',
    '-0.8%',
    'Cooling after run',
    false,
    [0.5, 0.48, 0.46, 0.47],
  ),
];

/// Mock scan results for fallback display
const mockScanResults = <ScanResult>[
  ScanResult(
    'ADBE',
    'Breakout long',
    'R:R 2.0',
    '598',
    '586',
    '620',
    'Volume +18% vs avg',
    [0.2, 0.25, 0.3, 0.27, 0.35],
  ),
  ScanResult(
    'MSFT',
    'Momentum swing',
    'R:R 2.1',
    '412',
    '404',
    '430',
    'Copilot suggests staggered exits',
    [0.4, 0.42, 0.38, 0.44, 0.5],
  ),
  ScanResult(
    'NVDA',
    'Pullback buy',
    'R:R 1.8',
    '488',
    '472',
    '520',
    'Watching semis breadth',
    [0.3, 0.28, 0.32, 0.35, 0.36],
  ),
];

/// Mock trade ideas for fallback display
const mockIdeas = <Idea>[
  Idea(
    'Breakout long',
    'ADBE',
    'R:R 2.0 | TechRating 82',
    'Entry 598 | Stop 586 | Target 620',
    [0.2, 0.25, 0.3, 0.27, 0.35],
  ),
  Idea(
    'Swing momentum',
    'MSFT',
    'R:R 2.1 | TechRating 79',
    'Entry 412 | Stop 404 | Target 430',
    [0.4, 0.42, 0.38, 0.44, 0.5],
  ),
  Idea(
    'Pullback buy',
    'NVDA',
    'R:R 1.8 | TechRating 76',
    'Entry 488 | Stop 472 | Target 520',
    [0.35, 0.3, 0.32, 0.36, 0.4],
  ),
];

/// Mock scoreboard slices for performance tracking
const scoreboardSlices = <ScoreboardSlice>[
  ScoreboardSlice(
    'Day trades',
    '+4.2% MTM',
    '64% win',
    'Avg hold 4h',
    Color(0xFF5EEAD4),
  ),
  ScoreboardSlice(
    'Swing',
    '+8.5% MTD',
    '58% win',
    'Avg hold 3d',
    AppColors.successGreen,
  ),
  ScoreboardSlice(
    'Long-term',
    '+12.4% YTD',
    '52% win',
    'Avg hold 6m',
    Color(0xFFB0CAFF),
  ),
];

/// Mock Copilot conversation for demonstration
const copilotMessages = <CopilotMessage>[
  CopilotMessage('user', 'What changed in semis today?'),
  CopilotMessage(
    'assistant',
    'Semis led by NVDA (+2.1%) and AMD (+1.6%). Breadth improved across the group with above-average volume. Resistance at 510 is being tested.',
    meta: 'Scan basis: Volume surge +16%, trend > 50d',
  ),
  CopilotMessage('user', 'Is the risk worth it if I enter NVDA now?'),
  CopilotMessage(
    'assistant',
    '- Entry 488, stop 472, target 520 (R:R 1.8)\n'
        '- Key risks: earnings in 10d, semis vol cluster near 500\n'
        '- Suggest staggered exits at 505 / 516 if intraday volatility stays elevated.',
    meta: 'Personalized to Swing profile, risk 1%',
  ),
];

/// Suggested Copilot prompts for quick access
const copilotPrompts = <String>[
  "Summarize today's scan",
  'Explain risk on NVDA setup',
  'Compare TSLA vs AAPL momentum',
  'What moved semis this week?',
];
