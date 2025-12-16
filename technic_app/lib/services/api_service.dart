/// Technic API Service
/// 
/// Handles all HTTP communication with the Technic backend API.
/// Provides methods for scanning, fetching ideas, copilot interactions, etc.
library;

import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../models/scan_result.dart';
import '../models/market_mover.dart';
import '../models/idea.dart';
import '../models/scoreboard_slice.dart';
import '../models/scanner_bundle.dart';
import '../models/copilot_message.dart';
import '../models/universe_stats.dart';
import '../models/symbol_detail.dart';

/// API Configuration
class ApiConfig {
  final String baseUrl;
  final String moversPath;
  final String scanPath;
  final String ideasPath;
  final String scoreboardPath;
  final String copilotPath;
  final String universeStatsPath;
  final String symbolPath;

  const ApiConfig({
    required this.baseUrl,
    required this.moversPath,
    required this.scanPath,
    required this.ideasPath,
    required this.scoreboardPath,
    required this.copilotPath,
    required this.universeStatsPath,
    required this.symbolPath,
  });

  factory ApiConfig.fromEnv() {
    final rawBase = const String.fromEnvironment(
      'TECHNIC_API_BASE',
      defaultValue: 'https://technic-m5vn.onrender.com',
    );
    final normalizedBase = _normalizeBaseForPlatform(rawBase);
    
    return ApiConfig(
      baseUrl: normalizedBase,
      moversPath: const String.fromEnvironment(
        'TECHNIC_API_MOVERS',
        defaultValue: '/v1/scan',
      ),
      scanPath: const String.fromEnvironment(
        'TECHNIC_API_SCANNER',
        defaultValue: '/v1/scan',
      ),
      ideasPath: const String.fromEnvironment(
        'TECHNIC_API_IDEAS',
        defaultValue: '/v1/scan',
      ),
      scoreboardPath: const String.fromEnvironment(
        'TECHNIC_API_SCOREBOARD',
        defaultValue: '/v1/scan',
      ),
      copilotPath: const String.fromEnvironment(
        'TECHNIC_API_COPILOT',
        defaultValue: '/v1/copilot',
      ),
      universeStatsPath: const String.fromEnvironment(
        'TECHNIC_API_UNIVERSE',
        defaultValue: '/v1/universe_stats',
      ),
      symbolPath: const String.fromEnvironment(
        'TECHNIC_API_SYMBOL',
        defaultValue: '/v1/symbol',
      ),
    );
  }

  Uri _uri(String path) {
    if (path.startsWith('http')) return Uri.parse(path);
    return Uri.parse('$baseUrl$path');
  }

  Uri moversUri() => _uri(moversPath);
  Uri scanUri() => _uri(scanPath);
  Uri ideasUri() => _uri(ideasPath);
  Uri scoreboardUri() => _uri(scoreboardPath);
  Uri copilotUri() => _uri(copilotPath);
  Uri universeStatsUri() => _uri(universeStatsPath);
  Uri symbolUri() => _uri(symbolPath);
}

/// Normalize base URL for Android emulator
String _normalizeBaseForPlatform(String base) {
  final isAndroid = !kIsWeb && defaultTargetPlatform == TargetPlatform.android;
  if (isAndroid) {
    const localHosts = {
      'http://localhost:8501',
      'http://127.0.0.1:8501',
      'http://localhost:8502',
      'http://127.0.0.1:8502',
    };
    if (localHosts.contains(base)) {
      final port = base.split(':').last;
      return 'http://10.0.2.2:$port';
    }
  }
  return base;
}

/// Scan results payload with progress
class ScanResultsPayload {
  final List<ScanResult> results;
  final String? progress;
  
  const ScanResultsPayload(this.results, this.progress);
}

/// Main API Service
class ApiService {
  ApiService({http.Client? client, ApiConfig? config})
      : _client = client ?? http.Client(),
        _config = config ?? ApiConfig.fromEnv();

  final http.Client _client;
  final ApiConfig _config;

  /// Fetch complete scanner bundle (movers + results + scoreboard)
  /// 
  /// The FastAPI /v1/scan endpoint expects POST with JSON body
  Future<ScannerBundle> fetchScannerBundle({
    Map<String, String>? params,
  }) async {
    try {
      // Build request body from params
      final body = <String, dynamic>{
        'max_symbols': int.tryParse(params?['max_symbols'] ?? '6000') ?? 6000,
        'trade_style': params?['trade_style'] ?? 'Short-term swing',
        'min_tech_rating': double.tryParse(params?['min_tech_rating'] ?? '0.0') ?? 0.0,
      };
      
      // Add sector filter if provided (comma-separated list)
      if (params?['sector'] != null && params!['sector']!.isNotEmpty) {
        final sectorList = params['sector']!.split(',').map((s) => s.trim()).toList();
        body['sectors'] = sectorList;
        debugPrint('[API] Sending sectors: $sectorList');
      } else {
        debugPrint('[API] No sectors in params. params: $params');
      }
      
      // Add lookback_days if provided
      if (params?['lookback_days'] != null) {
        body['lookback_days'] = int.tryParse(params!['lookback_days']!) ?? 90;
      }
      
      // Add options_mode if provided
      if (params?['options_mode'] != null) {
        body['options_mode'] = params!['options_mode'];
      }
      
      debugPrint('[API] Final request body: $body');
      debugPrint('[API] Sending request to: ${_config.scanUri()}');
      
      final res = await _client.post(
        _config.scanUri(),
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: jsonEncode(body),
      ).timeout(
        const Duration(minutes: 3), // 3 minute timeout for full scans
        onTimeout: () {
          debugPrint('[API] Request timed out after 3 minutes');
          throw TimeoutException('Scan request timed out. Try reducing max_symbols or check your connection.');
        },
      );
      
      debugPrint('[API] Response status: ${res.statusCode}');
      
      if (res.statusCode >= 200 && res.statusCode < 300) {
        final decoded = _decode(res.body);
        
        if (decoded is Map<String, dynamic>) {
          // Parse results
          final resultsList = decoded['results'] as List<dynamic>? ?? [];
          final scanResults = resultsList
              .map((e) => ScanResult.fromJson(Map<String, dynamic>.from(e as Map)))
              .toList();
          
          // Parse movers
          final moversList = decoded['movers'] as List<dynamic>? ?? [];
          final movers = moversList
              .map((e) => MarketMover.fromJson(Map<String, dynamic>.from(e as Map)))
              .toList();
          
          // Parse ideas (convert to scoreboard slices for now)
          final ideasList = decoded['ideas'] as List<dynamic>? ?? [];
          final scoreboard = ideasList.take(3).map((e) {
            final idea = Map<String, dynamic>.from(e as Map);
            return ScoreboardSlice(
              idea['title'] ?? 'Idea',
              idea['meta'] ?? '',
              '',
              idea['plan'] ?? '',
              Color(0xFF99BFFF),
            );
          }).toList();
          
          final progress = decoded['log']?.toString();
          
          return ScannerBundle(
            scanResults: scanResults,
            movers: movers,
            scoreboard: scoreboard,
            progress: progress,
          );
        }
      }
    } catch (e) {
      debugPrint('API error: $e');
    }
    
    // Fallback to empty bundle
    return const ScannerBundle(
      scanResults: [],
      movers: [],
      scoreboard: [],
      progress: null,
    );
  }

  /// Fetch market movers
  Future<List<MarketMover>> fetchMovers({
    Map<String, String>? params,
  }) async {
    return _getList(
      _config.moversUri(),
      (json) => MarketMover.fromJson(json),
      params: params,
    );
  }

  /// Fetch scan results with optional progress
  Future<ScanResultsPayload> fetchScanResults({
    Map<String, String>? params,
  }) async {
    final targetUri = params == null
        ? _config.scanUri()
        : _config.scanUri().replace(queryParameters: {
            ..._config.scanUri().queryParameters,
            ...params,
          });
    
    final res = await _client.get(
      targetUri,
      headers: {'Accept': 'application/json'},
    );
    
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final decoded = _decode(res.body);
      
      // Handle {"results": [...], "progress": "..."} shape
      if (decoded is Map<String, dynamic>) {
        final list = decoded['results'] as List<dynamic>? ?? [];
        final progress = decoded['progress']?.toString();
        final parsed = list
            .map((e) => ScanResult.fromJson(Map<String, dynamic>.from(e as Map)))
            .toList();
        return ScanResultsPayload(parsed, progress);
      }
      
      // Fallback: assume bare list
      if (decoded is List) {
        final parsed = decoded
            .map((e) => ScanResult.fromJson(Map<String, dynamic>.from(e as Map)))
            .toList();
        return ScanResultsPayload(parsed, null);
      }
    }
    
    return const ScanResultsPayload([], null);
  }

  /// Fetch scan results for a specific symbol
  Future<List<ScanResult>> fetchSymbolScan(String symbol) async {
    final uri = _config.symbolUri().replace(
      queryParameters: {
        ..._config.symbolUri().queryParameters,
        'symbol': symbol,
      },
    );
    
    final res = await _client.get(uri, headers: {'Accept': 'application/json'});
    
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final decoded = _decode(res.body);
      List<dynamic>? list;
      
      if (decoded is List) {
        list = decoded;
      } else if (decoded is Map && decoded['results'] is List) {
        list = decoded['results'] as List;
      }
      
      if (list != null) {
        return list
            .map((e) => ScanResult.fromJson(Map<String, dynamic>.from(e as Map)))
            .toList();
      }
    }
    
    return [];
  }

  /// Fetch trade ideas
  Future<List<Idea>> fetchIdeas({Map<String, String>? params}) async {
    return _getList(
      _config.ideasUri(),
      (json) => Idea.fromJson(json),
      params: params,
    );
  }

  /// Fetch scoreboard performance data
  Future<List<ScoreboardSlice>> fetchScoreboard() async {
    return _getList(
      _config.scoreboardUri(),
      (json) => ScoreboardSlice.fromJson(json),
    );
  }

  /// Fetch universe statistics (sectors, subindustries)
  Future<UniverseStats?> fetchUniverseStats() async {
    final res = await _client.get(
      _config.universeStatsUri(),
      headers: {'Accept': 'application/json'},
    );
    
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final decoded = _decode(res.body);
      if (decoded is Map<String, dynamic>) {
        return UniverseStats.fromJson(decoded);
      }
    }
    
    return null;
  }

  /// Send a message to Copilot AI
  Future<CopilotMessage> sendCopilot(
    String prompt, {
    String? symbol,
    String? optionsMode,
  }) async {
    final res = await _client.post(
      _config.copilotUri(),
      headers: {
        'Content-Type': 'application/json',
      },
      body: jsonEncode({
        'question': prompt,
        if (symbol != null) 'symbol': symbol,
        if (optionsMode != null) 'options_mode': optionsMode,
      }),
    );
    
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final decoded = _decode(res.body);
      
      if (decoded is Map<String, dynamic>) {
        return CopilotMessage.fromJson(decoded);
      }
      
      if (decoded is List && decoded.isNotEmpty) {
        return CopilotMessage.fromJson(
          Map<String, dynamic>.from(decoded.first as Map),
        );
      }
      
      return CopilotMessage('assistant', decoded.toString());
    }
    
    throw Exception('HTTP ${res.statusCode}: ${res.body}');
  }

  /// Fetch detailed information for a specific symbol
  /// 
  /// Returns comprehensive data including:
  /// - Price history (candlestick data)
  /// - MERIT Score and all quantitative metrics
  /// - Fundamentals
  /// - Events (earnings, dividends)
  /// - Factor breakdown
  Future<SymbolDetail> fetchSymbolDetail(String ticker, {int days = 90}) async {
    final uri = Uri.parse('${_config.baseUrl}${_config.symbolPath}/$ticker')
        .replace(queryParameters: {'days': days.toString()});
    
    debugPrint('[API] Fetching symbol detail: $uri');
    
    final res = await _client.get(
      uri,
      headers: {
        'Accept': 'application/json',
      },
    );
    
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final decoded = _decode(res.body);
      
      if (decoded is Map<String, dynamic>) {
        return SymbolDetail.fromJson(decoded);
      }
      
      throw Exception('Invalid response format for symbol detail');
    }
    
    if (res.statusCode == 404) {
      throw Exception('Symbol $ticker not found');
    }
    
    throw Exception('HTTP ${res.statusCode}: ${res.body}');
  }

  /// Generic list fetcher
  Future<List<T>> _getList<T>(
    Uri uri,
    T Function(Map<String, dynamic>) parser, {
    Map<String, String>? params,
  }) async {
    final targetUri = params == null
        ? uri
        : uri.replace(
            queryParameters: {
              ...uri.queryParameters,
              ...params,
            },
          );
    
    final res = await _client.get(
      targetUri,
      headers: {'Accept': 'application/json'},
    );
    
    if (res.statusCode >= 200 && res.statusCode < 300) {
      final decoded = _decode(res.body);
      final list = decoded is List
          ? decoded
          : (decoded is Map && decoded['data'] is List)
              ? decoded['data'] as List
              : null;
      
      if (list != null) {
        return list
            .map((e) => parser(Map<String, dynamic>.from(e as Map)))
            .toList();
      }
    }
    
    throw Exception('Failed to load $uri: HTTP ${res.statusCode}');
  }

  /// Decode JSON response
  dynamic _decode(String body) {
    try {
      return jsonDecode(body);
    } catch (_) {
      return body;
    }
  }

  /// Dispose resources
  void dispose() {
    _client.close();
  }
}
