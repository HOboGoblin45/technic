/// Technic API Service
/// 
/// Handles all HTTP communication with the Technic backend API.
/// Provides methods for scanning, fetching ideas, copilot interactions, etc.

import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../models/scan_result.dart';
import '../models/market_mover.dart';
import '../models/idea.dart';
import '../models/scoreboard_slice.dart';
import '../models/scanner_bundle.dart';
import '../models/copilot_message.dart';
import '../models/universe_stats.dart';

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
      defaultValue: 'http://localhost:8502',
    );
    final normalizedBase = _normalizeBaseForPlatform(rawBase);
    
    return ApiConfig(
      baseUrl: normalizedBase,
      moversPath: const String.fromEnvironment(
        'TECHNIC_API_MOVERS',
        defaultValue: '/api/movers',
      ),
      scanPath: const String.fromEnvironment(
        'TECHNIC_API_SCANNER',
        defaultValue: '/api/scanner',
      ),
      ideasPath: const String.fromEnvironment(
        'TECHNIC_API_IDEAS',
        defaultValue: '/api/ideas',
      ),
      scoreboardPath: const String.fromEnvironment(
        'TECHNIC_API_SCOREBOARD',
        defaultValue: '/api/scoreboard',
      ),
      copilotPath: const String.fromEnvironment(
        'TECHNIC_API_COPILOT',
        defaultValue: '/api/copilot',
      ),
      universeStatsPath: const String.fromEnvironment(
        'TECHNIC_API_UNIVERSE',
        defaultValue: '/api/universe_stats',
      ),
      symbolPath: const String.fromEnvironment(
        'TECHNIC_API_SYMBOL',
        defaultValue: '/api/symbol',
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
  Future<ScannerBundle> fetchScannerBundle({
    Map<String, String>? params,
  }) async {
    final results = await Future.wait([
      _safe(() => fetchMovers(params: params), <MarketMover>[]),
      _safe(
        () => fetchScanResults(params: params),
        const ScanResultsPayload([], null),
      ),
      _safe(() => fetchScoreboard(), <ScoreboardSlice>[]),
    ]);
    
    return ScannerBundle(
      movers: results[0] as List<MarketMover>,
      scanResults: (results[1] as ScanResultsPayload).results,
      scoreboard: results[2] as List<ScoreboardSlice>,
      progress: (results[1] as ScanResultsPayload).progress,
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
      headers: {'Content-Type': 'application/json'},
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

  /// Safe execution with fallback
  Future<T> _safe<T>(Future<T> Function() task, T fallback) async {
    try {
      return await task();
    } catch (e) {
      debugPrint('API fallback for $task: $e');
      return fallback;
    }
  }

  /// Dispose resources
  void dispose() {
    _client.close();
  }
}
