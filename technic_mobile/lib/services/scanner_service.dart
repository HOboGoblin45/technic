/// Scanner Service
/// Service for interacting with the scanner API
library;

import 'api_client.dart';
import 'api_config.dart';

/// Scanner Service
class ScannerService {
  final ApiClient _client;
  String? _authToken;

  ScannerService({ApiClient? client, String? authToken})
      : _client = client ?? apiClient,
        _authToken = authToken;

  /// Set authentication token for subsequent requests
  void setAuthToken(String? token) {
    _authToken = token;
  }

  /// Check API health
  Future<ApiResponse<Map<String, dynamic>>> checkHealth() async {
    return await _client.get(ApiEndpoints.health, authToken: _authToken);
  }

  /// Get model status
  Future<ApiResponse<Map<String, dynamic>>> getModelStatus() async {
    return await _client.get(ApiEndpoints.modelStatus, authToken: _authToken);
  }

  /// Predict scan outcomes
  ///
  /// Parameters:
  /// - sectors: List of sectors to scan (e.g., ['Technology', 'Healthcare'])
  /// - minTechRating: Minimum technical rating (0-100)
  /// - maxSymbols: Maximum number of symbols to return
  Future<ApiResponse<Map<String, dynamic>>> predictScan({
    List<String>? sectors,
    double? minTechRating,
    int? maxSymbols,
  }) async {
    final body = <String, dynamic>{};

    if (sectors != null && sectors.isNotEmpty) {
      body['sectors'] = sectors;
    }
    if (minTechRating != null) {
      body['min_tech_rating'] = minTechRating;
    }
    if (maxSymbols != null) {
      body['max_symbols'] = maxSymbols;
    }

    return await _client.post(
      ApiEndpoints.scanPredict,
      body: body,
      authToken: _authToken,
    );
  }

  /// Get parameter suggestions
  ///
  /// Returns ML-suggested parameters for optimal scan results
  Future<ApiResponse<Map<String, dynamic>>> getSuggestions() async {
    return await _client.get(ApiEndpoints.scanSuggest, authToken: _authToken);
  }

  /// Execute a scan
  ///
  /// Parameters:
  /// - sectors: List of sectors to scan
  /// - minTechRating: Minimum technical rating
  /// - maxSymbols: Maximum number of symbols
  /// - tradeStyle: Trading style ('swing', 'day', 'position')
  /// - riskProfile: Risk profile ('conservative', 'moderate', 'aggressive')
  Future<ApiResponse<Map<String, dynamic>>> executeScan({
    List<String>? sectors,
    double? minTechRating,
    int? maxSymbols,
    String? tradeStyle,
    String? riskProfile,
  }) async {
    final body = <String, dynamic>{};

    if (sectors != null && sectors.isNotEmpty) {
      body['sectors'] = sectors;
    }
    if (minTechRating != null) {
      body['min_tech_rating'] = minTechRating;
    }
    if (maxSymbols != null) {
      body['max_symbols'] = maxSymbols;
    }
    if (tradeStyle != null) {
      body['trade_style'] = tradeStyle;
    }
    if (riskProfile != null) {
      body['risk_profile'] = riskProfile;
    }

    return await _client.post(
      ApiEndpoints.scanExecute,
      body: body,
      authToken: _authToken,
    );
  }

  /// Train ML models
  ///
  /// Triggers model training with historical scan data
  Future<ApiResponse<Map<String, dynamic>>> trainModels() async {
    return await _client.post(ApiEndpoints.modelsTrain, authToken: _authToken);
  }

  /// Clear authentication and reset state
  void reset() {
    _authToken = null;
  }
}

// ============================================================================
// SINGLETON MANAGEMENT
// ============================================================================

/// Singleton instance for Scanner Service.
///
/// Lifecycle Management:
/// - This singleton wraps the API client for scanner-specific operations
/// - Call [disposeScannerService] when cleaning up (e.g., on logout or app termination)
/// - The underlying API client should be disposed separately via [disposeApiClient]
///
/// Example usage:
/// ```dart
/// // On logout or app termination
/// void cleanup() {
///   disposeScannerService();
///   disposeApiClient();
/// }
/// ```
ScannerService? _scannerServiceInstance;

/// Get the singleton Scanner Service instance
ScannerService get scannerService {
  _scannerServiceInstance ??= ScannerService();
  return _scannerServiceInstance!;
}

/// Dispose the Scanner Service singleton.
/// Note: This does NOT dispose the underlying API client.
/// Call [disposeApiClient] separately if needed.
void disposeScannerService() {
  _scannerServiceInstance?.reset();
  _scannerServiceInstance = null;
}
