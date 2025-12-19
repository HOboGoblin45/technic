/// API Configuration
/// Centralized configuration for API endpoints and settings
///
/// Environment Variables:
/// - API_BASE_URL: Override base URL (e.g., --dart-define=API_BASE_URL=https://api.example.com)
/// - PRODUCTION: Set to true for production mode (e.g., --dart-define=PRODUCTION=true)
library;

import '../utils/constants.dart' as constants;

class ApiConfig {
  // Environment-driven base URL
  // Override with: flutter run --dart-define=API_BASE_URL=https://your-api.com
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://localhost:8002',
  );

  // Production mode flag
  // Override with: flutter run --dart-define=PRODUCTION=true
  static const bool isProduction = bool.fromEnvironment(
    'PRODUCTION',
    defaultValue: false,
  );

  // Timeouts - reference centralized constants
  static Duration get connectTimeout => constants.apiTimeout;
  static Duration get receiveTimeout => constants.apiTimeout;
  static Duration get sendTimeout => constants.apiTimeout;

  // Retry configuration
  static const int maxRetries = 3;
  static const Duration retryDelay = Duration(seconds: 2);
  
  // Cache configuration - reference centralized constants
  static Duration get cacheExpiry => constants.cacheShortDuration;
  static const bool enableCache = true;
  
  // Headers
  static Map<String, String> get defaultHeaders => {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  };
  
}

/// API Endpoints
class ApiEndpoints {
  // Health & Status
  static const String health = '/health';
  static const String modelStatus = '/models/status';
  
  // Scanner
  static const String scanPredict = '/scan/predict';
  static const String scanSuggest = '/scan/suggest';
  static const String scanExecute = '/scan/execute';
  
  // Models
  static const String modelsTrain = '/models/train';
  static const String modelsStatus = '/models/status';
}
