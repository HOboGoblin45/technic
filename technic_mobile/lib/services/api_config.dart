/// API Configuration
/// Centralized configuration for API endpoints and settings
library;

class ApiConfig {
  // Base URLs
  static const String devBaseUrl = 'http://localhost:8002';
  static const String prodBaseUrl = 'https://your-production-url.com'; // Update when deployed
  
  // Current environment
  static const bool isProduction = false; // Set to true for production
  
  // Get active base URL
  static String get baseUrl => isProduction ? prodBaseUrl : devBaseUrl;
  
  // Timeouts
  static const Duration connectTimeout = Duration(seconds: 30);
  static const Duration receiveTimeout = Duration(seconds: 30);
  static const Duration sendTimeout = Duration(seconds: 30);
  
  // Retry configuration
  static const int maxRetries = 3;
  static const Duration retryDelay = Duration(seconds: 2);
  
  // Cache configuration
  static const Duration cacheExpiry = Duration(minutes: 5);
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
