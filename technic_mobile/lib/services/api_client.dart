/// API Client
/// HTTP client for making API requests with error handling and retries
library;

import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'api_config.dart';

/// API Response wrapper
class ApiResponse<T> {
  final bool success;
  final T? data;
  final String? error;
  final int? statusCode;
  
  ApiResponse({
    required this.success,
    this.data,
    this.error,
    this.statusCode,
  });
  
  factory ApiResponse.success(T data, {int? statusCode}) {
    return ApiResponse(
      success: true,
      data: data,
      statusCode: statusCode,
    );
  }
  
  factory ApiResponse.error(String error, {int? statusCode}) {
    return ApiResponse(
      success: false,
      error: error,
      statusCode: statusCode,
    );
  }
}

/// Main API Client
class ApiClient {
  final http.Client _client;
  
  ApiClient({http.Client? client}) : _client = client ?? http.Client();
  
  /// GET request
  Future<ApiResponse<Map<String, dynamic>>> get(
    String endpoint, {
    Map<String, String>? headers,
    Map<String, String>? queryParameters,
    String? authToken,
  }) async {
    try {
      final uri = _buildUri(endpoint, queryParameters);
      final response = await _client
          .get(uri, headers: _buildHeaders(headers, authToken: authToken))
          .timeout(ApiConfig.connectTimeout);
      
      return _handleResponse(response);
    } on SocketException {
      return ApiResponse.error('No internet connection');
    } on HttpException {
      return ApiResponse.error('Server error');
    } on FormatException {
      return ApiResponse.error('Invalid response format');
    } catch (e) {
      return ApiResponse.error('Unexpected error: $e');
    }
  }
  
  /// POST request
  Future<ApiResponse<Map<String, dynamic>>> post(
    String endpoint, {
    Map<String, dynamic>? body,
    Map<String, String>? headers,
    String? authToken,
  }) async {
    try {
      final uri = _buildUri(endpoint);
      final response = await _client
          .post(
            uri,
            headers: _buildHeaders(headers, authToken: authToken),
            body: body != null ? jsonEncode(body) : null,
          )
          .timeout(ApiConfig.sendTimeout);
      
      return _handleResponse(response);
    } on SocketException {
      return ApiResponse.error('No internet connection');
    } on HttpException {
      return ApiResponse.error('Server error');
    } on FormatException {
      return ApiResponse.error('Invalid response format');
    } catch (e) {
      return ApiResponse.error('Unexpected error: $e');
    }
  }
  
  /// PUT request
  Future<ApiResponse<Map<String, dynamic>>> put(
    String endpoint, {
    Map<String, dynamic>? body,
    Map<String, String>? headers,
    String? authToken,
  }) async {
    try {
      final uri = _buildUri(endpoint);
      final response = await _client
          .put(
            uri,
            headers: _buildHeaders(headers, authToken: authToken),
            body: body != null ? jsonEncode(body) : null,
          )
          .timeout(ApiConfig.sendTimeout);
      
      return _handleResponse(response);
    } on SocketException {
      return ApiResponse.error('No internet connection');
    } on HttpException {
      return ApiResponse.error('Server error');
    } on FormatException {
      return ApiResponse.error('Invalid response format');
    } catch (e) {
      return ApiResponse.error('Unexpected error: $e');
    }
  }
  
  /// DELETE request
  Future<ApiResponse<Map<String, dynamic>>> delete(
    String endpoint, {
    Map<String, String>? headers,
    String? authToken,
  }) async {
    try {
      final uri = _buildUri(endpoint);
      final response = await _client
          .delete(uri, headers: _buildHeaders(headers, authToken: authToken))
          .timeout(ApiConfig.connectTimeout);
      
      return _handleResponse(response);
    } on SocketException {
      return ApiResponse.error('No internet connection');
    } on HttpException {
      return ApiResponse.error('Server error');
    } on FormatException {
      return ApiResponse.error('Invalid response format');
    } catch (e) {
      return ApiResponse.error('Unexpected error: $e');
    }
  }
  
  /// Build URI with query parameters
  Uri _buildUri(String endpoint, [Map<String, String>? queryParameters]) {
    final baseUrl = ApiConfig.baseUrl;
    final url = '$baseUrl$endpoint';
    
    if (queryParameters != null && queryParameters.isNotEmpty) {
      return Uri.parse(url).replace(queryParameters: queryParameters);
    }
    
    return Uri.parse(url);
  }
  
  /// Build headers with optional auth token
  Map<String, String> _buildHeaders(
    Map<String, String>? additionalHeaders, {
    String? authToken,
  }) {
    final headers = Map<String, String>.from(ApiConfig.defaultHeaders);

    // Add authorization header if token provided
    if (authToken != null && authToken.isNotEmpty) {
      headers['Authorization'] = 'Bearer $authToken';
    }

    if (additionalHeaders != null) {
      headers.addAll(additionalHeaders);
    }

    return headers;
  }
  
  /// Handle HTTP response
  ApiResponse<Map<String, dynamic>> _handleResponse(http.Response response) {
    final statusCode = response.statusCode;
    
    // Success responses (200-299)
    if (statusCode >= 200 && statusCode < 300) {
      try {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return ApiResponse.success(data, statusCode: statusCode);
      } catch (e) {
        // If response body is empty or not JSON, return empty map
        return ApiResponse.success({}, statusCode: statusCode);
      }
    }
    
    // Error responses
    String errorMessage;
    try {
      final errorData = jsonDecode(response.body) as Map<String, dynamic>;
      errorMessage = errorData['detail'] ?? errorData['message'] ?? 'Request failed';
    } catch (e) {
      errorMessage = _getErrorMessageForStatusCode(statusCode);
    }
    
    return ApiResponse.error(errorMessage, statusCode: statusCode);
  }
  
  /// Get error message for status code
  String _getErrorMessageForStatusCode(int statusCode) {
    switch (statusCode) {
      case 400:
        return 'Bad request';
      case 401:
        return 'Unauthorized - please login';
      case 403:
        return 'Forbidden - insufficient permissions';
      case 404:
        return 'Resource not found';
      case 408:
        return 'Request timeout';
      case 429:
        return 'Too many requests - please try again later';
      case 500:
        return 'Internal server error';
      case 502:
        return 'Bad gateway';
      case 503:
        return 'Service unavailable';
      case 504:
        return 'Gateway timeout';
      default:
        return 'Request failed with status $statusCode';
    }
  }
  
  /// Close the client and release resources
  void close() {
    _client.close();
  }
}

// ============================================================================
// SINGLETON MANAGEMENT
// ============================================================================

/// Singleton instance for API client.
///
/// Lifecycle Management:
/// - This singleton is created when first accessed and lives for the app lifetime
/// - Call [disposeApiClient] when the app is being terminated to release resources
/// - In Flutter, call this in the app's dispose method or when signing out
///
/// Example usage in main.dart:
/// ```dart
/// void main() {
///   WidgetsBinding.instance.addObserver(
///     LifecycleEventHandler(
///       detachedCallBack: () => disposeApiClient(),
///     ),
///   );
///   runApp(MyApp());
/// }
/// ```
ApiClient? _apiClientInstance;

/// Get the singleton API client instance
ApiClient get apiClient {
  _apiClientInstance ??= ApiClient();
  return _apiClientInstance!;
}

/// Dispose the API client singleton and release HTTP resources.
/// Call this when the app is terminating or when cleaning up resources.
void disposeApiClient() {
  _apiClientInstance?.close();
  _apiClientInstance = null;
}
