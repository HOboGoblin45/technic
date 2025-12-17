/// API Error Handler
/// 
/// Provides user-friendly error messages and retry logic for API calls
library;

import 'dart:io';
import 'package:http/http.dart' as http;

/// API Error types
enum ApiErrorType {
  network,
  timeout,
  serverError,
  notFound,
  unauthorized,
  badRequest,
  unknown,
}

/// API Error with user-friendly message
class ApiError implements Exception {
  final ApiErrorType type;
  final String message;
  final String? technicalDetails;
  final int? statusCode;
  final bool canRetry;

  const ApiError({
    required this.type,
    required this.message,
    this.technicalDetails,
    this.statusCode,
    this.canRetry = false,
  });

  /// Create error from HTTP response
  factory ApiError.fromResponse(http.Response response) {
    final statusCode = response.statusCode;
    
    if (statusCode == 404) {
      return ApiError(
        type: ApiErrorType.notFound,
        message: 'Data not found',
        technicalDetails: 'HTTP 404: ${response.body}',
        statusCode: statusCode,
        canRetry: false,
      );
    }
    
    if (statusCode == 401 || statusCode == 403) {
      return ApiError(
        type: ApiErrorType.unauthorized,
        message: 'Authentication required',
        technicalDetails: 'HTTP $statusCode: ${response.body}',
        statusCode: statusCode,
        canRetry: false,
      );
    }
    
    if (statusCode == 400) {
      return ApiError(
        type: ApiErrorType.badRequest,
        message: 'Invalid request',
        technicalDetails: 'HTTP 400: ${response.body}',
        statusCode: statusCode,
        canRetry: false,
      );
    }
    
    if (statusCode >= 500) {
      return ApiError(
        type: ApiErrorType.serverError,
        message: 'Server error. Please try again.',
        technicalDetails: 'HTTP $statusCode: ${response.body}',
        statusCode: statusCode,
        canRetry: true,
      );
    }
    
    return ApiError(
      type: ApiErrorType.unknown,
      message: 'Something went wrong',
      technicalDetails: 'HTTP $statusCode: ${response.body}',
      statusCode: statusCode,
      canRetry: true,
    );
  }

  /// Create error from exception
  factory ApiError.fromException(dynamic error) {
    if (error is SocketException) {
      return const ApiError(
        type: ApiErrorType.network,
        message: 'No internet connection',
        technicalDetails: 'SocketException: Failed to connect',
        canRetry: true,
      );
    }
    
    if (error is HttpException) {
      return ApiError(
        type: ApiErrorType.network,
        message: 'Network error',
        technicalDetails: 'HttpException: ${error.message}',
        canRetry: true,
      );
    }
    
    if (error is FormatException) {
      return ApiError(
        type: ApiErrorType.unknown,
        message: 'Invalid data format',
        technicalDetails: 'FormatException: ${error.message}',
        canRetry: false,
      );
    }
    
    if (error.toString().contains('TimeoutException')) {
      return const ApiError(
        type: ApiErrorType.timeout,
        message: 'Request timed out',
        technicalDetails: 'TimeoutException: Request took too long',
        canRetry: true,
      );
    }
    
    return ApiError(
      type: ApiErrorType.unknown,
      message: 'Something went wrong',
      technicalDetails: error.toString(),
      canRetry: true,
    );
  }

  @override
  String toString() => message;
}

/// Retry configuration
class RetryConfig {
  final int maxAttempts;
  final Duration initialDelay;
  final double backoffMultiplier;
  final Duration maxDelay;

  const RetryConfig({
    this.maxAttempts = 3,
    this.initialDelay = const Duration(seconds: 1),
    this.backoffMultiplier = 2.0,
    this.maxDelay = const Duration(seconds: 10),
  });

  Duration getDelay(int attempt) {
    final delay = initialDelay * (backoffMultiplier * attempt);
    return delay > maxDelay ? maxDelay : delay;
  }
}

/// Execute API call with retry logic
Future<T> executeWithRetry<T>({
  required Future<T> Function() operation,
  RetryConfig config = const RetryConfig(),
  bool Function(dynamic error)? shouldRetry,
}) async {
  int attempt = 0;
  dynamic lastError;

  while (attempt < config.maxAttempts) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      attempt++;

      // Check if we should retry
      final apiError = error is ApiError ? error : ApiError.fromException(error);
      final canRetry = shouldRetry?.call(error) ?? apiError.canRetry;

      if (!canRetry || attempt >= config.maxAttempts) {
        throw apiError;
      }

      // Wait before retrying
      await Future.delayed(config.getDelay(attempt));
    }
  }

  // Should never reach here, but just in case
  throw ApiError.fromException(lastError);
}

/// Check if device has internet connection
Future<bool> hasInternetConnection() async {
  try {
    final result = await InternetAddress.lookup('google.com');
    return result.isNotEmpty && result[0].rawAddress.isNotEmpty;
  } catch (_) {
    return false;
  }
}
