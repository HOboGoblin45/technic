/// Error Display Widget
/// 
/// Shows user-friendly error messages with retry functionality
library;

import 'package:flutter/material.dart';
import '../theme/app_colors.dart';
import '../utils/api_error_handler.dart';

/// Error display widget with retry button
class ErrorDisplay extends StatelessWidget {
  final ApiError error;
  final VoidCallback? onRetry;
  final bool showTechnicalDetails;

  const ErrorDisplay({
    super.key,
    required this.error,
    this.onRetry,
    this.showTechnicalDetails = false,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Error icon
            Icon(
              _getErrorIcon(),
              size: 64,
              color: _getErrorColor(),
            ),
            const SizedBox(height: 24),

            // Error title
            Text(
              _getErrorTitle(),
              style: const TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),

            // Error message
            Text(
              error.message,
              style: const TextStyle(
                fontSize: 16,
                color: Colors.white70,
              ),
              textAlign: TextAlign.center,
            ),

            // Technical details (debug mode)
            if (showTechnicalDetails && error.technicalDetails != null) ...[
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white10,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  error.technicalDetails!,
                  style: const TextStyle(
                    fontSize: 12,
                    fontFamily: 'monospace',
                    color: Colors.white54,
                  ),
                  textAlign: TextAlign.left,
                ),
              ),
            ],

            // Retry button
            if (error.canRetry && onRetry != null) ...[
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh),
                label: const Text('Try Again'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 32,
                    vertical: 16,
                  ),
                  backgroundColor: AppColors.primaryBlue,
                ),
              ),
            ],

            // Help text
            const SizedBox(height: 24),
            Text(
              _getHelpText(),
              style: const TextStyle(
                fontSize: 14,
                color: Colors.white54,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  IconData _getErrorIcon() {
    switch (error.type) {
      case ApiErrorType.network:
        return Icons.wifi_off;
      case ApiErrorType.timeout:
        return Icons.access_time;
      case ApiErrorType.serverError:
        return Icons.cloud_off;
      case ApiErrorType.notFound:
        return Icons.search_off;
      case ApiErrorType.unauthorized:
        return Icons.lock;
      case ApiErrorType.badRequest:
        return Icons.error_outline;
      case ApiErrorType.unknown:
        return Icons.error_outline;
    }
  }

  Color _getErrorColor() {
    switch (error.type) {
      case ApiErrorType.network:
        return Colors.orange;
      case ApiErrorType.timeout:
        return Colors.amber;
      case ApiErrorType.serverError:
        return Colors.red;
      case ApiErrorType.notFound:
        return Colors.grey;
      case ApiErrorType.unauthorized:
        return Colors.red;
      case ApiErrorType.badRequest:
        return Colors.orange;
      case ApiErrorType.unknown:
        return Colors.red;
    }
  }

  String _getErrorTitle() {
    switch (error.type) {
      case ApiErrorType.network:
        return 'No Internet Connection';
      case ApiErrorType.timeout:
        return 'Request Timed Out';
      case ApiErrorType.serverError:
        return 'Server Error';
      case ApiErrorType.notFound:
        return 'Not Found';
      case ApiErrorType.unauthorized:
        return 'Authentication Required';
      case ApiErrorType.badRequest:
        return 'Invalid Request';
      case ApiErrorType.unknown:
        return 'Something Went Wrong';
    }
  }

  String _getHelpText() {
    switch (error.type) {
      case ApiErrorType.network:
        return 'Please check your internet connection and try again.';
      case ApiErrorType.timeout:
        return 'The request took too long. Please try again.';
      case ApiErrorType.serverError:
        return 'Our servers are experiencing issues. Please try again later.';
      case ApiErrorType.notFound:
        return 'The requested data could not be found.';
      case ApiErrorType.unauthorized:
        return 'Please sign in to continue.';
      case ApiErrorType.badRequest:
        return 'There was a problem with your request.';
      case ApiErrorType.unknown:
        return 'An unexpected error occurred. Please try again.';
    }
  }
}

/// Compact error banner for inline display
class ErrorBanner extends StatelessWidget {
  final ApiError error;
  final VoidCallback? onRetry;
  final VoidCallback? onDismiss;

  const ErrorBanner({
    super.key,
    required this.error,
    this.onRetry,
    this.onDismiss,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _getErrorColor().withOpacity(0.1),
        border: Border.all(
          color: _getErrorColor().withOpacity(0.3),
          width: 1,
        ),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(
            _getErrorIcon(),
            color: _getErrorColor(),
            size: 24,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _getErrorTitle(),
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: _getErrorColor(),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  error.message,
                  style: const TextStyle(
                    fontSize: 13,
                    color: Colors.white70,
                  ),
                ),
              ],
            ),
          ),
          if (error.canRetry && onRetry != null) ...[
            const SizedBox(width: 12),
            IconButton(
              onPressed: onRetry,
              icon: const Icon(Icons.refresh),
              color: _getErrorColor(),
              tooltip: 'Retry',
            ),
          ],
          if (onDismiss != null) ...[
            const SizedBox(width: 8),
            IconButton(
              onPressed: onDismiss,
              icon: const Icon(Icons.close),
              color: Colors.white54,
              tooltip: 'Dismiss',
            ),
          ],
        ],
      ),
    );
  }

  IconData _getErrorIcon() {
    switch (error.type) {
      case ApiErrorType.network:
        return Icons.wifi_off;
      case ApiErrorType.timeout:
        return Icons.access_time;
      case ApiErrorType.serverError:
        return Icons.cloud_off;
      case ApiErrorType.notFound:
        return Icons.search_off;
      case ApiErrorType.unauthorized:
        return Icons.lock;
      case ApiErrorType.badRequest:
        return Icons.error_outline;
      case ApiErrorType.unknown:
        return Icons.error_outline;
    }
  }

  Color _getErrorColor() {
    switch (error.type) {
      case ApiErrorType.network:
        return Colors.orange;
      case ApiErrorType.timeout:
        return Colors.amber;
      case ApiErrorType.serverError:
        return Colors.red;
      case ApiErrorType.notFound:
        return Colors.grey;
      case ApiErrorType.unauthorized:
        return Colors.red;
      case ApiErrorType.badRequest:
        return Colors.orange;
      case ApiErrorType.unknown:
        return Colors.red;
    }
  }

  String _getErrorTitle() {
    switch (error.type) {
      case ApiErrorType.network:
        return 'No Internet';
      case ApiErrorType.timeout:
        return 'Timeout';
      case ApiErrorType.serverError:
        return 'Server Error';
      case ApiErrorType.notFound:
        return 'Not Found';
      case ApiErrorType.unauthorized:
        return 'Auth Required';
      case ApiErrorType.badRequest:
        return 'Invalid Request';
      case ApiErrorType.unknown:
        return 'Error';
    }
  }
}

/// Show error snackbar
void showErrorSnackBar(BuildContext context, ApiError error, {VoidCallback? onRetry}) {
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(
      content: Row(
        children: [
          Icon(
            error.type == ApiErrorType.network ? Icons.wifi_off : Icons.error_outline,
            color: Colors.white,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(error.message),
          ),
        ],
      ),
      backgroundColor: Colors.red.shade700,
      action: error.canRetry && onRetry != null
          ? SnackBarAction(
              label: 'Retry',
              textColor: Colors.white,
              onPressed: onRetry,
            )
          : null,
      duration: const Duration(seconds: 4),
    ),
  );
}
