/// Scanner Provider
/// State management for scanner functionality using Riverpod
library;

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/scanner_service.dart';

/// Scanner state
enum ScannerState {
  idle,
  loading,
  success,
  error,
}

/// Scanner Provider
class ScannerProvider with ChangeNotifier {
  final ScannerService _service;
  
  ScannerProvider({ScannerService? service}) 
      : _service = service ?? scannerService;
  
  // State
  ScannerState _state = ScannerState.idle;
  List<Map<String, dynamic>> _results = [];
  String? _errorMessage;
  Map<String, dynamic>? _scanMetadata;
  
  // Getters
  ScannerState get state => _state;
  List<Map<String, dynamic>> get results => _results;
  String? get errorMessage => _errorMessage;
  Map<String, dynamic>? get scanMetadata => _scanMetadata;
  bool get isLoading => _state == ScannerState.loading;
  bool get hasError => _state == ScannerState.error;
  bool get hasResults => _results.isNotEmpty;
  
  /// Check API health
  Future<bool> checkHealth() async {
    try {
      final response = await _service.checkHealth();
      return response.success;
    } catch (e) {
      debugPrint('Health check failed: $e');
      return false;
    }
  }
  
  /// Get model status
  Future<Map<String, dynamic>?> getModelStatus() async {
    try {
      final response = await _service.getModelStatus();
      if (response.success) {
        return response.data;
      }
      return null;
    } catch (e) {
      debugPrint('Model status check failed: $e');
      return null;
    }
  }
  
  /// Get parameter suggestions
  Future<Map<String, dynamic>?> getSuggestions() async {
    try {
      final response = await _service.getSuggestions();
      if (response.success) {
        return response.data;
      }
      return null;
    } catch (e) {
      debugPrint('Get suggestions failed: $e');
      return null;
    }
  }
  
  /// Execute scan
  Future<void> executeScan({
    List<String>? sectors,
    double? minTechRating,
    int? maxSymbols,
    String? tradeStyle,
    String? riskProfile,
  }) async {
    _setState(ScannerState.loading);
    _errorMessage = null;
    
    try {
      final response = await _service.executeScan(
        sectors: sectors,
        minTechRating: minTechRating,
        maxSymbols: maxSymbols,
        tradeStyle: tradeStyle,
        riskProfile: riskProfile,
      );
      
      if (response.success && response.data != null) {
        // Extract results from response
        final data = response.data!;
        
        // Handle different response formats
        if (data.containsKey('results')) {
          _results = List<Map<String, dynamic>>.from(data['results'] ?? []);
        } else if (data.containsKey('symbols')) {
          _results = List<Map<String, dynamic>>.from(data['symbols'] ?? []);
        } else {
          // If response is a list directly
          _results = [data];
        }
        
        // Store metadata
        _scanMetadata = {
          'timestamp': DateTime.now().toIso8601String(),
          'count': _results.length,
          'parameters': {
            'sectors': sectors,
            'minTechRating': minTechRating,
            'maxSymbols': maxSymbols,
            'tradeStyle': tradeStyle,
            'riskProfile': riskProfile,
          },
          ...data, // Include all response data
        };
        
        _setState(ScannerState.success);
      } else {
        _errorMessage = response.error ?? 'Scan failed';
        _setState(ScannerState.error);
      }
    } catch (e) {
      _errorMessage = 'Unexpected error: $e';
      _setState(ScannerState.error);
      debugPrint('Scan execution failed: $e');
    }
  }
  
  /// Predict scan outcomes
  Future<Map<String, dynamic>?> predictScan({
    List<String>? sectors,
    double? minTechRating,
    int? maxSymbols,
  }) async {
    try {
      final response = await _service.predictScan(
        sectors: sectors,
        minTechRating: minTechRating,
        maxSymbols: maxSymbols,
      );
      
      if (response.success) {
        return response.data;
      }
      return null;
    } catch (e) {
      debugPrint('Predict scan failed: $e');
      return null;
    }
  }
  
  /// Clear results
  void clearResults() {
    _results = [];
    _scanMetadata = null;
    _errorMessage = null;
    _setState(ScannerState.idle);
  }
  
  /// Clear error
  void clearError() {
    _errorMessage = null;
    if (_state == ScannerState.error) {
      _setState(ScannerState.idle);
    }
  }
  
  /// Set state and notify listeners
  void _setState(ScannerState newState) {
    _state = newState;
    notifyListeners();
  }
}

// ============================================================================
// RIVERPOD PROVIDERS
// ============================================================================

/// Scanner Service Provider (singleton)
final scannerServiceProvider = Provider<ScannerService>((ref) {
  return scannerService;
});

/// Scanner Provider (ChangeNotifier wrapped for Riverpod)
final scannerNotifierProvider = ChangeNotifierProvider<ScannerProvider>((ref) {
  return ScannerProvider(service: ref.read(scannerServiceProvider));
});
