/// Test API Connection
/// Simple test to verify the mobile app can connect to the backend API

import 'technic_mobile/lib/services/scanner_service.dart';

void main() async {
  print('==============================================');
  print('Testing API Connection');
  print('==============================================\n');
  
  // Test 1: Health Check
  print('Test 1: Health Check');
  print('Calling: GET /health');
  final healthResponse = await scannerService.checkHealth();
  
  if (healthResponse.success) {
    print('✓ SUCCESS');
    print('Response: ${healthResponse.data}');
  } else {
    print('✗ FAILED');
    print('Error: ${healthResponse.error}');
  }
  print('');
  
  // Test 2: Model Status
  print('Test 2: Model Status');
  print('Calling: GET /models/status');
  final modelResponse = await scannerService.getModelStatus();
  
  if (modelResponse.success) {
    print('✓ SUCCESS');
    print('Response: ${modelResponse.data}');
  } else {
    print('✗ FAILED');
    print('Error: ${modelResponse.error}');
  }
  print('');
  
  // Test 3: Get Suggestions
  print('Test 3: Get Parameter Suggestions');
  print('Calling: GET /scan/suggest');
  final suggestResponse = await scannerService.getSuggestions();
  
  if (suggestResponse.success) {
    print('✓ SUCCESS');
    print('Response: ${suggestResponse.data}');
  } else {
    print('✗ FAILED');
    print('Error: ${suggestResponse.error}');
  }
  print('');
  
  // Test 4: Predict Scan
  print('Test 4: Predict Scan Outcomes');
  print('Calling: POST /scan/predict');
  final predictResponse = await scannerService.predictScan(
    sectors: ['Technology'],
    minTechRating: 50.0,
    maxSymbols: 10,
  );
  
  if (predictResponse.success) {
    print('✓ SUCCESS');
    print('Response: ${predictResponse.data}');
  } else {
    print('✗ FAILED');
    print('Error: ${predictResponse.error}');
  }
  print('');
  
  print('==============================================');
  print('API Connection Tests Complete');
  print('==============================================');
}
