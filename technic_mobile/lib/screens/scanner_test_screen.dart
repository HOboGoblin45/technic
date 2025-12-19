/// Scanner Test Screen
/// Demo screen to test backend integration
library;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/scanner_provider.dart';
import '../widgets/mac_button.dart';
import '../widgets/mac_card.dart';
import '../theme/spacing.dart';

class ScannerTestScreen extends StatefulWidget {
  const ScannerTestScreen({super.key});
  
  @override
  State<ScannerTestScreen> createState() => _ScannerTestScreenState();
}

class _ScannerTestScreenState extends State<ScannerTestScreen> {
  final List<String> _selectedSectors = ['Technology'];
  double _minTechRating = 50.0;
  int _maxSymbols = 10;
  
  @override
  void initState() {
    super.initState();
    // Check API health on load
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _checkHealth();
    });
  }
  
  Future<void> _checkHealth() async {
    final provider = context.read<ScannerProvider>();
    final isHealthy = await provider.checkHealth();
    
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(isHealthy ? '✓ API Connected' : '✗ API Disconnected'),
          backgroundColor: isHealthy ? Colors.green : Colors.red,
        ),
      );
    }
  }
  
  Future<void> _runScan() async {
    final provider = context.read<ScannerProvider>();
    
    await provider.executeScan(
      sectors: _selectedSectors,
      minTechRating: _minTechRating,
      maxSymbols: _maxSymbols,
      tradeStyle: 'swing',
      riskProfile: 'moderate',
    );
  }
  
  Future<void> _getPrediction() async {
    final provider = context.read<ScannerProvider>();
    
    final prediction = await provider.predictScan(
      sectors: _selectedSectors,
      minTechRating: _minTechRating,
      maxSymbols: _maxSymbols,
    );
    
    if (mounted && prediction != null) {
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Scan Prediction'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Predicted Results: ${prediction['predicted_results']}'),
              Text('Estimated Duration: ${prediction['predicted_duration']?.toStringAsFixed(1)}s'),
              Text('Confidence: ${(prediction['confidence'] * 100).toStringAsFixed(0)}%'),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Close'),
            ),
          ],
        ),
      );
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scanner Test'),
        actions: [
          IconButton(
            icon: const Icon(Icons.health_and_safety),
            onPressed: _checkHealth,
            tooltip: 'Check API Health',
          ),
        ],
      ),
      body: Consumer<ScannerProvider>(
        builder: (context, provider, child) {
          return SingleChildScrollView(
            padding: Spacing.edgeInsetsLG,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Status Card
                MacCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'API Status',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: Spacing.sm),
                      Text('State: ${provider.state.name}'),
                      if (provider.hasError)
                        Text(
                          'Error: ${provider.errorMessage}',
                          style: const TextStyle(color: Colors.red),
                        ),
                    ],
                  ),
                ),
                
                const SizedBox(height: Spacing.lg),
                
                // Parameters Card
                MacCard(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Scan Parameters',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: Spacing.md),
                      
                      // Sectors
                      Text('Sectors: ${_selectedSectors.join(', ')}'),
                      const SizedBox(height: Spacing.sm),
                      
                      // Min Tech Rating
                      Text('Min Tech Rating: ${_minTechRating.toInt()}'),
                      Slider(
                        value: _minTechRating,
                        min: 0,
                        max: 100,
                        divisions: 20,
                        label: _minTechRating.toInt().toString(),
                        onChanged: (value) {
                          setState(() => _minTechRating = value);
                        },
                      ),
                      
                      // Max Symbols
                      Text('Max Symbols: $_maxSymbols'),
                      Slider(
                        value: _maxSymbols.toDouble(),
                        min: 5,
                        max: 50,
                        divisions: 9,
                        label: _maxSymbols.toString(),
                        onChanged: (value) {
                          setState(() => _maxSymbols = value.toInt());
                        },
                      ),
                    ],
                  ),
                ),
                
                const SizedBox(height: Spacing.lg),
                
                // Action Buttons
                MacButton(
                  text: 'Get Prediction',
                  icon: Icons.analytics,
                  onPressed: provider.isLoading ? null : _getPrediction,
                  isFullWidth: true,
                ),
                
                const SizedBox(height: Spacing.md),
                
                MacButton(
                  text: provider.isLoading ? 'Scanning...' : 'Run Scan',
                  icon: Icons.search,
                  onPressed: provider.isLoading ? null : _runScan,
                  isLoading: provider.isLoading,
                  isFullWidth: true,
                ),
                
                const SizedBox(height: Spacing.lg),
                
                // Results Card
                if (provider.hasResults)
                  MacCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Results (${provider.results.length})',
                          style: Theme.of(context).textTheme.titleLarge,
                        ),
                        const SizedBox(height: Spacing.md),
                        ...provider.results.take(5).map((result) {
                          return Padding(
                            padding: const EdgeInsets.only(bottom: Spacing.sm),
                            child: Text(
                              result.toString(),
                              style: const TextStyle(fontSize: 12),
                            ),
                          );
                        }),
                        if (provider.results.length > 5)
                          Text('... and ${provider.results.length - 5} more'),
                      ],
                    ),
                  ),
                
                // Metadata Card
                if (provider.scanMetadata != null)
                  MacCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Scan Metadata',
                          style: Theme.of(context).textTheme.titleLarge,
                        ),
                        const SizedBox(height: Spacing.md),
                        Text('Count: ${provider.scanMetadata!['count']}'),
                        Text('Timestamp: ${provider.scanMetadata!['timestamp']}'),
                      ],
                    ),
                  ),
              ],
            ),
          );
        },
      ),
    );
  }
}
