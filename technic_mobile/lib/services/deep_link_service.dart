/// Deep Link Service
///
/// Handles incoming deep links and universal links for the app.
/// Supports technic:// URL scheme and https://technic.app universal links.
library;

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:app_links/app_links.dart';

/// Deep link route types
enum DeepLinkRoute {
  symbol,      // technic://symbol/AAPL
  scanner,     // technic://scanner
  watchlist,   // technic://watchlist
  alerts,      // technic://alerts
  copilot,     // technic://copilot
  settings,    // technic://settings
  unknown,     // Unrecognized route
}

/// Parsed deep link data
class DeepLinkData {
  final DeepLinkRoute route;
  final String? symbol;
  final Map<String, String> queryParams;
  final String rawUri;

  const DeepLinkData({
    required this.route,
    this.symbol,
    this.queryParams = const {},
    required this.rawUri,
  });

  @override
  String toString() {
    return 'DeepLinkData(route: $route, symbol: $symbol, params: $queryParams)';
  }
}

/// Deep Link Service
class DeepLinkService {
  DeepLinkService({AppLinks? appLinks}) : _appLinks = appLinks ?? AppLinks();

  final AppLinks _appLinks;
  StreamSubscription<Uri>? _linkSubscription;

  // Stream controller for deep link events
  final _deepLinkController = StreamController<DeepLinkData>.broadcast();

  /// Stream of parsed deep links
  Stream<DeepLinkData> get deepLinks => _deepLinkController.stream;

  /// Initialize the deep link service
  ///
  /// Call this during app startup to begin listening for deep links.
  Future<void> initialize() async {
    try {
      debugPrint('[DeepLink] Initializing deep link service');

      // Check for initial link (app opened via deep link)
      final initialUri = await _appLinks.getInitialLink();
      if (initialUri != null) {
        debugPrint('[DeepLink] Initial link: $initialUri');
        _handleUri(initialUri);
      }

      // Listen for incoming links while app is running
      _linkSubscription = _appLinks.uriLinkStream.listen(
        (uri) {
          debugPrint('[DeepLink] Received link: $uri');
          _handleUri(uri);
        },
        onError: (error) {
          debugPrint('[DeepLink] Error receiving link: $error');
        },
      );

      debugPrint('[DeepLink] Service initialized');
    } catch (e) {
      debugPrint('[DeepLink] Error initializing: $e');
    }
  }

  /// Parse and handle incoming URI
  void _handleUri(Uri uri) {
    final data = parseUri(uri);
    _deepLinkController.add(data);
  }

  /// Parse a URI into DeepLinkData
  ///
  /// Supports both custom scheme (technic://) and universal links (https://technic.app/)
  DeepLinkData parseUri(Uri uri) {
    debugPrint('[DeepLink] Parsing URI: $uri');

    final String path;
    final Map<String, String> queryParams = Map.from(uri.queryParameters);

    // Handle both custom scheme and universal links
    if (uri.scheme == 'technic') {
      // technic://symbol/AAPL or technic://scanner
      path = uri.host + uri.path;
    } else if (uri.host == 'technic.app' || uri.host == 'www.technic.app') {
      // https://technic.app/symbol/AAPL
      path = uri.path.startsWith('/') ? uri.path.substring(1) : uri.path;
    } else {
      // Unknown source
      return DeepLinkData(
        route: DeepLinkRoute.unknown,
        rawUri: uri.toString(),
      );
    }

    // Parse the path
    final segments = path.split('/').where((s) => s.isNotEmpty).toList();

    if (segments.isEmpty) {
      return DeepLinkData(
        route: DeepLinkRoute.unknown,
        rawUri: uri.toString(),
      );
    }

    final routeName = segments[0].toLowerCase();

    switch (routeName) {
      case 'symbol':
      case 'stock':
      case 's':
        // technic://symbol/AAPL
        final symbol = segments.length > 1 ? segments[1].toUpperCase() : null;
        return DeepLinkData(
          route: DeepLinkRoute.symbol,
          symbol: symbol,
          queryParams: queryParams,
          rawUri: uri.toString(),
        );

      case 'scanner':
      case 'scan':
        return DeepLinkData(
          route: DeepLinkRoute.scanner,
          queryParams: queryParams,
          rawUri: uri.toString(),
        );

      case 'watchlist':
      case 'watch':
        return DeepLinkData(
          route: DeepLinkRoute.watchlist,
          queryParams: queryParams,
          rawUri: uri.toString(),
        );

      case 'alerts':
      case 'alert':
        return DeepLinkData(
          route: DeepLinkRoute.alerts,
          queryParams: queryParams,
          rawUri: uri.toString(),
        );

      case 'copilot':
      case 'chat':
      case 'ai':
        return DeepLinkData(
          route: DeepLinkRoute.copilot,
          queryParams: queryParams,
          rawUri: uri.toString(),
        );

      case 'settings':
      case 'preferences':
        return DeepLinkData(
          route: DeepLinkRoute.settings,
          queryParams: queryParams,
          rawUri: uri.toString(),
        );

      default:
        // Check if the first segment is a stock symbol (all caps, 1-5 letters)
        if (RegExp(r'^[A-Z]{1,5}$').hasMatch(segments[0].toUpperCase())) {
          return DeepLinkData(
            route: DeepLinkRoute.symbol,
            symbol: segments[0].toUpperCase(),
            queryParams: queryParams,
            rawUri: uri.toString(),
          );
        }

        return DeepLinkData(
          route: DeepLinkRoute.unknown,
          queryParams: queryParams,
          rawUri: uri.toString(),
        );
    }
  }

  /// Generate a deep link URL for a symbol
  static String symbolLink(String symbol) {
    return 'technic://symbol/${symbol.toUpperCase()}';
  }

  /// Generate a deep link URL for scanner
  static String scannerLink({Map<String, String>? params}) {
    final uri = Uri(
      scheme: 'technic',
      host: 'scanner',
      queryParameters: params,
    );
    return uri.toString();
  }

  /// Generate a deep link URL for watchlist
  static String watchlistLink() {
    return 'technic://watchlist';
  }

  /// Generate a deep link URL for alerts
  static String alertsLink() {
    return 'technic://alerts';
  }

  /// Generate a deep link URL for copilot
  static String copilotLink({String? initialMessage}) {
    if (initialMessage != null) {
      return 'technic://copilot?message=${Uri.encodeComponent(initialMessage)}';
    }
    return 'technic://copilot';
  }

  /// Generate a universal link URL for a symbol
  static String universalSymbolLink(String symbol) {
    return 'https://technic.app/symbol/${symbol.toUpperCase()}';
  }

  /// Dispose the service
  void dispose() {
    _linkSubscription?.cancel();
    _deepLinkController.close();
    debugPrint('[DeepLink] Service disposed');
  }
}
