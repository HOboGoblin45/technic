/// Authentication Service
/// 
/// Handles user authentication including login, signup, logout,
/// and JWT token management with secure storage.
library;

import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// User model
class User {
  final String id;
  final String email;
  final String name;
  
  const User({
    required this.id,
    required this.email,
    required this.name,
  });
  
  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id']?.toString() ?? '',
      email: json['email']?.toString() ?? '',
      name: json['name']?.toString() ?? '',
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'email': email,
      'name': name,
    };
  }
}

/// Authentication response
class AuthResponse {
  final User user;
  final String accessToken;
  final String? refreshToken;
  
  const AuthResponse({
    required this.user,
    required this.accessToken,
    this.refreshToken,
  });
  
  factory AuthResponse.fromJson(Map<String, dynamic> json) {
    return AuthResponse(
      user: User.fromJson(json['user'] as Map<String, dynamic>),
      accessToken: json['access_token']?.toString() ?? '',
      refreshToken: json['refresh_token']?.toString(),
    );
  }
}

/// Authentication Service
class AuthService {
  AuthService({
    http.Client? client,
    FlutterSecureStorage? storage,
    String? baseUrl,
  })  : _client = client ?? http.Client(),
        _storage = storage ?? const FlutterSecureStorage(),
        _baseUrl = baseUrl ?? 'https://technic-m5vn.onrender.com';

  final http.Client _client;
  final FlutterSecureStorage _storage;
  final String _baseUrl;

  // Storage keys
  static const String _accessTokenKey = 'access_token';
  static const String _refreshTokenKey = 'refresh_token';
  static const String _userIdKey = 'user_id';
  static const String _userEmailKey = 'user_email';
  static const String _userNameKey = 'user_name';

  /// Login with email and password
  /// 
  /// Returns AuthResponse with user data and tokens on success.
  /// Throws Exception on failure.
  Future<AuthResponse> login(String email, String password) async {
    try {
      debugPrint('[Auth] Attempting login for: $email');
      
      final uri = Uri.parse('$_baseUrl/api/auth/login');
      final res = await _client.post(
        uri,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode({
          'email': email,
          'password': password,
        }),
      );

      if (res.statusCode == 200 || res.statusCode == 201) {
        final decoded = jsonDecode(res.body) as Map<String, dynamic>;
        final authResponse = AuthResponse.fromJson(decoded);
        
        // Store tokens and user data securely
        await _storeAuthData(authResponse);
        
        debugPrint('[Auth] Login successful for: $email');
        return authResponse;
      }

      // Handle error responses
      final errorBody = _tryDecodeError(res.body);
      final errorMessage = errorBody['message'] ?? errorBody['detail'] ?? 'Login failed';
      
      debugPrint('[Auth] Login failed: $errorMessage');
      throw Exception(errorMessage);
    } catch (e) {
      debugPrint('[Auth] Login error: $e');
      if (e is Exception) rethrow;
      throw Exception('Login failed: $e');
    }
  }

  /// Sign up new user
  /// 
  /// Returns AuthResponse with user data and tokens on success.
  /// Throws Exception on failure.
  Future<AuthResponse> signup(String email, String password, String name) async {
    try {
      debugPrint('[Auth] Attempting signup for: $email');
      
      final uri = Uri.parse('$_baseUrl/api/auth/signup');
      final res = await _client.post(
        uri,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode({
          'email': email,
          'password': password,
          'name': name,
        }),
      );

      if (res.statusCode == 200 || res.statusCode == 201) {
        final decoded = jsonDecode(res.body) as Map<String, dynamic>;
        final authResponse = AuthResponse.fromJson(decoded);
        
        // Store tokens and user data securely
        await _storeAuthData(authResponse);
        
        debugPrint('[Auth] Signup successful for: $email');
        return authResponse;
      }

      // Handle error responses
      final errorBody = _tryDecodeError(res.body);
      final errorMessage = errorBody['message'] ?? errorBody['detail'] ?? 'Signup failed';
      
      debugPrint('[Auth] Signup failed: $errorMessage');
      throw Exception(errorMessage);
    } catch (e) {
      debugPrint('[Auth] Signup error: $e');
      if (e is Exception) rethrow;
      throw Exception('Signup failed: $e');
    }
  }

  /// Logout user and clear all stored data
  Future<void> logout() async {
    try {
      debugPrint('[Auth] Logging out user');
      
      // Optionally call backend logout endpoint
      final token = await getAccessToken();
      if (token != null) {
        try {
          final uri = Uri.parse('$_baseUrl/api/auth/logout');
          await _client.post(
            uri,
            headers: {
              'Authorization': 'Bearer $token',
              'Accept': 'application/json',
            },
          );
        } catch (e) {
          debugPrint('[Auth] Backend logout failed (continuing): $e');
        }
      }
      
      // Clear all stored data
      await _clearAuthData();
      
      debugPrint('[Auth] Logout complete');
    } catch (e) {
      debugPrint('[Auth] Logout error: $e');
      // Still clear local data even if backend call fails
      await _clearAuthData();
    }
  }

  /// Check if user is authenticated
  /// 
  /// Returns true if valid access token exists.
  Future<bool> isAuthenticated() async {
    final token = await getAccessToken();
    return token != null && token.isNotEmpty;
  }

  /// Get current user from stored data
  /// 
  /// Returns User object if authenticated, null otherwise.
  Future<User?> getCurrentUser() async {
    try {
      final id = await _storage.read(key: _userIdKey);
      final email = await _storage.read(key: _userEmailKey);
      final name = await _storage.read(key: _userNameKey);
      
      if (id != null && email != null && name != null) {
        return User(id: id, email: email, name: name);
      }
      
      return null;
    } catch (e) {
      debugPrint('[Auth] Error getting current user: $e');
      return null;
    }
  }

  /// Get stored access token
  Future<String?> getAccessToken() async {
    try {
      return await _storage.read(key: _accessTokenKey);
    } catch (e) {
      debugPrint('[Auth] Error reading access token: $e');
      return null;
    }
  }

  /// Get stored refresh token
  Future<String?> getRefreshToken() async {
    try {
      return await _storage.read(key: _refreshTokenKey);
    } catch (e) {
      debugPrint('[Auth] Error reading refresh token: $e');
      return null;
    }
  }

  /// Refresh access token using refresh token
  /// 
  /// Returns new access token on success.
  /// Throws Exception on failure.
  Future<String> refreshAccessToken() async {
    try {
      final refreshToken = await getRefreshToken();
      if (refreshToken == null) {
        throw Exception('No refresh token available');
      }
      
      debugPrint('[Auth] Refreshing access token');
      
      final uri = Uri.parse('$_baseUrl/api/auth/refresh');
      final res = await _client.post(
        uri,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: jsonEncode({
          'refresh_token': refreshToken,
        }),
      );

      if (res.statusCode == 200) {
        final decoded = jsonDecode(res.body) as Map<String, dynamic>;
        final newAccessToken = decoded['access_token']?.toString();
        
        if (newAccessToken != null) {
          // Store new access token
          await _storage.write(key: _accessTokenKey, value: newAccessToken);
          
          debugPrint('[Auth] Access token refreshed');
          return newAccessToken;
        }
      }

      throw Exception('Token refresh failed');
    } catch (e) {
      debugPrint('[Auth] Token refresh error: $e');
      if (e is Exception) rethrow;
      throw Exception('Token refresh failed: $e');
    }
  }

  /// Store authentication data securely
  Future<void> _storeAuthData(AuthResponse authResponse) async {
    await _storage.write(key: _accessTokenKey, value: authResponse.accessToken);
    
    if (authResponse.refreshToken != null) {
      await _storage.write(key: _refreshTokenKey, value: authResponse.refreshToken!);
    }
    
    await _storage.write(key: _userIdKey, value: authResponse.user.id);
    await _storage.write(key: _userEmailKey, value: authResponse.user.email);
    await _storage.write(key: _userNameKey, value: authResponse.user.name);
  }

  /// Clear all stored authentication data
  Future<void> _clearAuthData() async {
    await _storage.delete(key: _accessTokenKey);
    await _storage.delete(key: _refreshTokenKey);
    await _storage.delete(key: _userIdKey);
    await _storage.delete(key: _userEmailKey);
    await _storage.delete(key: _userNameKey);
  }

  /// Try to decode error response
  Map<String, dynamic> _tryDecodeError(String body) {
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map<String, dynamic>) {
        return decoded;
      }
      return {'message': body};
    } catch (_) {
      return {'message': body};
    }
  }

  /// Dispose resources
  void dispose() {
    _client.close();
  }
}
