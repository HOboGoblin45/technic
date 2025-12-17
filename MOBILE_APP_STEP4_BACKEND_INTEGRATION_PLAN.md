# Step 4: Backend Integration - Implementation Plan

## Overview

**Goal:** Connect the Flutter mobile app to the Python backend API  
**Duration:** 2-3 hours  
**Current Status:** Ready to start  
**Prerequisites:** ✅ Step 3 complete (Mac aesthetic ready)

---

## Backend API Status

### Available APIs:
1. **Scanner API** (api_ml_enhanced.py)
   - POST /scan - Run stock scanner
   - GET /scan/status - Check scan status
   - GET /scan/results - Get scan results

2. **Monitoring API** (monitoring_api_optimized.py)
   - GET /health - Health check
   - GET /metrics - System metrics
   - GET /performance - Performance stats

3. **ML API** (api_ml_enhanced.py)
   - POST /predict - ML predictions
   - GET /models - Available models
   - GET /model/status - Model status

### Backend Location:
- **Base URL:** http://localhost:8000 (development)
- **Production URL:** TBD (Render deployment)

---

## Implementation Tasks

### Task 1: API Client Setup (30 min)

**Goal:** Create HTTP client for API communication

**Subtasks:**
1. Add dependencies to pubspec.yaml
   - `http: ^1.1.0` - HTTP requests
   - `dio: ^5.4.0` - Advanced HTTP client (optional)
   - `flutter_secure_storage: ^9.0.0` - Secure token storage

2. Create API client structure
   - `lib/services/api_client.dart` - Base HTTP client
   - `lib/services/api_config.dart` - API configuration
   - `lib/services/api_endpoints.dart` - Endpoint constants

3. Implement error handling
   - Network errors
   - Timeout errors
   - API errors (4xx, 5xx)

**Files to create:**
- `technic_mobile/lib/services/api_client.dart`
- `technic_mobile/lib/services/api_config.dart`
- `technic_mobile/lib/services/api_endpoints.dart`
- `technic_mobile/lib/models/api_response.dart`

---

### Task 2: Authentication (30 min)

**Goal:** Implement user authentication

**Subtasks:**
1. Create authentication service
   - Login/logout
   - Token management
   - Session persistence

2. Create auth models
   - User model
   - Auth token model
   - Auth state model

3. Implement secure storage
   - Store auth tokens
   - Retrieve tokens
   - Clear tokens on logout

**Files to create:**
- `technic_mobile/lib/services/auth_service.dart`
- `technic_mobile/lib/models/user.dart`
- `technic_mobile/lib/models/auth_token.dart`

---

### Task 3: Scanner Integration (45 min)

**Goal:** Connect scanner screen to backend API

**Subtasks:**
1. Create scanner service
   - Start scan
   - Check scan status
   - Get scan results
   - Cancel scan

2. Create scanner models
   - Scan request model
   - Scan response model
   - Stock result model

3. Update scanner screen
   - Connect to API
   - Show loading states
   - Display results
   - Handle errors

**Files to create:**
- `technic_mobile/lib/services/scanner_service.dart`
- `technic_mobile/lib/models/scan_request.dart`
- `technic_mobile/lib/models/scan_response.dart`
- `technic_mobile/lib/models/stock_result.dart`

**Files to modify:**
- `technic_mobile/lib/screens/scanner_screen.dart`

---

### Task 4: Watchlist Integration (30 min)

**Goal:** Connect watchlist to backend

**Subtasks:**
1. Create watchlist service
   - Get watchlist
   - Add symbol
   - Remove symbol
   - Update alerts

2. Create watchlist models
   - Watchlist item model
   - Alert model

3. Update watchlist screen
   - Load from API
   - Sync changes
   - Handle errors

**Files to create:**
- `technic_mobile/lib/services/watchlist_service.dart`
- `technic_mobile/lib/models/watchlist_item.dart`
- `technic_mobile/lib/models/alert.dart`

**Files to modify:**
- `technic_mobile/lib/screens/watchlist_screen.dart`

---

### Task 5: State Management (30 min)

**Goal:** Implement state management for API data

**Subtasks:**
1. Choose state management solution
   - Provider (recommended - simple)
   - Riverpod (advanced)
   - Bloc (complex)

2. Create providers/controllers
   - Auth provider
   - Scanner provider
   - Watchlist provider

3. Integrate with screens
   - Connect providers
   - Update UI on state changes
   - Handle loading/error states

**Files to create:**
- `technic_mobile/lib/providers/auth_provider.dart`
- `technic_mobile/lib/providers/scanner_provider.dart`
- `technic_mobile/lib/providers/watchlist_provider.dart`

---

### Task 6: Testing & Error Handling (30 min)

**Goal:** Test integration and handle errors gracefully

**Subtasks:**
1. Test API calls
   - Test successful requests
   - Test error scenarios
   - Test timeout handling

2. Implement error UI
   - Error messages
   - Retry buttons
   - Offline mode indicators

3. Add loading states
   - Skeleton screens
   - Progress indicators
   - Pull-to-refresh

**Files to create:**
- `technic_mobile/lib/widgets/error_widget.dart`
- `technic_mobile/lib/widgets/loading_widget.dart`
- `test/services/api_client_test.dart`

---

## Dependencies to Add

```yaml
dependencies:
  # HTTP & Networking
  http: ^1.1.0
  dio: ^5.4.0  # Optional, more features
  
  # State Management
  provider: ^6.1.1  # Recommended
  
  # Secure Storage
  flutter_secure_storage: ^9.0.0
  
  # JSON Serialization
  json_annotation: ^4.8.1
  
dev_dependencies:
  # Code Generation
  build_runner: ^2.4.7
  json_serializable: ^6.7.1
  
  # Testing
  mockito: ^5.4.4
  http_mock_adapter: ^0.6.1
```

---

## API Endpoints Reference

### Scanner API
```
POST   /scan                 - Start new scan
GET    /scan/status/{id}     - Check scan status
GET    /scan/results/{id}    - Get scan results
DELETE /scan/{id}            - Cancel scan
```

### Watchlist API
```
GET    /watchlist            - Get user's watchlist
POST   /watchlist            - Add symbol to watchlist
DELETE /watchlist/{symbol}   - Remove symbol
PUT    /watchlist/{symbol}   - Update alerts
```

### Auth API
```
POST   /auth/login           - User login
POST   /auth/logout          - User logout
POST   /auth/refresh         - Refresh token
GET    /auth/me              - Get current user
```

---

## Error Handling Strategy

### Network Errors:
- No internet connection → Show offline indicator
- Timeout → Show retry button
- Server unreachable → Show error message

### API Errors:
- 400 Bad Request → Show validation errors
- 401 Unauthorized → Redirect to login
- 403 Forbidden → Show permission error
- 404 Not Found → Show not found message
- 500 Server Error → Show generic error

### User Experience:
- Always show loading states
- Provide clear error messages
- Offer retry options
- Cache data when possible
- Support offline mode

---

## Testing Strategy

### Unit Tests:
- Test API client methods
- Test model serialization
- Test error handling
- Test state management

### Integration Tests:
- Test API calls end-to-end
- Test authentication flow
- Test scanner flow
- Test watchlist sync

### Manual Tests:
- Test with real backend
- Test error scenarios
- Test offline mode
- Test performance

---

## Timeline

### Hour 1: Foundation
- ✅ Task 1: API Client Setup (30 min)
- ✅ Task 2: Authentication (30 min)

### Hour 2: Core Features
- ✅ Task 3: Scanner Integration (45 min)
- ✅ Task 5: State Management (15 min)

### Hour 3: Polish
- ✅ Task 4: Watchlist Integration (30 min)
- ✅ Task 6: Testing & Error Handling (30 min)

---

## Success Criteria

### Technical:
- ✅ All API calls work correctly
- ✅ Authentication flow complete
- ✅ Scanner connects to backend
- ✅ Watchlist syncs with backend
- ✅ Error handling implemented
- ✅ Loading states added

### User Experience:
- ✅ Fast response times
- ✅ Clear error messages
- ✅ Smooth loading states
- ✅ Offline support (basic)
- ✅ Retry functionality

---

## Next Steps After Completion

1. **Deploy Backend** - Deploy to Render/AWS
2. **Update API URLs** - Point to production
3. **Add Real-Time Updates** - WebSocket integration
4. **Implement Caching** - Reduce API calls
5. **Add Analytics** - Track usage
6. **Performance Optimization** - Reduce load times

---

*Ready to start Step 4: Backend Integration*  
*Estimated completion: 2-3 hours*  
*Let's build a fully functional mobile app!*
