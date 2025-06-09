# Complete Authentication Data Flow Analysis

> **Purpose**: Establish the "social truth" between frontend and backend agents about authentication flows, failure points, and data transformations.

## 1. Complete iOS to Backend Authentication Flow

```mermaid
sequenceDiagram
    participant iOS as iOS App
    participant FB as Firebase Auth
    participant Backend as Backend API
    participant MW as Middleware
    participant FBA as Firebase Admin SDK
    participant EP as Endpoint

    iOS->>FB: 1. User login (email/password)
    FB-->>iOS: 2. Firebase User + ID Token
    
    Note over iOS: Token Management Service
    iOS->>iOS: 3. Check token expiry
    iOS->>FB: 4. getIDToken(forcingRefresh: true)
    FB-->>iOS: 5. Fresh ID Token (JWT)
    
    iOS->>Backend: 6. HTTPS Request + Bearer Token
    Note over iOS,Backend: Authorization: Bearer eyJhbGci...
    
    Backend->>MW: 7. Request enters middleware
    MW->>MW: 8. Check exempt paths
    MW->>MW: 9. Extract token from header
    
    MW->>FBA: 10. verify_id_token(token)
    FBA->>FB: 11. Verify with Firebase
    FB-->>FBA: 12. Token validation result
    FBA-->>MW: 13. User info or None
    
    alt Token Valid
        MW->>MW: 14a. Create UserContext
        MW->>MW: 15a. Set request.state.user
        MW->>EP: 16a. Call next (endpoint)
        EP-->>iOS: 17a. Success response
    else Token Invalid
        MW-->>iOS: 14b. 401 Unauthorized
    end
```

## 2. Current Breaking Point

```mermaid
graph TD
    A[iOS sends valid token] -->|✅ Working| B[Backend receives request]
    B -->|✅ Working| C[Middleware executes]
    C -->|✅ Working| D[Token extracted]
    D -->|❌ FAILING HERE| E[Firebase Admin SDK verify]
    E -->|Never reaches| F[UserContext created]
    F -->|Never reaches| G[request.state.user set]
    G -->|Never reaches| H[Endpoint accessed]
    
    style D fill:#ff6666
    style E fill:#ff0000
```

## 3. Firebase Token Verification Deep Dive

```mermaid
flowchart LR
    subgraph "Token Verification Process"
        A[Token String] --> B{Firebase Admin Initialized?}
        B -->|No| C[InitError]
        B -->|Yes| D[verify_id_token]
        
        D --> E{Token Format Valid?}
        E -->|No| F[InvalidIdTokenError]
        E -->|Yes| G{Token Expired?}
        
        G -->|Yes| H[ExpiredIdTokenError]
        G -->|No| I{Certificate Valid?}
        
        I -->|No| J[CertificateFetchError]
        I -->|Yes| K{User Exists?}
        
        K -->|No| L[UserNotFoundError]
        K -->|Yes| M{Token Revoked?}
        
        M -->|Yes| N[RevokedIdTokenError]
        M -->|No| O[✅ Return User Info]
    end
```

## 4. Middleware State Management

```mermaid
stateDiagram-v2
    [*] --> RequestReceived
    RequestReceived --> CheckExemptPath
    
    CheckExemptPath --> PassThrough: Path is exempt
    CheckExemptPath --> ExtractToken: Path requires auth
    
    ExtractToken --> VerifyToken: Token found
    ExtractToken --> Return401: No token
    
    VerifyToken --> CreateUserContext: Token valid
    VerifyToken --> Return401: Token invalid
    
    CreateUserContext --> SetRequestState
    SetRequestState --> CallEndpoint
    
    CallEndpoint --> [*]: Response sent
    PassThrough --> CallEndpoint: No auth needed
    Return401 --> [*]: 401 sent
```

## 5. Current System Configuration

```mermaid
graph TB
    subgraph "iOS App"
        A1[TokenManagementService]
        A2[Firebase SDK]
        A3[APIClient]
    end
    
    subgraph "Modal Production"
        B1[FastAPI App]
        B2[Firebase Admin SDK]
        B3[Middleware Stack]
        B4[Firestore Client]
    end
    
    subgraph "Firebase/GCP"
        C1[Firebase Auth Service]
        C2[Service Account]
        C3[JWT Verification]
    end
    
    A1 --> A2
    A2 --> C1
    A3 --> B1
    B1 --> B3
    B3 --> B2
    B2 --> C2
    C2 --> C3
    
    style B2 stroke:#ff0000,stroke-width:4px
```

## 6. Request Lifecycle with Logging

```mermaid
sequenceDiagram
    participant Client
    participant Modal
    participant FastAPI
    participant Middleware
    participant FirebaseAdmin
    participant Endpoint

    Client->>Modal: HTTPS Request
    Note over Client: ✅ Token: eyJhbGci...
    
    Modal->>FastAPI: Forward to app
    Note over Modal: ✅ Request received
    
    FastAPI->>Middleware: Enter middleware
    Note over Middleware: ✅ MIDDLEWARE ACTUALLY RUNNING
    
    Middleware->>Middleware: Extract token
    Note over Middleware: ✅ Token extracted
    
    Middleware->>FirebaseAdmin: verify_id_token()
    Note over FirebaseAdmin: ❌ FAILING SILENTLY
    
    FirebaseAdmin--xMiddleware: None/Error
    Note over Middleware: ❌ No user context set
    
    Middleware-->>Client: 401 Unauthorized
    Note over Client: ❌ Authentication required
```

## Problem Analysis

Based on these flows, the issue is at **Step 10-13** in the first diagram. The Firebase Admin SDK is:

1. **Failing to verify valid tokens**
2. **Not logging the actual error**
3. **Returning None instead of user info**

### Root Causes Could Be:

1. **Service Account Issue**: The Firebase service account JSON in Modal might be invalid or for wrong project
2. **Firebase Admin Not Initialized**: The SDK might not be properly initialized
3. **Certificate Fetch Error**: Can't download Google's public keys to verify JWT
4. **Project ID Mismatch**: Backend using different Firebase project than iOS

### The Smoking Gun:

The Modal logs show the middleware runs but `request.state.user` is never set, which means `verify_id_token()` is returning None/failing.