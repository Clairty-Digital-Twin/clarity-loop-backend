# Complete Authentication Data Flow Analysis

> **Purpose**: Establish the "source of truth" between frontend and backend agents about authentication flows, failure points, and data transformations.

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

## 7. iOS Token Generation Flow

```mermaid
sequenceDiagram
    participant User
    participant iOS as iOS App
    participant TMS as TokenManagementService
    participant FBS as Firebase SDK
    participant FBAuth as Firebase Auth Server

    User->>iOS: Login with email/password
    iOS->>FBS: signIn(email, password)
    FBS->>FBAuth: Authentication request
    FBAuth-->>FBS: User + initial token
    FBS-->>iOS: FirebaseUser object
    
    Note over iOS: Later, when making API call
    iOS->>TMS: getValidToken()
    TMS->>TMS: Check token expiry
    
    alt Token expired or near expiry
        TMS->>FBS: getIDToken(forcingRefresh: true)
        FBS->>FBAuth: Request fresh token
        FBAuth-->>FBS: New ID token (JWT)
        FBS-->>TMS: Fresh token
    else Token still valid
        TMS-->>iOS: Cached token
    end
    
    iOS->>Backend: API request with Bearer token
```

## 8. Backend Token Processing Pipeline

```mermaid
flowchart TB
    subgraph "1. Request Entry"
        A[HTTPS Request] --> B{Modal Proxy}
        B --> C[FastAPI App]
    end
    
    subgraph "2. Middleware Layer"
        C --> D[firebase_auth_middleware]
        D --> E{Path Exempt?}
        E -->|Yes| F[Skip Auth]
        E -->|No| G[Extract Bearer Token]
        G --> H{Token Present?}
        H -->|No| I[Return 401]
        H -->|Yes| J[Parse Token]
    end
    
    subgraph "3. Firebase Verification"
        J --> K[auth_provider.verify_token]
        K --> L{Provider Initialized?}
        L -->|No| M[Initialize Firebase Admin]
        L -->|Yes| N[firebase_auth.verify_id_token]
        M --> N
        
        N --> O{Token Valid?}
        O -->|Invalid Format| P[InvalidIdTokenError]
        O -->|Expired| Q[ExpiredIdTokenError]
        O -->|Revoked| R[RevokedIdTokenError]
        O -->|User Disabled| S[UserDisabledError]
        O -->|Cert Error| T[CertificateFetchError]
        O -->|Valid| U[Return decoded_token]
        
        P --> V[Return None]
        Q --> V
        R --> V
        S --> V
        T --> V
    end
    
    subgraph "4. User Context Creation"
        U --> W[Extract user info]
        W --> X{Has Firestore?}
        X -->|Yes| Y[get_or_create_user_context]
        X -->|No| Z[_create_user_context]
        Y --> AA[Check/Create DB record]
        Z --> AB[Basic UserContext]
        AA --> AB
    end
    
    subgraph "5. Request State"
        AB --> AC[request.state.user = context]
        AC --> AD[Call endpoint]
        F --> AD
        I --> AE[401 Response]
        V --> AE
    end
    
    style N fill:#ff6666
    style V fill:#ff0000
```

## 9. Token Data Transformation

```mermaid
graph LR
    subgraph "iOS Token (JWT)"
        A[JWT Header] --> B[Algorithm: RS256]
        C[JWT Payload] --> D[uid: abc123]
        C --> E[email: user@example.com]
        C --> F[exp: 1736611200]
        C --> G[iat: 1736607600]
        C --> H[auth_time: 1736607600]
        I[JWT Signature] --> J[RSA Signature]
    end
    
    subgraph "Firebase Decoded Token"
        K[decoded_token dict] --> L[uid: abc123]
        K --> M[email: user@example.com]
        K --> N[email_verified: true]
        K --> O[custom_claims: {...}]
        K --> P[exp: 1736611200]
    end
    
    subgraph "User Info Dict"
        Q[user_info] --> R[user_id: abc123]
        Q --> S[email: user@example.com]
        Q --> T[verified: true]
        Q --> U[roles: ['patient']]
        Q --> V[custom_claims: {...}]
    end
    
    subgraph "UserContext Object"
        W[UserContext] --> X[user_id: abc123]
        W --> Y[email: user@example.com]
        W --> Z[role: UserRole.PATIENT]
        W --> AA[permissions: [READ_OWN_DATA, WRITE_OWN_DATA]]
        W --> AB[is_verified: true]
    end
    
    A -.-> K
    K -.-> Q
    Q -.-> W
```

## 10. Modal Deployment Environment

```mermaid
graph TB
    subgraph "Modal Container"
        A[modal_deploy_optimized.py] --> B[FastAPI App Instance]
        C[Environment Variables] --> D[GOOGLE_APPLICATION_CREDENTIALS]
        C --> E[FIREBASE_PROJECT_ID]
        C --> F[ENVIRONMENT=production]
        
        G[Mounted Secrets] --> H[googlecloud-secret]
        H --> I[service-account.json]
        
        B --> J[Middleware Stack]
        J --> K[firebase_auth_middleware]
        K --> L[Firebase Admin SDK]
        L --> M{Uses credentials}
        I --> M
        D --> M
    end
    
    subgraph "External Services"
        N[Google Certificate Servers]
        O[Firebase Auth API]
        P[Firestore Database]
    end
    
    M -.-> N
    M -.-> O
    K -.-> P
    
    style L fill:#ff6666
    style M fill:#ff6666
```

## Problem Analysis

Based on these comprehensive flows, the issue is at the **Firebase Admin SDK verification** step.

### Current Evidence

1. ✅ **iOS generates valid tokens** (confirmed by user)
2. ✅ **Backend receives the token** (logs show token in middleware)
3. ✅ **Middleware executes** (logs show "MIDDLEWARE ACTUALLY RUNNING")
4. ✅ **Token is extracted** (length and preview logged)
5. ❌ **Firebase Admin SDK verification fails** (returns None)
6. ❌ **No specific error logged** (need enhanced logging to see why)

### Root Causes (In Order of Likelihood)

1. **Certificate Fetch Error**:
   - Firebase Admin SDK can't download Google's public keys
   - Usually due to network issues or wrong project configuration
   - Would explain silent failure

2. **Project ID Mismatch**:
   - iOS using different Firebase project than backend
   - Tokens from project A won't verify in project B
   - Check Firebase_PROJECT_ID in Modal vs iOS

3. **Service Account Permissions**:
   - Service account might lack proper permissions
   - Needs "Firebase Authentication Admin" role

4. **Token Format Issue**:
   - Token might be malformed or truncated
   - Check full token is being sent (no length limits)

### Immediate Debugging Steps

1. **Deploy enhanced logging** (already added above)
2. **Check Modal logs** for specific Firebase error
3. **Verify environment variables** in Modal deployment
4. **Test with debug endpoint** (to be created)

### The Smoking Gun

The fact that `verify_id_token()` returns None without throwing means Firebase Admin SDK is catching an exception internally. The enhanced logging will reveal which specific exception.
