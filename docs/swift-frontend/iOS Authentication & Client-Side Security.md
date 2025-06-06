# **CLARITY Pulse iOS App – Secure Authentication & Client-Side Security Blueprint**

## 1. Firebase SDK Integration (SwiftUI, iOS 17+)

1. **Set up Firebase Project & Auth:** Create a Firebase project and enable Email/Password sign-in (and any other providers needed) in the Firebase console. Download the `GoogleService-Info.plist` for the iOS app and add it to your Xcode project.
2. **Add Firebase SDK via Swift Package Manager:** In Xcode, go to *File > Add Packages* and add the Firebase Apple SDK (`https://github.com/firebase/firebase-ios-sdk.git`). Select **FirebaseAuth** (and any other required Firebase products). Xcode will fetch the package and integrate it.
3. **Project Configuration:** In your app target’s *Signing & Capabilities*, enable **Keychain Sharing** (this allows Firebase to store credentials in the Keychain). Also ensure your Info.plist contains any needed URL schemes (for example, the reversed client ID if using Google/Apple sign-in).
4. **Initialize Firebase in SwiftUI App:** Import Firebase in your app entry point and configure it at launch. For example, in the `@main` App struct or App Delegate, call:

   ```swift
   import FirebaseCore
   @main
   struct ClarityPulseApp: App {
       init() {
           FirebaseApp.configure()  // Initialize Firebase SDK:contentReference[oaicite:4]{index=4}
       }
       // ...
       var body: some Scene {
           WindowGroup {
               ContentView().environmentObject(AuthViewModel())
           }
       }
   }
   ```

   This ensures Firebase is set up before you use Auth or other services. (If using the SwiftUI App lifecycle without an AppDelegate, calling `FirebaseApp.configure()` in `init()` as shown is a convenient approach.)
5. **Verify Setup:** Upon launching the app, you should be able to call Firebase APIs. For example, try creating a test user or logging in (see flows below). Firebase will handle device-specific setup (e.g. using the iOS Keychain to persist auth credentials by default). Ensure that API calls succeed and no Firebase configuration errors appear in Xcode’s console.

## 2. Authentication Flows (Registration, Login, Verification, Reset, Logout)

Implement authentication flows using SwiftUI views backed by an **AuthViewModel** (MVVM) or an equivalent state manager (like TCA’s reducer). The view model will interact with Firebase Auth and publish user session state, allowing the UI to reactively update. For example, your `ContentView` can switch between auth screens and the main app based on whether a user is logged in.

* **Registration (Email/Password):** Provide a SwiftUI form for new users to enter email, password, and any required info. On submit, call Firebase to create the account:

  ```swift
  try await Auth.auth().createUser(withEmail: email, password: password)
  ```

  This returns a `AuthDataResult` with the new user, and Firebase automatically signs them in. After a successful signup, send a verification email:

  ```swift
  Auth.auth().currentUser?.sendEmailVerification { error in /* ... */ }
  ```

  This triggers Firebase to email a confirmation link to the user. In the UI, prompt the user to check their email. You can require email verification before allowing login: check `Auth.auth().currentUser?.isEmailVerified` – if false, perhaps restrict access or show a reminder. (On the backend, also enforce that email is verified before granting data access for security.)

* **Login (Email/Password):** Create a login view with fields for email and password. On submit, call:

  ```swift
  Auth.auth().signIn(withEmail: email, password: password) { result, error in ... }
  ```

  This signs in the user if credentials are correct. Handle error cases (e.g. wrong password, no user found) by displaying messages. If the account is not verified (as above), you can detect `user.isEmailVerified` after sign-in and navigate accordingly (e.g. prompt them to verify). Upon successful login, the AuthViewModel should update the published `userSession` (e.g. to `Auth.auth().currentUser`) which triggers the UI to transition into the authenticated part of the app.

* **Email Verification Flow:** The verification email sent during registration contains a link for the user to confirm their address. Typically, the user taps the link which either opens your app or a browser. **SwiftUI Deep Link Handling (Optional):** If you configure a custom URL scheme or Firebase Dynamic Link for email verification, your app can catch the verification event. Firebase Auth will update the user’s `isEmailVerified` status. You can refresh the user (e.g. call `Auth.auth().currentUser?.reload()`) when the app opens via that link to update their status. If not handling in-app, simply inform the user to restart the app after verifying. Always ensure on the backend that unverified emails don’t fetch protected data.

* **Password Reset:** Provide a “Forgot Password” option on the login screen. When pressed, prompt for the user’s email and call:

  ```swift
  Auth.auth().sendPasswordReset(withEmail: email) { error in /* ... */ }
  ```

  Firebase will email a secure link to reset the password. Display a confirmation to the user (e.g. “Check your email for reset instructions”). No further action is needed in-app; the user will follow the email link to choose a new password, after which they can log in normally.

* **Logout:** To log out, simply call `try Auth.auth().signOut()` (catching any errors). This clears the Firebase session and credentials on the device. In addition, **clear any sensitive client data** on logout: e.g. reset your view model state, and if you cached any user data locally (in SwiftData or elsewhere), consider wiping or encrypting it if not needed after logout. After sign-out, the app should return to a login screen. (Because Firebase stored tokens in Keychain, calling signOut also removes them from Keychain, preventing reuse without re-authentication.)

**SwiftUI Implementation Tips:** Use `@State` and `@EnvironmentObject` to bind form fields and view model state. For asynchronous Auth calls, you can use `Task { ... }` with `await` (Firebase Auth provides completion handlers; you can wrap them with `async`/`await` or use the Combine publishers if available). The above examples show both closure-based and async usage. Also consider using `Auth.auth().addStateDidChangeListener` to monitor auth state changes globally – in SwiftUI, the AuthViewModel can set up this listener to update `userSession` when Firebase signs in or out a user (for instance, if the user’s token is revoked in the background).

## 3. Token Management (JWT Storage & Refresh)

**Secure Storage of JWTs:** Firebase Authentication issues a short-lived **ID Token** (JWT) and a long-lived **Refresh Token** upon sign-in. The ID token (a JWT) is what you’ll send to the CLARITY backend to authenticate requests, and is valid for about 1 hour. The refresh token is used by the client to get new ID tokens after expiry. These tokens **must be stored securely**. **Do not store JWTs or refresh tokens in UserDefaults or plaintext files.** Instead, leverage the iOS Keychain, which encrypts data at rest and ties access to the device security. The Firebase iOS SDK by default persists the user's credentials in the Keychain for you. (That’s why a returning user stays logged in – the SDK finds a stored refresh token in Keychain.) Ensure Keychain usage is enabled as noted (Keychain Sharing capability) so the tokens can be saved. By default, Firebase marks these items to be accessible *after* device unlock. For additional security, you might mark them accessible only when the device is unlocked and not to be migrated to backups (see Biometric section below for using Keychain access control flags).

**Handling Expired Tokens:** Since ID tokens expire hourly, your app should be prepared to fetch a new token when needed. Firebase makes this easy: whenever you call `Auth.auth().currentUser?.getIDToken()` (or any API call that requires a fresh token), the SDK will automatically use the refresh token to get a new ID token if the current one is expired. You can force a refresh by adding the parameter (e.g. `getIDTokenForcingRefresh(true)` in completion handler API). In practice, before calling a protected backend API, retrieve the current user’s ID token:

```swift
Auth.auth().currentUser?.getIDToken(completion: { token, error in 
    sendRequest(authToken: token)
})
```

This ensures you send a valid, non-expired JWT. Never rely on a cached token string without refreshing if it might be expired. As a rule, **do not use the Firebase UID alone to authenticate to your backend** – always use the JWT in an Authorization header (the backend will verify it). The refresh token itself is not sent to your backend; it stays on the device.

**Auto-Renewal and Revocation:** Implement a global handler for authentication status. For example, if a backend call returns an HTTP 401 (Unauthorized) due to an expired or invalid token, your app can intercept that response. The handler can then attempt to refresh the token (call `getIDToken(forceRefresh: true)`) and retry the request with the new token. If the refresh fails (e.g. if the refresh token was revoked or user disabled), that indicates the session is no longer valid – in this case, prompt the user to log in again. It’s good practice to centrally manage this logic (for instance, in your `NetworkManager` or using a custom `URLProtocol` or combine `.retry` operator) so that individual API calls don’t all need to handle token expiry. Keep in mind that refresh tokens can be revoked by backend action (password change, account deletion, admin revocation) – the Firebase SDK will then treat the user as signed out on the next action. Your Auth state listener (if used) would fire and your app should respond by transitioning to the login screen.

Finally, store *only what is needed*. You generally do **not** need to manually cache the ID token string persistently – just call `getIDToken()` when needed. The refresh token is secretly stored by Firebase for you. If you choose to manage tokens yourself (advanced cases), **use Keychain** with proper accessibility. For example, use `kSecAttrAccessibleWhenUnlocked` to allow use only when the phone is unlocked by the user. For maximum security, require biometric or passcode to retrieve the token (discussed next). The principle is to minimize exposure of credentials in memory or storage.

## 4. Authenticated API Requests (JWT Usage in Networking)

All communication with the CLARITY custom backend should include the Firebase-issued JWT for authentication. The backend expects this token for protected endpoints (as noted, e.g. `Authorization: Bearer <JWT>`). Here’s how to implement secure API calls from the app:

* **Attaching JWT to Requests:** Whenever you call the CLARITY backend (e.g. via `URLSession` or a third-party HTTP client), attach the current Firebase ID token in an HTTP header. The standard is an `Authorization` header with the Bearer schema:

  ```swift
  var request = URLRequest(url: URL(string: baseURL + "/api/v1/health-data")!)
  request.httpMethod = "GET"
  let token = try await Auth.auth().currentUser?.getIDToken()  // ensure not expired
  request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
  // proceed to perform request...
  ```

  Here we use `getIDToken()` to fetch a fresh token if needed, then include it. Always use **HTTPS** for all backend communications – this is mandatory for HIPAA (encryption in transit) and also required by App Transport Security by default. With HTTPS, the JWT (which is sensitive) will be encrypted in transit. Avoid sending tokens as URL query parameters since those can be logged; headers are preferable.

* **Request Interception & Token Refresh:** Implement a central interceptor to handle unauthorized responses. For example, if a request returns a 401 Unauthorized, your code can automatically attempt to refresh the token and retry:

  ```swift
  if response.statusCode == 401 {
      Auth.auth().currentUser?.getIDTokenForcingRefresh(true) { newToken, err in
          if let newToken = newToken {
              // Update header and retry request once
          } else {
              // Refresh failed (session invalid), force logout
          }
      }
  }
  ```

  This logic can be in a URLSession delegate, a custom URLProtocol, or simply wrapped in a function that all network calls go through. Ensure that only one refresh happens at a time (to avoid token race conditions). After refreshing, repeat the request. If it still fails or refresh is not possible (e.g. no currentUser), then prompt the user to log in again – the session has expired or been revoked.

**Security considerations:** Do not log the token or expose it in UI. Also, consider **token lifecycle** – e.g., if the app goes to background for a long time, the next foreground request might need a refresh. You can proactively refresh the token when the app comes back online after a long interval, or simply handle it on-demand as above. By using Firebase’s provided methods, the heavy lifting of token management is done for you. The backend will use Firebase Admin SDK to validate the JWT on each request, ensuring the token is authentic and not expired (this is already in place given the backend uses Firebase JWT verification).

## 5. Biometric Authentication (Face ID/Touch ID Unlock)

To add an extra layer of security, especially to protect PHI within the app, implement biometric authentication **after the user has logged in** (this is analogous to “unlocking” the app with Face ID, on top of their Firebase login). The idea is: user logs in with email/password (or other method) normally, and then opts in to enable biometric lock for subsequent app access.

**Setup Face ID/Touch ID:** Using the **LocalAuthentication** framework, you can check if biometrics are available:

```swift
let context = LAContext()
if context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: nil) {
    // Biometric auth is available (Face ID or Touch ID)
}
```

Present a UI toggle or prompt to enable “Face ID unlock”. In Info.plist, include `NSFaceIDUsageDescription` with a message explaining why you use Face ID (required to access Face ID).

**Securing Keychain with Biometric Access:** Tie the user’s session token to biometric authentication by storing it in the Keychain with an access control policy. Specifically, use **`SecAccessControlCreateWithFlags`** to add the `.userPresence` or `.biometricAny` requirement. For example:

```swift
import Security
let access = SecAccessControlCreateWithFlags(nil, 
                 kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly, 
                 .userPresence, nil)!   // Require user presence (biometric or passcode):contentReference[oaicite:27]{index=27}

let tokenData = tokenString.data(using: .utf8)!
let query: [CFString: Any] = [
    kSecClass: kSecClassGenericPassword,
    kSecAttrAccount: "SessionToken",
    kSecAttrService: "ClarityPulse",             // app-specific service name
    kSecAttrAccessControl: access,
    kSecValueData: tokenData
]
let status = SecItemAdd(query as CFDictionary, nil)
```

In this example, the token (could be the refresh token or an encryption key for sensitive data) is stored such that it **can only be read if the device has a passcode set and the user authenticates via Face ID/Touch ID at the time of access**. The flag `kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly` ensures the item isn’t accessible on devices without a passcode and isn’t synced to iCloud or backups (it’s tied to this device only). The `.userPresence` requirement means any read attempt will prompt for biometric authentication or passcode.

**Using the Biometric-Protected Token:** On app launch, instead of automatically going to the main content if a Firebase user is cached, you first attempt to read the protected item. This will trigger the Face ID prompt: “Unlock CLARITY Pulse to continue”. On successful Face ID, retrieve the token and use it to re-authenticate or decrypt data. For instance, you might store the **refresh token** in this secure manner. Upon Face ID unlock, retrieve it and call `Auth.auth().signIn(withCustomToken:)` or `signIn(withEmail:password:)` if you saved creds – but since Firebase SDK already persists the session, a more common approach is to use biometric unlock as a **UI lock**, not fully logging out of Firebase. In practice, you can simply check for presence of your keychain item: call `SecItemCopyMatching` with the same query (without `kSecValueData` but with `kSecReturnData`) – the system will automatically prompt for Face ID, and if the user cancels or fails, you can then treat it as a locked state (hide sensitive UI or require manual login as fallback).

**Fallbacks and UX:** If Face ID fails or the user cancels, you can allow a fallback to device passcode (the `.userPresence` flag already falls back to iPhone passcode by default). If they fail that too, you may log them out for safety or keep the app on a locked screen. Make sure to handle the LAContext error cases (user cancel, etc.) and decide when to allow retries or not. It’s also wise to provide a way to disable the biometric lock (maybe in settings, after a successful unlock).

In summary, biometric auth in this context acts as a **second factor** to protect the app’s data from unauthorized access even if someone gains access to the device while it’s unlocked. It’s optional for the user but highly recommended for safeguarding PHI. From a HIPAA perspective, this adds an additional *authentication* safeguard on top of standard login, which is a plus.

## 6. SwiftData Security (Local Data Protection)

**Secure Local Storage:** If the app uses SwiftData (the new Core Data replacement in iOS 17) to cache or persist user data (which may include PHI like health metrics), it is critical to protect that data at rest. **iOS Data Protection** provides file-level encryption out of the box. By default, files in the app’s Documents or App Support directory are encrypted with NSFileProtectionCompleteUntilFirstUserAuthentication (accessible after the first device unlock). For maximum security, you should elevate this to **Complete Protection**. This means the app’s data files are only accessible when the device is unlocked by the user. You can enable this by setting the “Data Protection” capability in Xcode (which by default sets the default to NSFileProtectionComplete for new files) or by programmatically specifying file protection on the store file.

If SwiftData uses a Core Data store under the hood, you can retrieve the SQLite file (likely in your Application Support directory) and apply NSFileProtectionComplete to it. For example:

```swift
let storeURL = URL(... path to SQLite ...)
try FileManager.default.setAttributes([.protectionKey: FileProtectionType.complete], 
                                      ofItemAtPath: storeURL.path)
```

This will ensure the OS encrypts the database file when the device is locked. Going forward, any writes will maintain that protection. Alternatively, when configuring the SwiftData container if such an API exists, you could specify the protection level on the NSPersistentStoreDescription. Always test that your data is still accessible when the app is unlocked, and observe that if you lock the device, the data can’t be read (which is the goal – iOS will decrypt on unlock).

**Encrypted Containers:** iOS’s hardware encryption (tied to the user passcode) is very robust. It meets HIPAA’s requirements for encryption at rest (as long as the user has a passcode set). In most cases, using NSFileProtectionComplete is sufficient for HIPAA compliance on-device. Additional encryption layers are generally not needed **unless** you anticipate scenarios where data might reside outside the protected file system (for example, backing up to iCloud in an unencrypted form – which iCloud backup actually is encrypted, but Apple holds keys – or syncing data to other storage). If you want to add an extra layer for defense-in-depth:

* You could encrypt particularly sensitive fields before saving to SwiftData using CryptoKit (e.g. encrypt a string using a key stored in the Keychain). This way, even if the database were somehow accessed, the field is gibberish without the key. This can be overkill but might be considered for things like encryption of a user’s personal identifiers.
* Another approach is to use an encrypted database like **SQLCipher** for Core Data. SQLCipher can transparently encrypt the SQLite with a key you provide. This key could be derived from the user’s passcode or stored in Keychain. However, this is an external library and adds complexity; with iOS’s built-in protection, SQLCipher is often unnecessary unless policy dictates dual encryption.

**Protect Data in Transit & Backups:** Ensure that any files or exports of PHI use encryption. If the user can export data or if the app generates any PDF/CSV of health data to share, use encryption or iOS’s secure APIs to do so. Also consider marking sensitive files with the “do not backup” attribute (`URLResourceValues.isExcludedFromBackup`) to prevent them from going to iCloud backups, if that aligns with your data retention policy. This isn’t strictly required by HIPAA if iCloud is considered a secure business associate, but some organizations prefer PHI not leave the device at all.

In short, **all PHI stored on the device should be encrypted at rest** – leveraging iOS file protection is the easiest way to achieve this, and meets the requirement that ePHI be encrypted to safe standards when stored. The app should also only store the minimum necessary data on the device. If certain sensitive info isn’t needed offline, consider not caching it at all.

## 7. HIPAA Compliance Considerations (iOS Frontend)

Building a HIPAA-compliant app involves not just encryption and auth, but also UX decisions to prevent inadvertent data exposure. Below are best practices on the iOS frontend to uphold privacy:

* **Screen Lock/Blur on Background:** When the app moves to the background or the user switches apps, iOS takes a snapshot of your UI for the app switcher. To prevent any sensitive info from being visible in that snapshot, implement a blur or overlay when the app is backgrounded. In SwiftUI, you can do this by observing the scene phase and layering a privacy screen. For example, use an `@Environment(\.scenePhase)` in your top-level view and on `.onChange` to `.inactive`/`.background`, overlay a full-screen view (like your app logo or just a blurred rectangle) over all content. Remove it when the app becomes active. This ensures that in the iOS task switcher, no PHI (like a health metric or user name) is readable. Many healthcare apps do this as part of HIPAA best practices. (Alternatively, set `UIView.isHidden` or use UIWindow-level code in AppDelegate’s `applicationWillResignActive` to hide content.) The key is to **hide or obscure sensitive UI when the app isn’t foregrounded**.

* **Secure Text Entry & No Autofill:** For fields that contain sensitive personal data (beyond just passwords), use the appropriate secure controls. Always use `SecureField` for passwords (this prevents iOS from caching the text or showing it in notifications, and it disables iOS screen recording for that field). For other PHI like maybe Social Security Number or medical record numbers, consider marking the textContentType appropriately or using secure fields so iOS doesn’t try to autofill or predictively learn them. Also, **never log sensitive field values**. For example, don’t print the password or any PHI to Xcode console, as even debug logs could be a liability.

* **Logging and Analytics:** Ensure that any analytics or crash reporting in the app is configured **not** to send PHI. For instance, if using Firebase Analytics or Crashlytics, avoid logging user identifiers or health info. Use GDPR-compliant modes or disable them entirely if not needed. If you do use OSLog for debugging, use the “private” flag for any user data (e.g. `logger.log("User weight = \(weight, privacy: .private)")`), which will redact the actual value in logs. Best practice is to avoid logging PHI at all. Also, strip screenshots or views that might contain PHI from any bug report the user might generate.

* **Prevent Data Leakage in UI/Background:** Other iOS features to consider: by default, Siri and system widgets won’t have access to your app’s data unless you explicitly donate it. So, for instance, ensure you’re not carelessly exposing health info to Siri or Search indices. Do not store PHI in pasteboard (and if you do, clear it or use the general pasteboard with expiration). If your app uses notifications to remind or show data, **don’t include sensitive details in notification content** – keep it generic (e.g. “You have a new message” rather than “Your lab result ABC is ready”), because notifications can show on the lock screen. Mark push notifications as “hidden previews” if possible or let the user decide to hide previews.

* **Session Timeout (Auto-Logout):** As a further safeguard, implement an **idle timeout** for the app. This means if the app has been idle (no user interaction) for a certain period (say 5 or 10 minutes), you automatically lock the app (require re-authentication or at least re-biometric). This reduces risk of someone picking up the device that’s unlocked and accessing data. Enforcing automatic logout or lock after inactivity is recommended under HIPAA guidelines. You can track user interaction timestamps or use an `Timer` that resets on activity. In SwiftUI, you might watch for `scenePhase` changes: if the app was in background for more than X minutes, require login again. The Blurify HIPAA guide explicitly notes “enable automatic logout” as a best practice to protect data. Shorter sessions mean less opportunity for unauthorized access.

* **Access Control and Roles:** On the frontend, ensure that the UI only shows data the user is allowed to see. (This is more of a backend concern, but if your app has, say, different user roles or family accounts, make sure switching accounts or roles doesn’t leave remnants of the previous context on screen.)

* **Jailbreak/Device Security:** While not strictly a UI issue, be aware of the environment. If needed, the app can detect if it’s running on a jailbroken device (which could indicate the OS security is compromised) and refuse to run or at least warn the user. This is often done in high-security apps. It’s not a formal HIPAA requirement, but it’s a defense measure.

* **User Privacy Features:** Provide users with controls over their data within the app – for instance, the ability to logout (we have that), to clear locally stored data (maybe a “Clear cache” option if needed), and to understand that their data is protected. From a UX perspective, these features build trust that their sensitive health data is handled carefully.

* **Testing for Privacy:** As part of development, test scenarios like: rotate to background and come back (is data hidden?), take a screenshot (iOS will allow it, but ensure no special content is inadvertently shown to other apps), and multi-tasking (on iPad, your app could be in a multi-window environment; ensure sensitive data isn’t visible in widget form or in the app switcher preview).

By following these steps, the app’s frontend will minimize the exposure of PHI. In essence, assume that **any data shown on screen could be seen by an unauthorized person** and take measures (blur, lock, hide) to reduce that risk when the app is not actively in use. Combine this with strong authentication and encryption to fully secure the client side.

## 8. Session Management Best Practices

Session management ties together many of the above pieces to ensure a smooth **and secure** user experience. Key considerations include keeping the user logged in when appropriate, but also expiring sessions safely when needed:

* **Persisting Sessions Across Restarts:** Firebase Auth will keep the user’s session by default. As long as the user hasn’t signed out, `Auth.auth().currentUser` will be non-nil on app launch (the credentials are pulled from Keychain). Leverage this to skip login on app launch if possible. For example, your AuthViewModel can check at init: `isLoggedIn = (Auth.auth().currentUser != nil)` and the UI can navigate accordingly. However, be mindful of token expiration – if the app was closed for more than an hour, the stored ID token is expired. Firebase will transparently fetch a new one using the refresh token when needed, but you might proactively call `getIDToken` once at startup to warm it up (and catch any errors if, say, the refresh token was revoked).

* **Token Expiration Handling:** As described, implement global handling for expired tokens. This typically means on any API call failure due to auth, refresh and retry (section 4). Also consider background refresh: If your app performs background fetches (perhaps via `backgroundTasks` or push notifications), ensure you call `getIDToken` in those scenarios to get a valid token (Firebase can refresh even in background if the refresh token is present and network is available). If background refresh is not critical, it’s simpler to just handle it when the app becomes active.

* **Auto-Logout and Timeouts:** Define an **idle timeout** policy based on your organization’s rules (for example, logout after 15 minutes of inactivity). You can implement this by recording `Date()` of the last user interaction (tap, scroll, network call, etc.). If the app goes to background, calculate the inactivity duration. If it exceeds your threshold, you can either log the user out (`Auth.auth().signOut()`) or lock the UI behind the biometric/passcode screen (if enabled). Many apps choose to just require re-authentication (biometric or password) after a timeout rather than fully signing out, to balance security and convenience. In either case, the **“automatic logout after inactivity”** is a strong safeguard – it ensures even if a user leaves their phone unattended while unlocked, the app won’t expose data for long.

* **Session Invalidations:** Use Firebase’s capabilities to respond to security events. For example, if the backend (or an admin) revokes the user’s tokens (maybe due to a reported lost device or credential change), you should detect that. In practice, the next token refresh will fail – at that point, treat it as a logout event. Ensure your app cleans up and shows the login screen. It’s good to maybe show a message like “For your security, your session has expired. Please log in again.” This covers scenarios like password reset (which Firebase treats as revocation of other sessions).

* **Logout Flow:** When a user manually logs out, as noted, clear all sensitive data. This includes removing any cached data in SwiftData (you might delete the user’s local database or at least wipe PHI rows) unless you need to keep it (if you do keep it for next login, ensure it’s encrypted as per section 6). Also reset your view state (maybe pop to root view). Because the JWT and refresh token are in Keychain, `Auth.signOut()` takes care of removing them, but if you stored anything else (like our biometric Keychain item), you should remove those too on a full logout (especially if the user is ending their session on a shared device scenario).

* **Use of MVVM/TCA for Session State:** Architecturally, manage the session with a single source of truth. In MVVM, that’s your AuthViewModel (with perhaps an `@Published var currentUser`). In TCA, that could be part of your AppCore state (e.g. an enum for auth vs authenticated state). This makes it easy to test session transitions. For example, you can unit test that when a `LogoutButtonTapped` action is sent, the state resets and the user is navigated to login. A modular design will allow you to isolate the authentication logic (Firebase calls can be in a small AuthService class injected for testability). This way you can simulate token expiration events, biometric auth success/failure, etc., in tests to ensure the app responds correctly. Keeping the session logic decoupled also means you can swap Firebase with another auth provider in the future without tangling it through the UI.

* **Idle Session UI/UX:** If using Face ID to unlock the app (as above), the idle timeout can be implemented by requiring Face ID on resume after X minutes. This gives a good user experience (just a glance to unlock) while maintaining security. If the user fails biometric, you can fall back to asking for their password to re-login (multi-factor style). Make sure these flows are intuitive.

In conclusion, session management is about finding the right balance: **don’t frustrate the user with too-frequent logins, but don’t keep sensitive data open indefinitely.** Using persistent logins with Keychain, combined with biometric unlock and auto-timeouts, achieves both security and convenience. All sessions should ultimately obey the rule of least privilege and time-limited access – which, practically, our implementation does by expiring tokens and logging out after inactivity. By following this blueprint – integrating Firebase Auth correctly, storing tokens safely, securing local data, and guarding the UI – the CLARITY Pulse iOS app will be well-positioned to protect user data in compliance with HIPAA and industry best practices.

**Sources:**

* Firebase iOS Integration Guide, Firebase Auth Tutorial
* Firebase Auth Usage (createUser, signIn, email verify, reset)
* Firebase Token Management and Security
* Apple Security (Keychain & Biometrics)
* iOS Data Protection (File Protection)
* HIPAA Compliance Guidance

