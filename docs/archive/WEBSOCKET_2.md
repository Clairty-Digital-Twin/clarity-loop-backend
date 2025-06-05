**1. WebSocket Endpoint Implementation Issues:** The chat WebSocket endpoint appears incomplete and not properly integrated into the FastAPI app. In the `development` branch, the API v1 router includes sub-routers for auth, health data, PAT analysis, and Gemini insights – **but no chat router is included**. This means any `@router.websocket("/chat/{user_id}")` endpoint (likely defined in a `chat.py`) is never mounted, causing 404 responses instead of WebSocket handshakes. This leads to errors like *“Expected WebSocket upgrade”* when clients attempt to connect (the server returns a normal HTTP 404 or 401, not a 101 Switching Protocols). **Routing Fix:** Include the chat router in the API (e.g. `router.include_router(chat_router, prefix="/chat")`) so that `/api/v1/chat/{user_id}` is recognized. Ensure the route path and prefix align with tests (e.g. if tests use `/api/v1/chat/123`, the router’s prefix and path should match).

Additionally, the endpoint’s handshake and lifecycle logic may be misaligned with FastAPI best practices. Each WebSocket endpoint **must accept the connection and handle messaging in a loop**. Verify that the implementation calls `await websocket.accept()` **before** receiving or sending messages. If this is missing, the handshake won’t complete. The handler should then continually receive messages and send responses until disconnect. For example, a proper pattern is:

```python
@router.websocket("/chat/{user_id}")
async def chat_ws(websocket: WebSocket, user_id: str):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            # ... process message ...
            await websocket.send_text(reply)
    except WebSocketDisconnect:
        # ... cleanup ...
        pass
```

Make sure to **catch `WebSocketDisconnect`** to avoid noisy stack traces when clients disconnect. If the current code doesn’t handle this, a client closing the connection can propagate an exception and fail the test. The planned design (per project docs) was to save conversation context on disconnect, so add a `except WebSocketDisconnect:` block to perform any final persistence and then quietly exit. Without this, tests could fail with unhandled `WebSocketDisconnect` exceptions when the client drops.

Authentication for WebSockets is another concern. The backend uses Firebase auth middleware to enforce JWTs on API calls. That same middleware runs on WebSocket connection requests (the initial HTTP Upgrade request). If an Authorization header is missing or invalid, the middleware might abort with an HTTP error (resulting in no upgrade). This could explain tests failing with *“Expected WebSocket upgrade”* – the server likely returned 401/403 due to missing auth, so the client never switched protocols. **Solution:** Ensure the WebSocket route is either exempt from auth or properly requires auth in a way tests can satisfy. FastAPI’s `Depends(get_current_user)` won’t work directly in a websocket signature, so relying on the middleware is fine. Just document that clients must send the `Authorization: Bearer <token>` header in the WebSocket handshake. In testing, make sure to provide this header (see next section). If you intended unauthenticated access for development, consider marking the `/chat` route as exempt in the middleware configuration or adjusting the middleware to not block upgrades. Otherwise, tests and clients need to include a token.

Finally, the chat logic should align with the intended functionality (real-time health chat). The task specs indicate the WebSocket should stream AI responses and maintain context. If the implementation currently sends only a single response or lacks context management, that’s a functional gap (though not a direct test failure, it’s a misalignment with requirements). For completeness and future tests, consider refactoring the handler to use an async generator from the Gemini service for streaming chunks of the AI response. For example, yield partial responses in a loop:

```python
async for chunk in gemini_client.stream_chat_response(...):
    await websocket.send_text(chunk)
```

and accumulate history in a `ConversationContext` object. This ensures responsiveness and mimics real-time streaming. Also validate that `user_id` from the path is used to fetch context or authorize the user (e.g., ensure the JWT user matches the path `user_id` to prevent cross-user access). In short, integrate the WebSocket route, call `accept()`, loop on `receive_text/send_text`, handle disconnects, and follow the planned streaming approach so the implementation meets design and avoids runtime errors.

**2. WebSocket Test Suite Problems (`httpx.AsyncClient` vs. FastAPI TestClient):** The WebSocket tests are failing due to misuse of the HTTP client. The test suite uses `httpx.AsyncClient` for API calls, which **does not support WebSocket upgrades by default**. In `tests/conftest.py`, the `async_client` fixture instantiates an `AsyncClient` with `transport=None` and a base URL, meaning it isn’t bound to the FastAPI app. It should be using the app as an ASGI client. In fact, the project’s own testing guide shows using `AsyncClient(app=app, base_url="http://test")`. The missing `app=app` is a bug. As a result, when tests call `await async_client.get("/api/v1/chat/123")` (or similar) to hit the websocket endpoint, httpx issues a normal HTTP GET to `http://test` without any WebSocket handshake. FastAPI’s router sees a request to a websocket path without an Upgrade header and raises `RuntimeError("Expected WebSocket upgrade")`. This matches the error seen.

**Fix the test client configuration:** Update the `async_client` fixture to mount the FastAPI app. For example:

```python
@pytest.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
```

This way, `async_client` will route requests to the in-memory app, supporting FastAPI’s WebSocket handling. However, even with this fix, httpx alone cannot perform a WebSocket upgrade. **The proper way to test WebSockets is using FastAPI’s TestClient (Starlette TestClient)**, which provides a `websocket_connect` interface. For example:

```python
client = TestClient(app)
with client.websocket_connect(f"/api/v1/chat/{user_id}", headers=auth_headers) as ws:
    ws.send_text("Hello")
    data = ws.receive_text()
    assert data == "Hello, world"  # example expected response
```

The TestClient handles the handshake internally and returns a WebSocket session object. By contrast, `httpx.AsyncClient` does not have a `websocket_connect` method – attempting to call it (or using httpx for a WS URL) will lead to attribute errors or upgrade errors. If any test tried something like `async_client.websocket_connect(...)`, it would throw an `AttributeError` (since `AsyncClient` has no such attribute), which could be one of the “AttributeError” failures observed. **Recommendation:** Refactor WebSocket tests to use `TestClient.websocket_connect`. This may mean writing those tests as standard (non-async) pytest functions, since TestClient runs synchronously. You can still use async fixtures for app setup, but the actual websocket interaction should be done in a normal test function (or within an `asyncio.run` block if absolutely needed). This change will properly exercise the WS endpoints.

**3. Mismatch Between Test Calls and Server Expectations:** The tests must send the correct headers and protocol upgrade to match the server’s requirements. As noted, if the auth middleware is on, the **Authorization header must be included** in the handshake. The test suite defines a `auth_headers` fixture with a dummy Bearer token, but it’s only being applied to HTTP requests (e.g. `client.get()` calls). Ensure that in any websocket test, you pass `headers=auth_headers` to `websocket_connect()` so the JWT is present during the upgrade request. Otherwise, the server will immediately reject the connection. Also, avoid sending irrelevant headers like `Content-Type: application/json` during a WebSocket upgrade – it’s not needed for a handshake (the upgrade uses headers like `Upgrade`, `Connection`, `Sec-WebSocket-Key`, etc.). Including JSON content-type isn’t a standard part of a WS request; if your test client is inheriting that from `auth_headers`, it shouldn’t break the handshake, but it’s unnecessary. It’s safer to pass only the auth header to avoid any odd side effects. In summary, align the test handshake with what the server expects: provide the `Authorization` token and ensure the request is actually a WebSocket upgrade. Using the TestClient as described will automatically handle the low-level details (setting `Upgrade: websocket` headers, etc.).

Another potential mismatch is the path usage. If the route is `/api/v1/chat/{user_id}`, tests should supply a realistic user ID (and that ID should correspond to an authorized user if the server checks it). For example, if using the dummy token, the middleware might set `user_id="test-user-123"`. The test could connect to `/api/v1/chat/test-user-123` to mirror that. If the code currently doesn’t validate that path ID matches the token’s user, consider adding a check server-side (to prevent one user connecting to another’s channel). This wasn’t explicitly mentioned in failures, but it’s a logical safety check to implement.

**4. Analysis of Test Failures and Root Causes:** Let’s break down the specific errors observed:

- **`RuntimeError: Expected WebSocket upgrade`** – **Root cause:** The test made a regular HTTP request to a WebSocket endpoint. This happens when using the wrong client or method (e.g. httpx AsyncClient GET) for an endpoint defined with `@websocket`. FastAPI/Starlette raises this error when a client hits a websocket route without upgrading protocols. In our case, the combination of `async_client` misconfiguration and not using a websocket-aware client triggered this. **Fix:** Use the correct testing approach (TestClient) or at least configure httpx with the ASGI app and a websocket tool. The simplest fix is adopting TestClient’s `websocket_connect`, as described above, so that the handshake occurs and the endpoint is actually invoked.

- **`WebSocketDisconnect` exceptions:** These likely arose from the server side when tests did manage to connect or during teardown. If the test client disconnected (or failed to connect fully) and the server code didn’t handle it, FastAPI may log an exception of type `WebSocketDisconnect`. This can fail a test if not caught. **Root cause:** Not catching disconnects in the endpoint (as discussed in section 1) or a test closing a connection abruptly. **Fix:** Implement a try/except in the endpoint around the receive/send loop to catch `WebSocketDisconnect`. In tests, if using the context manager, the disconnect is handled gracefully on exit; just ensure the server doesn’t treat a normal close as an error. After adding a disconnect handler, rerun tests to confirm the absence of this error.

- **`AttributeError` related to mocks (e.g. missing `MagicMock`):** This refers to test code issues with mocking. One example could be using `MagicMock` in a test without importing it. For instance, a test might do `some_function.return_value = MagicMock()` but only imported `Mock` from `unittest.mock`. This would raise an AttributeError or NameError. We see that `tests/conftest.py` imports `MagicMock` for its own use, but that doesn’t automatically import it in every test module’s namespace. Any test module that wants to use `MagicMock` must import it or use `Mock` (which was imported). **Fix:** Audit the test files for any usage of `MagicMock` or `AsyncMock` where they aren’t imported. Add the appropriate import at the top (`from unittest.mock import MagicMock, AsyncMock`). For example, if an integration test tries to stub out a dependency with `MagicMock()` without import, add that import or change it to use `Mock()` which is already imported in that file. Ensuring all mock classes are imported will resolve those AttributeError failures.

- **Other potential AttributeErrors in WS tests:** As mentioned, calling `async_client.websocket_connect` would be an AttributeError. After switching to `TestClient`, this is resolved. If the tests attempted to call methods on the result of `websocket_connect` incorrectly (for example, treating it like an `httpx.Response` and calling `.json()`), that would also cause attribute errors. The correct usage is to use `ws.send_text()`, `ws.receive_text()`, etc., on the WebSocket session object. Double-check the test assertions – if they assumed a JSON response from a websocket, that’s incorrect (websocket messages are not accessed via `.json()` or `.text` on a Response). They should use the WebSocket interface. Adjust any such misuses in the tests.

**5. Proposed Code-Level Fixes:** Below is a summary of precise changes to apply:

- **Integrate the WebSocket Route:** In `clarity/api/v1/__init__.py`, include the chat router. For example:

  ```python
  from clarity.api.v1 import chat  # import the module if exists
  router.include_router(chat.router, prefix="/chat")
  ```

  This ensures the endpoint is reachable at `/api/v1/chat/{user_id}`.

- **WebSocket Endpoint Improvements:** In the chat endpoint function, add `await websocket.accept()` at the top. Implement the message loop and send replies accordingly. Catch `WebSocketDisconnect` to handle client departure gracefully. For example:

  ```python
  from fastapi import WebSocketDisconnect
  @router.websocket("/chat/{user_id}")
  async def chat_ws(websocket: WebSocket, user_id: str):
      await websocket.accept()
      try:
          while True:
              data = await websocket.receive_text()
              # process data, e.g., get AI response
              await websocket.send_text(response_text)
      except WebSocketDisconnect:
          logger.info(f"Client {user_id} disconnected")
  ```

  If conversation context and streaming are requirements, integrate those as per design (using the `ConversationContext` to store history and streaming chunked responses). This not only fixes test errors but aligns the behavior with expected use (real-time streaming chat).

- **Async Auth in WebSocket:** If needed, verify the JWT on connect. One pattern is to parse a query param or header manually inside the endpoint (since `Depends` can’t easily be used on `websocket`). For example, you could read a token query like `token = websocket.query_params.get("token")` or check headers via `websocket.headers`. However, since the FirebaseAuthMiddleware is already applied globally, you may rely on it (it should abort unauthorized connections before calling your endpoint). In tests, just ensure to provide the header so the middleware permits the connection.

- **Fix `async_client` Fixture:** Change it to use the ASGI app. In `tests/conftest.py`, modify:

  ```diff
  ```

- async with AsyncClient(transport=None, base\_URL="[http://test](http://test)") as ac:

- async with AsyncClient(app=app, base\_URL="[http://test](http://test)") as ac:
  yield ac

  ```
  This connects httpx to the FastAPI app in memory:contentReference[oaicite:15]{index=15}. With this change, normal HTTP calls (`async_client.get/post`) hit the app. (Still, use TestClient for websockets as noted.)

  ```

- **Refactor WebSocket Tests to Use TestClient:** For each test that involves WebSocket communication, use the synchronous TestClient. Example refactor for a failing test:

  ```python
  def test_chat_endpoint_flow(app: FastAPI, auth_headers):
      client = TestClient(app)
      # Include auth header in WS handshake
      with client.websocket_connect("/api/v1/chat/test-user-123", headers=auth_headers) as ws:
          ws.send_text("Hello AI")
          reply = ws.receive_text()
          assert "Hello" in reply   # example assertion
  ```

  Remove any `await async_client.get("/chat…")` calls – those should be replaced with the above approach. This ensures a true WebSocket upgrade takes place and allows two-way communication in the test. If the test must run in an async context (due to other async fixtures), you can either isolate the WS test as a purely sync test, or use `anyio.from_thread` utilities to run TestClient calls in a thread. Simpler is to keep WS tests synchronous.

- **Import MagicMock in Tests:** Find tests where `MagicMock` or `AsyncMock` is used without import. Add `from unittest.mock import MagicMock` at the top of those files. For instance, if a test does `patch(..., return_value=MagicMock(...))`, include the import. This prevents `NameError`/`AttributeError`. In many cases, using `Mock()` might suffice, but if the test specifically needs MagicMock (which auto-creates attributes on-the-fly), include it. Example:

  ```python
  # At top of test module
  from unittest.mock import MagicMock, AsyncMock
  ```

  This addresses the “missing import” issue noted.

After applying these fixes, rerun the test suite. The WebSocket tests should establish proper connections (no more *Expected WebSocket upgrade* runtime errors), and the chat endpoint should handle connects/disconnects cleanly (preventing uncaught `WebSocketDisconnect`). In summary, the changes align the implementation with FastAPI’s async WebSocket best practices – ensuring the endpoint is correctly registered, handling auth and disconnects, and using the right tools in tests to simulate WebSocket clients.

**Sources:**

- Project API router setup (missing chat route)
- Testing fixture misconfiguration (AsyncClient without app) vs. intended usage
- Planned WebSocket streaming logic for chat (documentation snippet)
