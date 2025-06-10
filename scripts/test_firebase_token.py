#!/usr/bin/env python3
"""Test Firebase token verification directly."""

import os
import sys

import firebase_admin
from firebase_admin import auth, credentials


def test_token_verification(token: str):
    """Test token verification with detailed error reporting."""
    try:
        # Initialize Firebase Admin SDK if needed
        try:
            app = firebase_admin.get_app()
            print(f"✅ Using existing Firebase app: {app.name}")
        except ValueError:
            # Try to initialize with environment credentials
            if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                cred = credentials.ApplicationDefault()
                app = firebase_admin.initialize_app(cred)
                print("✅ Initialized Firebase with Application Default Credentials")
            else:
                print("❌ No GOOGLE_APPLICATION_CREDENTIALS found")
                return

        print(f"\n🔍 Testing token (length: {len(token)})")
        print(f"🔍 Token preview: {token[:20]}...{token[-20:]}")

        # Try to verify the token
        try:
            decoded = auth.verify_id_token(token, check_revoked=True)
            print("\n✅ Token verified successfully!")
            print(f"  - UID: {decoded.get('uid')}")
            print(f"  - Email: {decoded.get('email')}")
            print(f"  - Project: {decoded.get('firebase', {}).get('sign_in_provider')}")
            print(f"  - Issued at: {decoded.get('iat')}")
            print(f"  - Expires at: {decoded.get('exp')}")
            print(f"  - Audience: {decoded.get('aud')}")
            print(f"  - Issuer: {decoded.get('iss')}")
        except auth.InvalidIdTokenError as e:
            print(f"\n❌ Invalid token: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Details: {e!s}")
        except auth.ExpiredIdTokenError as e:
            print(f"\n❌ Expired token: {e}")
        except auth.RevokedIdTokenError as e:
            print(f"\n❌ Revoked token: {e}")
        except auth.UserDisabledError as e:
            print(f"\n❌ User disabled: {e}")
        except Exception as e:
            print(f"\n❌ Unexpected error: {type(e).__name__}")
            print(f"   Details: {e!s}")
            import traceback

            traceback.print_exc()

    except Exception as e:
        print(f"❌ Failed to initialize Firebase: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        print("Usage: python test_firebase_token.py <token>")
        sys.exit(1)

    test_token_verification(token)
