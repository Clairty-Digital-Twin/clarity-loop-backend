# Architecture & Dependency Guidelines

## Professional Practices to Prevent Circular Imports

### 1. **Dependency Inversion Principle (DIP)**

```
High-level modules should not depend on low-level modules.
Both should depend on abstractions.
```

**Implementation:**

- Define interfaces in `core/interfaces.py`
- Implement concrete classes in their respective modules
- Inject dependencies rather than importing directly

### 2. **Layered Architecture**

```
Presentation Layer (API Routes)
    ↓ (depends on)
Application Layer (Services)
    ↓ (depends on)
Domain Layer (Models/Business Logic)
    ↓ (depends on)
Infrastructure Layer (Database/External APIs)
```

**Rules:**

- Higher layers can depend on lower layers
- Lower layers NEVER depend on higher layers
- Use interfaces to invert dependencies when needed

### 3. **Module Organization**

```
src/clarity/
├── core/
│   ├── interfaces.py       # Abstract interfaces (no dependencies)
│   ├── config.py          # Configuration (minimal dependencies)
│   └── exceptions.py      # Custom exceptions (no dependencies)
├── models/               # Domain models (depend only on core)
├── services/            # Business logic (depend on models + interfaces)
├── storage/            # Infrastructure (implements interfaces)
├── auth/              # Authentication (implements interfaces)
├── api/              # API routes (depend on services)
└── main.py          # Application factory (orchestrates everything)
```

### 4. **Dependency Injection Container**

```python
# Example: core/container.py
from dependency_injector import containers, providers
from clarity.core.interfaces import IAuthProvider
from clarity.auth.firebase_auth import FirebaseAuthProvider

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    auth_provider = providers.Singleton(
        FirebaseAuthProvider,
        credentials_path=config.firebase.credentials_path,
        project_id=config.firebase.project_id,
    )
```

### 5. **Import Guidelines**

**✅ DO:**

```python
# Use dependency injection
def create_auth_middleware(auth_provider: IAuthProvider):
    return AuthMiddleware(auth_provider)

# Use factory functions
def create_application() -> FastAPI:
    app = FastAPI()
    container = Container()
    # Configure dependencies
    return app

# Import at function level when needed
def get_user_service():
    from clarity.services.user_service import UserService
    return UserService()
```

**❌ DON'T:**

```python
# Module-level imports that create cycles
from clarity.main import app
from clarity.services.user_service import user_service

# Direct circular references
from clarity.api.routes import router
```

### 6. **Testing Architecture**

```python
# tests/conftest.py - Dependency injection for tests
@pytest.fixture
def auth_provider():
    return MockAuthProvider()

@pytest.fixture
def app(auth_provider):
    return create_test_application(auth_provider=auth_provider)
```

### 7. **Static Analysis Tools**

Add to your CI/CD pipeline:

```bash
# Check for circular imports
pip install pydeps
pydeps src/clarity --show-deps

# Use import-linter
pip install import-linter
import-linter --config=.import-linter.toml
```

### 8. **Configuration File (.import-linter.toml)**

```toml
[tool.importlinter]
root_package = "clarity"

[[tool.importlinter.contracts]]
name = "Prevent circular imports"
type = "forbidden"
source_modules = ["clarity"]
forbidden_modules = ["clarity"]
ignore_imports = [
    "clarity.main -> clarity.core.config",
]

[[tool.importlinter.contracts]]
name = "Layer dependencies"
type = "layers"
layers = [
    "clarity.api",
    "clarity.services", 
    "clarity.models",
    "clarity.core",
]
```

### 9. **Factory Pattern Implementation**

```python
# main.py - Clean factory without circular imports
def create_application() -> FastAPI:
    # 1. Create core components
    settings = get_settings()
    
    # 2. Create infrastructure
    auth_provider = create_auth_provider(settings)
    db_client = create_db_client(settings)
    
    # 3. Create services
    services = create_services(db_client, auth_provider)
    
    # 4. Create FastAPI app
    app = FastAPI(title="CLARITY Digital Twin Platform")
    
    # 5. Configure middleware
    configure_middleware(app, auth_provider, settings)
    
    # 6. Include routers
    configure_routes(app, services)
    
    return app
```

### 10. **Monitoring & Prevention**

**Pre-commit Hooks:**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-circular-imports
        name: Check for circular imports
        entry: python -m scripts.check_circular_imports
        language: system
        pass_filenames: false
```

**GitHub Actions:**

```yaml
# .github/workflows/architecture-check.yml
name: Architecture Check
on: [push, pull_request]
jobs:
  check-dependencies:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check circular imports
        run: |
          pip install pydeps import-linter
          import-linter --config=.import-linter.toml
```

## Benefits of This Architecture

1. **No Circular Dependencies** - Clear dependency flow
2. **Testable** - Easy to mock dependencies
3. **Maintainable** - Clear separation of concerns
4. **Scalable** - Easy to add new features
5. **Uncle Bob Approved** - Follows Clean Architecture principles

## Migration Strategy

1. **Phase 1:** Create interfaces module (✅ Done)
2. **Phase 2:** Implement dependency injection container
3. **Phase 3:** Refactor services to use interfaces
4. **Phase 4:** Add static analysis tools
5. **Phase 5:** Configure monitoring & prevention
