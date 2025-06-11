# 🧹 AGENT 2: Linting & Code Cleanup

## PRIMARY MISSION
Fix 13,543 linting errors and remove all Firebase references

## STRATEGY

### 1. Auto-fix First
```bash
# Run auto-fix
ruff check . --fix

# Format code
black .

# Check what's left
make lint
```

### 2. Priority Fixes
1. **Undefined names** (critical errors)
2. **Import errors** 
3. **Firebase/Firestore comments**
4. **Unused imports**

### 3. Firebase Reference Cleanup
```bash
# Find all Firebase references
grep -r "Firebase\|Firestore\|firebase" . --include="*.py" | grep -v ".git"

# Common replacements:
# "Firebase" → "AWS Cognito" (in auth contexts)
# "Firestore" → "DynamoDB" (in database contexts)
# Remove lines like "# No Firebase dependencies"
```

### 4. Critical Files to Fix
- `src/clarity/auth/dependencies.py`
- `src/clarity/storage/dynamodb_client.py`
- `src/clarity/models/health_data.py`
- All files in `tests/`

## COMMANDS
```bash
# 1. Auto-fix everything possible
ruff check . --fix
black .

# 2. See remaining issues
make lint | head -50

# 3. Focus on source code first
ruff check src/ --fix

# 4. Then tests
ruff check tests/ --fix
```

## SUCCESS CRITERIA
- `make lint` passes with 0 errors
- No Firebase/Firestore references remain
- Code is properly formatted