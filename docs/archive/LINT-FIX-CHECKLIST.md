# 🔧 Lint Error Fix Checklist

## 📊 **Error Summary (76 total)**

### **Type Annotation Errors (ANN001, ANN204) - 15 errors**

- [ ] `src/clarity/ml/inference_engine.py` - Add return types for `__aenter__`, `__aexit__`
- [ ] `tests/ml/test_gemini_service.py` - Add type annotations for test method parameters
- [ ] `tests/ml/test_inference_engine.py` - Add type annotations for test method parameters

### **Unused Import/Variable Errors (F401, F841) - 4 errors**

- [ ] `src/clarity/ml/inference_engine.py:22` - Remove unused `Optional` import
- [ ] `tests/ml/test_gemini_service.py:126` - Remove unused `mock_init` variable
- [ ] `tests/ml/test_gemini_service.py:177` - Remove unused `mock_init` variable

### **Code Style Errors (PLR6301) - 45 errors**

- [ ] Convert test methods to static methods where appropriate
- [ ] Most test methods can be converted to `@staticmethod`

### **Logging Errors (G004) - 4 errors**

- [ ] `src/clarity/ml/inference_engine.py` - Replace f-strings in logging with % formatting

### **Exception Handling Errors (BLE001, TRY401, B017, PT011) - 5 errors**

- [ ] Replace blind `Exception` catches with specific exceptions
- [ ] Remove redundant exception objects in logging calls
- [ ] Make pytest.raises more specific

### **Other Style Issues (SIM117, RUF029, ARG005) - 3 errors**

- [ ] Combine nested `with` statements
- [ ] Remove unnecessary `async` from functions
- [ ] Remove unused lambda arguments

## 🚀 **Quick Fix Commands**

```bash
# Auto-fix what's possible
ruff check --fix .

# Manual fixes needed for:
# 1. Type annotations in test methods
# 2. Converting test methods to static methods
# 3. Specific exception handling improvements
```

## ⏱️ **Estimated Time: 2-3 hours**

Most errors are straightforward style fixes that can be batch-processed.
