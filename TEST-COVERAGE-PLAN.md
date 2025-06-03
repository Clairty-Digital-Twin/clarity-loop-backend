# 📊 Test Coverage Improvement Plan

## 🎯 **Target: 51% → 85% Coverage**

### **Priority 1: Critical Low-Coverage Files**

#### **1. `src/clarity/storage/firestore_client.py` (11% → 85%)**
**Current**: 448 lines, only 11% covered
**Missing Tests**:
- [ ] Connection establishment and failure scenarios
- [ ] Document CRUD operations with validation
- [ ] Query operations with filters and pagination
- [ ] Transaction handling and rollback scenarios
- [ ] Error handling for network failures
- [ ] Batch operations testing
- [ ] Collection management operations

**Estimated Time**: 3-4 hours

#### **2. `src/clarity/services/auth_service.py` (16% → 85%)**
**Current**: 188 lines, only 16% covered
**Missing Tests**:
- [ ] Token generation and validation flows
- [ ] User registration with various scenarios
- [ ] Login with different credential types
- [ ] Token refresh mechanisms
- [ ] MFA integration testing
- [ ] Password reset workflows
- [ ] Session management
- [ ] Error handling for auth failures

**Estimated Time**: 2-3 hours

#### **3. `src/clarity/api/v1/health_data.py` (38% → 85%)**
**Current**: 117 lines, only 38% covered
**Missing Tests**:
- [ ] Authentication failure scenarios
- [ ] Request validation errors
- [ ] Business rule violation handling
- [ ] Large payload handling
- [ ] Concurrent request scenarios
- [ ] Rate limiting behavior
- [ ] Error response formatting

**Estimated Time**: 1-2 hours

### **Priority 2: Medium Coverage Files**

#### **4. `src/clarity/services/health_data_service.py` (59% → 85%)**
**Current**: 100 lines, 59% covered
**Missing Tests**:
- [ ] Edge cases in data validation
- [ ] Repository error handling
- [ ] Business rule enforcement
- [ ] Data transformation scenarios

**Estimated Time**: 1 hour

#### **5. `src/clarity/api/v1/auth.py` (55% → 85%)**
**Current**: 137 lines, 55% covered
**Missing Tests**:
- [ ] Registration edge cases
- [ ] Login failure scenarios
- [ ] Token refresh edge cases
- [ ] Logout cleanup verification

**Estimated Time**: 1 hour

### **Priority 3: Unused/Untested Code**

#### **6. `src/clarity/core/decorators.py` (0% → 85%)**
**Current**: 197 lines, 0% covered
**Action**: Either implement tests or remove if unused

#### **7. `src/clarity/ml/model_integrity.py` (0% → 85%)**
**Current**: 146 lines, 0% covered
**Action**: Either implement tests or remove if unused

## 🧪 **Test Implementation Strategy**

### **Integration Tests Needed**
```python
# Example: Firestore integration tests
@pytest.mark.integration
async def test_firestore_health_data_crud():
    """Test complete CRUD operations with real Firestore emulator."""
    # Setup emulator, test create/read/update/delete
    pass

@pytest.mark.integration  
async def test_auth_service_complete_flow():
    """Test complete authentication flow."""
    # Test registration → login → token refresh → logout
    pass
```

### **Error Scenario Tests**
```python
# Example: Error handling tests
async def test_health_data_upload_network_failure():
    """Test handling of network failures during upload."""
    # Mock network failure, verify graceful handling
    pass

async def test_auth_service_invalid_credentials():
    """Test various invalid credential scenarios."""
    # Test malformed tokens, expired tokens, etc.
    pass
```

### **Edge Case Tests**
```python
# Example: Edge case tests
async def test_large_health_data_payload():
    """Test handling of large health data uploads."""
    # Test with maximum allowed payload size
    pass

async def test_concurrent_auth_requests():
    """Test concurrent authentication requests."""
    # Test race conditions and thread safety
    pass
```

## 📈 **Coverage Boost Timeline**

| File | Current | Target | Time | Priority |
|------|---------|--------|------|----------|
| firestore_client.py | 11% | 85% | 3-4h | High |
| auth_service.py | 16% | 85% | 2-3h | High |
| health_data.py | 38% | 85% | 1-2h | High |
| health_data_service.py | 59% | 85% | 1h | Medium |
| auth.py | 55% | 85% | 1h | Medium |

**Total Estimated Time**: 8-11 hours

## 🚀 **Quick Wins for Coverage**

1. **Add Firestore emulator tests** - Biggest coverage boost
2. **Test error scenarios** - Easy to add, high impact
3. **Test edge cases** - Fill in missing branches
4. **Add integration tests** - Test real workflows

## ✅ **Success Metrics**

- [ ] Overall coverage: 51% → 85%
- [ ] All critical files above 80% coverage
- [ ] No files below 70% coverage (except intentionally excluded)
- [ ] All tests passing with new coverage 