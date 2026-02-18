# Learnings - Logger Refactor

## Current Logger Implementation

### File Structure
- `src/utils/logger.h` - Main logger implementation (249 lines)
- `src/core/vioManager.h:95-96` - Logger member variables
- `src/core/vioManager.cpp:66-74` - Logger initialization
- `src/core/vioManager.cpp:678-740` - SaveResultsToFile() usage

### Current Classes
1. `LogValueTUM` - TUM format data structure
   - **BUG**: Line 20 has missing commas in `value_name`
   - `std::vector<std::string> value_name{"#timestamp" "px" "py" ...}` - strings concatenate!
   
2. `LogValueFull` - Full format data structure (more fields)

3. `LoggerBase<Derived>` - CRTP base class
   - Opens file in constructor
   - `InitLogFile()` creates file with optional header
   - `filename_` stored as member

4. `LoggerTUM` - TUM format logger
   - Constructor: `use_title = false` (NO HEADER!)
   - `SaveValues()` opens/closes file every call (performance issue)

5. `LoggerFull` - Full format logger
   - Constructor: `use_title = true` (has header)

### Current Problems
1. **Performance**: Opens/closes file on every `SaveValues()` call
2. **No singleton**: Creates new instances via `std::make_shared`
3. **TUM format missing header**: `use_title = false`
4. **Bug in LogValueTUM::value_name**: Missing commas between strings

### Test Directory
- Existing tests in `src/test/` (not `src/tests/`)
- CMakeLists.txt pattern for test executables

## Refactor Requirements

### Must Have
- Multi-instance singleton (by filename)
- Thread safety (mutex)
- TUM format with header
- Keep file stream open
- Return bool status code

### Must NOT Have
- Don't change output data format (only add header)
- Don't add log rotation/levels/async
- Don't modify test files in src/test/

## TUM Format
```
# timestamp px py pz qx qy qz qw
1234567890.0 1.0 2.0 3.0 0.0 0.0 0.0 1.0
```

## Test File Created (TDD RED Phase)

### Test File
- `src/test/test_logger.cpp` - Unit tests for Logger singleton

### Test Cases Defined
1. `GetInstance_SameFilename_ReturnsSameInstance` - Singleton behavior
2. `GetInstance_DifferentFilenames_ReturnsDifferentInstances` - Multi-instance by filename
3. `LogValues_WritesToCorrectFile` - Correct file output
4. `LogValues_TUMFormat_HasHeader` - TUM format header verification
5. `LogValues_ThreadSafe_ConcurrentWrites` - Thread safety with 4 threads × 100 writes
6. `LogValues_Failure_ReturnsFalse` - Invalid path handling
7. `SaveValues_AfterClose_ReturnsFalse` - Post-close behavior

### Expected Logger Interface (defined in test file)
```cpp
namespace utils {
struct LogValueTUM {
    double timestamp, px, py, pz, qx, qy, qz, qw;
};

class Logger {
public:
    static Logger* GetInstance(const std::string& filename);
    bool SaveValues(const LogValueTUM& value);
    void Close();
private:
    Logger(const std::string& filename);
    ~Logger();
};
}
```

### Build Status
- Test file compiles successfully
- Link fails (expected) - Logger singleton not implemented yet
- CMakeLists.txt updated with test target

### CMake Pattern Used
```cmake
catkin_add_gtest(test_logger test_logger.cpp)
if (TARGET test_logger)
    target_include_directories(test_logger PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
    target_link_libraries(test_logger ${catkin_LIBRARIES} ${thirdparty_libraries} ${GTEST_LIBRARIES} pthread)
endif()
```

## Implementation Completed (TDD GREEN Phase)

### Files Modified/Created
1. `src/utils/logger.h` - Refactored with new Logger singleton class
   - `LogValueTUM` changed from class to simple struct (removed buggy `value_name`)
   - `LoggerTUM` changed from CRTP base class to standalone class (for backward compatibility)
   - Added new `Logger` singleton class with declarations only

2. `src/utils/logger.cpp` - NEW FILE with Logger implementation
   - Static member definitions for `instances_` and `instances_mutex_`
   - Constructor, destructor, GetInstance, SaveValues, Close implementations

3. `src/test/CMakeLists.txt` - Updated to include logger.cpp in test build
   - Added `${CMAKE_CURRENT_SOURCE_DIR}/../utils/logger.cpp` to test sources

### Implementation Details

#### Logger Singleton Pattern
- Multi-instance singleton using `std::map<std::string, std::unique_ptr<Logger>>`
- Thread-safe map access with `std::mutex instances_mutex_`
- Thread-safe file writes with `std::mutex file_mutex_`
- Public destructor (required for `unique_ptr`)
- Private constructor (enforces singleton pattern)

#### Key Design Decisions
1. **Header + CPP split**: Test file has its own forward declarations, so implementation must be in CPP file for linker to find it
2. **Fixed format**: Using `std::fixed` format (not scientific) so numbers like `1.0` appear as `1.0000000000` (matches test expectations)
3. **Header on first write**: TUM header `# timestamp px py pz qx qy qz qw` written on first `SaveValues()` call
4. **File stays open**: `std::ofstream file_` is a member, opened in constructor, closed in destructor/Close()
5. **Flush after write**: `file_.flush()` called after each write for data durability

#### Test Results
All 7 tests pass:
- `GetInstance_SameFilename_ReturnsSameInstance` ✓
- `GetInstance_DifferentFilenames_ReturnsDifferentInstances` ✓
- `LogValues_WritesToCorrectFile` ✓
- `LogValues_TUMFormat_HasHeader` ✓
- `LogValues_ThreadSafe_ConcurrentWrites` ✓
- `LogValues_Failure_ReturnsFalse` ✓
- `SaveValues_AfterClose_ReturnsFalse` ✓

### Gotchas Encountered
1. **Private destructor + unique_ptr**: `unique_ptr` requires public destructor for deletion
2. **Test file forward declarations**: Test defines its own Logger interface, implementation must be in separate TU
3. **Number formatting**: Tests expect `1.0` format, need `std::fixed` (not scientific or default)
4. **LSP false positives**: Language server shows errors that don't appear in actual build

### CMake Pattern for Test with External Source
```cmake
catkin_add_gtest(test_logger test_logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/logger.cpp
)
if (TARGET test_logger)
    target_include_directories(test_logger PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
    target_link_libraries(test_logger ${catkin_LIBRARIES} ${thirdparty_libraries} ${GTEST_LIBRARIES} pthread)
endif()
```
