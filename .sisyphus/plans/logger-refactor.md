# Logger Refactor: Singleton Pattern with TUM Format

## TL;DR

> **Quick Summary**: 重构现有 Logger 为线程安全的多实例单例模式，支持 TUM 格式（带表头），使用 TDD 开发流程。
> 
> **Deliverables**:
> - 重构后的 `src/utils/logger.h` - 单例 Logger 类
> - 单元测试 `src/tests/logger_test.cpp`
> - 更新调用代码 `src/core/vioManager.cpp/.h`
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - sequential dependencies
> **Critical Path**: Test → Logger Implementation → vioManager Update

---

## Context

### Original Request
将现有 log 记录代码改为更科学的方式：
1. 单例模式
2. TUM 格式（带表头）
3. 合并 LoggerTUM 和 LoggerFull 为一个灵活的 Logger

### Interview Summary
**Key Discussions**:
- **Singleton 范围**: 多实例单例（按文件名区分），使用 `std::map<std::string, Logger*>` 管理
- **线程安全**: 需要，使用 `std::mutex` 保护
- **文件流管理**: 保持文件打开，单例持有 `std::ofstream`
- **API 变更**: 改为单例调用，删除 `vioManager.h` 中的 `shared_ptr` 成员
- **错误处理**: 返回 `bool` 状态码
- **测试策略**: TDD - 先写测试

**Research Findings**:
- 当前 `LoggerTUM` 构造时 `use_title = false`，没有表头
- `LogValueTUM::value_name` 有 bug：字符串之间缺少逗号
- 每次调用 `SaveValues()` 都会打开/关闭文件，性能低

### Metis Review
**Identified Gaps** (addressed):
- Singleton 范围问题 → 用户选择：多实例单例（按文件名）
- API 兼容性 → 用户选择：改为单例调用
- 错误处理 → 用户选择：返回状态码
- 测试策略 → 用户选择：TDD

---

## Work Objectives

### Core Objective
重构 Logger 为线程安全的多实例单例，支持 TUM 格式输出（带表头），合并 LoggerTUM 和 LoggerFull。

### Concrete Deliverables
- `src/utils/logger.h` - 新的 Singleton Logger 类
- `src/tests/logger_test.cpp` - 单元测试
- `src/core/vioManager.cpp` - 更新调用方式
- `src/core/vioManager.h` - 移除 shared_ptr 成员变量

### Definition of Done
- [x] 单元测试通过
- [x] 编译无错误
- [x] TUM 格式输出正确（带表头）
- [x] 线程安全验证

### Must Have
- 多实例单例（按文件名区分）
- 线程安全（mutex 保护）
- TUM 格式表头（`# timestamp x y z qw qx qy qz`）
- 文件流保持打开
- 返回 `bool` 状态码

### Must NOT Have (Guardrails)
- 不要改变输出数据格式（只添加表头）
- 不要添加 log rotation、log levels、async logging
- 不要引入外部日志库依赖
- 不要修改 test 文件夹中的测试代码

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO（需新建）
- **Automated tests**: TDD
- **Framework**: gtest（项目已有依赖）

### QA Policy
每个任务包含 agent-executed QA scenarios。

| Deliverable Type | Verification Tool | Method |
|------------------|-------------------|--------|
| Logger 类 | Bash (gtest) | 运行单元测试 |
| 编译 | Bash (catkin) | catkin build vio_msckf |
| 输出格式 | Bash (diff) | 对比输出文件 |

---

## Execution Strategy

### Sequential Execution (有依赖关系)

```
Step 1: 编写单元测试 (TDD RED phase)
    ↓
Step 2: 实现 Logger 单例 (TDD GREEN phase)
    ↓
Step 3: 更新 vioManager 调用代码
    ↓
Step 4: 集成测试与验证
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| 1. 编写测试 | — | 2 |
| 2. 实现Logger | 1 | 3 |
| 3. 更新调用 | 2 | 4 |
| 4. 集成验证 | 3 | — |

---

## TODOs

- [x] 1. **编写单元测试** (TDD RED Phase)

  **What to do**:
  - 创建 `src/tests/logger_test.cpp`
  - 测试用例：
    1. `GetInstance_SameFilename_ReturnsSameInstance` - 同一文件名返回同一实例
    2. `GetInstance_DifferentFilenames_ReturnsDifferentInstances` - 不同文件名返回不同实例
    3. `LogValues_WritesToCorrectFile` - 正确写入文件
    4. `LogValues_TUMFormat_HasHeader` - TUM 格式有表头
    5. `LogValues_ThreadSafe_ConcurrentWrites` - 多线程并发写入无数据损坏
    6. `LogValues_Failure_ReturnsFalse` - 无效路径返回 false
  - 测试应失败（RED phase）

  **Must NOT do**:
  - 不要实现 Logger 类（仅写测试）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 测试代码相对简单，遵循 TDD 模式
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Task 2

  **References**:
  - `src/utils/logger.h:1-249` - 当前 Logger 实现，理解现有接口
  - `src/core/vioManager.cpp:680-738` - 当前 Logger 使用方式

  **QA Scenarios**:
  ```
  Scenario: Tests fail because Logger not implemented
    Tool: Bash
    Steps:
      1. cd /home/gao/ws/catkin_ws && catkin build vio_msckf
      2. ./devel/lib/vio_msckf/logger_test (or appropriate test binary)
    Expected Result: Tests FAIL (RED phase)
    Evidence: .sisyphus/evidence/task-1-test-fail.log
  ```

  **Commit**: YES
  - Message: `test(logger): add unit tests for singleton logger`
  - Files: `src/tests/logger_test.cpp`, `src/tests/CMakeLists.txt` (if needed)

- [x] 2. **实现 Logger 单例类** (TDD GREEN Phase)

  **What to do**:
  - 重构 `src/utils/logger.h`：
    1. 创建 `Logger` 单例类（多实例，按文件名区分）
    2. 使用 `std::map<std::string, std::unique_ptr<Logger>>` 管理实例
    3. 使用 `std::mutex` 保护 map 访问
    4. 每个实例持有 `std::ofstream`，保持打开
    5. TUM 格式：第一行写入表头 `# timestamp px py pz qx qy qz qw`
    6. `bool SaveValues(const LogValueTUM& value)` - 返回状态码
    7. 析构时关闭文件
  - 删除旧的 `LoggerTUM`, `LoggerFull`, `LoggerBase` 类（或标记 deprecated）
  - 修复 `LogValueTUM::value_name` 的 bug（缺少逗号）

  **Must NOT do**:
  - 不要添加 log rotation、log levels 功能
  - 不要改变输出数据格式（只添加表头）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要理解单例模式、线程安全、文件 I/O
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Depends On**: Task 1
  - **Blocks**: Task 3

  **References**:
  - `src/utils/logger.h:1-249` - 当前实现
  - `src/core/vioManager.cpp:680-738` - 当前使用方式，理解数据流
  - TUM format: `# timestamp tx ty tz qx qy qz qw`

  **QA Scenarios**:
  ```
  Scenario: Tests pass after implementation
    Tool: Bash
    Steps:
      1. cd /home/gao/ws/catkin_ws && catkin build vio_msckf
      2. Run gtest
    Expected Result: All tests PASS (GREEN phase)
    Evidence: .sisyphus/evidence/task-2-test-pass.log

  Scenario: TUM format has header
    Tool: Bash
    Steps:
      1. Create a test log file
      2. head -n 1 test_output.csv
    Expected Result: First line starts with '#'
    Evidence: .sisyphus/evidence/task-2-tum-header.txt
  ```

  **Commit**: YES
  - Message: `refactor(logger): implement singleton pattern with TUM format`
  - Files: `src/utils/logger.h`

- [x] 3. **更新 vioManager 调用代码**

  **What to do**:
  - 修改 `src/core/vioManager.h`:
    - 删除 `std::shared_ptr<utils::LoggerFull> vio_logger;`
    - 删除 `std::shared_ptr<utils::LoggerTUM> vio_logger_tum;`
  - 修改 `src/core/vioManager.cpp`:
    - 删除 `vio_logger = std::make_shared<...>` 调用
    - 删除 `vio_logger_tum = std::make_shared<...>` 调用
    - 改用 `Logger::GetInstance(filename)->SaveValues(value)` 方式
    - 处理返回的状态码（可选：LOG(WARNING) 如果失败）

  **Must NOT do**:
  - 不要改变日志内容（只改变调用方式）
  - 不要修改其他文件的 Logger 调用

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的 API 调用修改
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Depends On**: Task 2
  - **Blocks**: Task 4

  **References**:
  - `src/core/vioManager.h:95-96` - 当前 Logger 成员变量
  - `src/core/vioManager.cpp:68-73` - 当前 Logger 初始化
  - `src/core/vioManager.cpp:680-738` - 当前 SaveValues 调用

  **QA Scenarios**:
  ```
  Scenario: Build succeeds after API change
    Tool: Bash
    Steps:
      1. cd /home/gao/ws/catkin_ws && catkin build vio_msckf
    Expected Result: Build PASS, no errors
    Evidence: .sisyphus/evidence/task-3-build.log
  ```

  **Commit**: YES
  - Message: `refactor(vioManager): update logger usage to singleton`
  - Files: `src/core/vioManager.cpp`, `src/core/vioManager.h`

- [x] 4. **集成测试与验证**

  **What to do**:
  - 运行完整测试：
    1. 编译整个项目
    2. 运行单元测试
    3. 使用测试数据运行 VIO，验证输出格式
  - 验证 TUM 格式：
    - 表头存在
    - 数据格式正确
  - 验证线程安全：
    - 多线程写入不会产生数据损坏

  **Must NOT do**:
  - 不要修改已完成的代码

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 集成测试需要全面验证
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Depends On**: Task 3

  **References**:
  - `src/utils/logger.h` - 新 Logger 实现
  - `src/tests/logger_test.cpp` - 单元测试

  **QA Scenarios**:
  ```
  Scenario: Full integration test
    Tool: Bash
    Steps:
      1. cd /home/gao/ws/catkin_ws && catkin build vio_msckf
      2. Run gtest
      3. Run VIO with test bag
      4. Check output log file format
    Expected Result: All tests pass, log format correct
    Evidence: .sisyphus/evidence/task-4-integration.log
  ```

  **Commit**: NO (verification task)

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — `oracle`
  验证所有 "Must Have" 已实现，所有 "Must NOT Have" 未出现。

- [x] F2. **Code Quality Review** — `unspecified-high`
  运行编译检查、代码审查。

- [x] F3. **Integration Test** — `unspecified-high`
  验证 TUM 格式输出正确。

- [x] F4. **Thread Safety Test** — `deep`
  验证多线程写入不会产生数据竞争。

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `test(logger): add unit tests for singleton logger` | src/tests/logger_test.cpp | gtest pass |
| 2 | `refactor(logger): implement singleton pattern with TUM format` | src/utils/logger.h | gtest pass |
| 3 | `refactor(vioManager): update logger usage to singleton` | src/core/vioManager.cpp, src/core/vioManager.h | catkin build |

---

## Success Criteria

### Verification Commands
```bash
# 编译
cd /home/gao/ws/catkin_ws && catkin build vio_msckf

# 单元测试
./build/vio_msckf/logger_test

# 输出格式验证
head -n 2 output_log.csv
# Expected:
# # timestamp px py pz qx qy qz qw
# 1234567890.0 1.0 2.0 3.0 0.0 0.0 0.0 1.0
```

### Final Checklist
- [x] 所有 "Must Have" 已实现
- [x] 所有 "Must NOT Have" 未出现
- [x] 单元测试通过
- [x] TUM 格式正确（带表头）
- [x] 线程安全验证通过
