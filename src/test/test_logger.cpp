/**
 * @file test_logger.cpp
 * @brief Unit tests for Logger singleton class (TDD RED Phase)
 *
 * These tests define the expected interface for the new Logger singleton.
 * Tests will FAIL until Logger singleton is implemented.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Forward declaration of expected new Logger interface
// This will fail to compile until Logger singleton is implemented
namespace utils
{

/**
 * @brief TUM format data structure for logging
 */
struct LogValueTUM
{
    double timestamp = 0;
    double px = 0;
    double py = 0;
    double pz = 0;
    double qx = 0;
    double qy = 0;
    double qz = 0;
    double qw = 1.0;
};

/**
 * @brief Logger singleton class for thread-safe file logging
 *
 * Expected interface (not implemented yet):
 * - Multi-instance singleton by filename
 * - Thread-safe writes
 * - TUM format with header
 * - Returns bool status code
 */
class Logger
{
public:
    /**
     * @brief Get singleton instance for a specific filename
     * @param filename Full path to log file
     * @return Pointer to Logger instance, nullptr on failure
     */
    static Logger* GetInstance(const std::string& filename);

    /**
     * @brief Save a TUM format value to the log file
     * @param value The value to log
     * @return true on success, false on failure
     */
    bool SaveValues(const LogValueTUM& value);

    /**
     * @brief Close the log file and cleanup
     */
    void Close();

private:
    Logger(const std::string& filename);
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};

}  // namespace utils

// =============================================================================
// Test Fixtures
// =============================================================================

class LoggerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create unique temp filenames for each test
        test_dir_ = "/tmp/logger_test_" + std::to_string(std::rand());
        test_file1_ = test_dir_ + "/test1.tum";
        test_file2_ = test_dir_ + "/test2.tum";

        // Create test directory
        system(("mkdir -p " + test_dir_).c_str());
    }

    void TearDown() override
    {
        // Cleanup test files
        system(("rm -rf " + test_dir_).c_str());
    }

    std::string test_dir_;
    std::string test_file1_;
    std::string test_file2_;
};

// =============================================================================
// Test Cases
// =============================================================================

/**
 * @brief Test: Same filename returns same instance (singleton behavior)
 */
TEST_F(LoggerTest, GetInstance_SameFilename_ReturnsSameInstance)
{
    // Arrange & Act
    utils::Logger* instance1 = utils::Logger::GetInstance(test_file1_);
    utils::Logger* instance2 = utils::Logger::GetInstance(test_file1_);

    // Assert
    EXPECT_NE(instance1, nullptr);
    EXPECT_NE(instance2, nullptr);
    EXPECT_EQ(instance1, instance2) << "Same filename should return same instance";

    // Cleanup
    if (instance1)
    {
        instance1->Close();
    }
}

/**
 * @brief Test: Different filenames return different instances
 */
TEST_F(LoggerTest, GetInstance_DifferentFilenames_ReturnsDifferentInstances)
{
    // Arrange & Act
    utils::Logger* instance1 = utils::Logger::GetInstance(test_file1_);
    utils::Logger* instance2 = utils::Logger::GetInstance(test_file2_);

    // Assert
    EXPECT_NE(instance1, nullptr);
    EXPECT_NE(instance2, nullptr);
    EXPECT_NE(instance1, instance2) << "Different filenames should return different instances";

    // Cleanup
    if (instance1)
    {
        instance1->Close();
    }
    if (instance2)
    {
        instance2->Close();
    }
}

/**
 * @brief Test: LogValues writes correct data to file
 */
TEST_F(LoggerTest, LogValues_WritesToCorrectFile)
{
    // Arrange
    utils::Logger* logger = utils::Logger::GetInstance(test_file1_);
    ASSERT_NE(logger, nullptr);

    utils::LogValueTUM value;
    value.timestamp = 1234567890.0;
    value.px = 1.0;
    value.py = 2.0;
    value.pz = 3.0;
    value.qx = 0.0;
    value.qy = 0.0;
    value.qz = 0.0;
    value.qw = 1.0;

    // Act
    bool result = logger->SaveValues(value);
    logger->Close();

    // Assert
    EXPECT_TRUE(result) << "SaveValues should return true on success";

    // Verify file contents
    std::ifstream file(test_file1_);
    ASSERT_TRUE(file.is_open()) << "Should be able to open log file";

    std::string line;
    std::getline(file, line);  // Skip header line
    std::getline(file, line);  // Read data line

    // Check that data line contains expected values
    EXPECT_NE(line.find("1234567890"), std::string::npos);
    EXPECT_NE(line.find("1.0"), std::string::npos);
    EXPECT_NE(line.find("2.0"), std::string::npos);
    EXPECT_NE(line.find("3.0"), std::string::npos);
}

/**
 * @brief Test: TUM format has correct header
 */
TEST_F(LoggerTest, LogValues_TUMFormat_HasHeader)
{
    // Arrange
    utils::Logger* logger = utils::Logger::GetInstance(test_file1_);
    ASSERT_NE(logger, nullptr);

    utils::LogValueTUM value;
    value.timestamp = 100.0;

    // Act
    logger->SaveValues(value);
    logger->Close();

    // Assert - Check header format
    std::ifstream file(test_file1_);
    ASSERT_TRUE(file.is_open());

    std::string header;
    std::getline(file, header);

    // TUM format header should be: "# timestamp px py pz qx qy qz qw"
    EXPECT_EQ(header[0], '#') << "TUM header should start with #";
    EXPECT_NE(header.find("timestamp"), std::string::npos);
    EXPECT_NE(header.find("px"), std::string::npos);
    EXPECT_NE(header.find("py"), std::string::npos);
    EXPECT_NE(header.find("pz"), std::string::npos);
    EXPECT_NE(header.find("qx"), std::string::npos);
    EXPECT_NE(header.find("qy"), std::string::npos);
    EXPECT_NE(header.find("qz"), std::string::npos);
    EXPECT_NE(header.find("qw"), std::string::npos);
}

/**
 * @brief Test: Thread safety - concurrent writes should not corrupt data
 */
TEST_F(LoggerTest, LogValues_ThreadSafe_ConcurrentWrites)
{
    // Arrange
    utils::Logger* logger = utils::Logger::GetInstance(test_file1_);
    ASSERT_NE(logger, nullptr);

    const int num_threads = 4;
    const int writes_per_thread = 100;
    std::vector<std::thread> threads;

    // Act - Multiple threads writing concurrently
    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < writes_per_thread; ++i)
            {
                utils::LogValueTUM value;
                value.timestamp = static_cast<double>(t * writes_per_thread + i);
                value.px = static_cast<double>(t);
                value.py = static_cast<double>(i);
                EXPECT_TRUE(logger->SaveValues(value));
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads)
    {
        thread.join();
    }

    logger->Close();

    // Assert - Count lines in file (should be header + num_threads * writes_per_thread)
    std::ifstream file(test_file1_);
    ASSERT_TRUE(file.is_open());

    int line_count = 0;
    std::string line;
    while (std::getline(file, line))
    {
        line_count++;
    }

    // 1 header + (num_threads * writes_per_thread) data lines
    int expected_lines = 1 + (num_threads * writes_per_thread);
    EXPECT_EQ(line_count, expected_lines) << "All writes should be recorded without data loss";
}

/**
 * @brief Test: Invalid path returns false
 */
TEST_F(LoggerTest, LogValues_Failure_ReturnsFalse)
{
    // Arrange - Use invalid path (root directory, no permission)
    std::string invalid_path = "/nonexistent_directory_12345/test.tum";

    // Act
    utils::Logger* logger = utils::Logger::GetInstance(invalid_path);

    // Assert - Should return nullptr for invalid path
    EXPECT_EQ(logger, nullptr) << "GetInstance should return nullptr for invalid path";
}

/**
 * @brief Test: SaveValues returns false after Close()
 */
TEST_F(LoggerTest, SaveValues_AfterClose_ReturnsFalse)
{
    // Arrange
    utils::Logger* logger = utils::Logger::GetInstance(test_file1_);
    ASSERT_NE(logger, nullptr);

    utils::LogValueTUM value;
    value.timestamp = 1.0;

    // Act
    logger->Close();
    bool result = logger->SaveValues(value);

    // Assert
    EXPECT_FALSE(result) << "SaveValues should return false after Close()";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    std::srand(std::time(nullptr));
    return RUN_ALL_TESTS();
}
