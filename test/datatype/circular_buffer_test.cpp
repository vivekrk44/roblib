#include <gtest/gtest.h>
#include <string>
#include "roblib/datatype/circular_buffer.hpp" 

// Test fixture for CircularBuffer tests
class CircularBufferTest : public ::testing::Test {
protected:
    // Set up function that runs before each test
    void SetUp() override {
        // Initialize buffers with different types for testing
        // No explicit setup needed as we'll create new buffers in each test
    }

    // Tear down function that runs after each test
    void TearDown() override {
        // No explicit teardown needed
    }
};

// Test default constructor and initial state
TEST_F(CircularBufferTest, InitialState) {
    CircularBuffer<int, 5> buffer;
    
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.capacity(), 5);
    EXPECT_TRUE(buffer.empty());
    EXPECT_FALSE(buffer.full());
}

// Test adding elements to the buffer
TEST_F(CircularBufferTest, AddElements) {
    CircularBuffer<int, 3> buffer;
    
    EXPECT_TRUE(buffer.add(10));
    EXPECT_EQ(buffer.size(), 1);
    EXPECT_FALSE(buffer.empty());
    
    EXPECT_TRUE(buffer.add(20));
    EXPECT_EQ(buffer.size(), 2);
    
    EXPECT_TRUE(buffer.add(30));
    EXPECT_EQ(buffer.size(), 3);
    EXPECT_TRUE(buffer.full());
    
    // Buffer should be full now
    EXPECT_TRUE(buffer.add(40)); // Should overwrite the first element
    EXPECT_EQ(buffer.size(), 3); // Size remains the same
}

// Test getting elements from the buffer
TEST_F(CircularBufferTest, GetElements) {
    CircularBuffer<int, 3> buffer;
    
    buffer.add(10);
    buffer.add(20);
    buffer.add(30);
    
    EXPECT_EQ(buffer.get(0), 10);
    EXPECT_EQ(buffer.get(1), 20);
    EXPECT_EQ(buffer.get(2), 30);
    
    // Test out of range
    EXPECT_THROW(buffer.get(3), std::out_of_range);
}

// Test setting elements in the buffer
TEST_F(CircularBufferTest, SetElements) {
    CircularBuffer<int, 3> buffer;
    
    buffer.add(10);
    buffer.add(20);
    buffer.add(30);
    
    buffer.set(1, 25);
    EXPECT_EQ(buffer.get(1), 25);
    
    // Test out of range
    EXPECT_THROW(buffer.set(3, 40), std::out_of_range);
}

// Test circular behavior
TEST_F(CircularBufferTest, CircularBehavior) {
    CircularBuffer<int, 3> buffer;
    
    buffer.add(10);
    buffer.add(20);
    buffer.add(30);
    
    // This should overwrite the oldest element (10)
    buffer.add(40);
    
    EXPECT_EQ(buffer.get(0), 20);
    EXPECT_EQ(buffer.get(1), 30);
    EXPECT_EQ(buffer.get(2), 40);
    
    // Add another element to overwrite the next oldest
    buffer.add(50);
    
    EXPECT_EQ(buffer.get(0), 30);
    EXPECT_EQ(buffer.get(1), 40);
    EXPECT_EQ(buffer.get(2), 50);
}

// Test removing elements from the head
TEST_F(CircularBufferTest, RemoveHead) {
    CircularBuffer<int, 5> buffer;
    
    buffer.add(10);
    buffer.add(20);
    buffer.add(30);
    buffer.add(40);
    
    buffer.removeHead(1);
    EXPECT_EQ(buffer.size(), 3);
    EXPECT_EQ(buffer.get(0), 10);
    EXPECT_EQ(buffer.get(1), 20);
    EXPECT_EQ(buffer.get(2), 30);
    
    buffer.removeHead(2);
    EXPECT_EQ(buffer.size(), 1);
    EXPECT_EQ(buffer.get(0), 10);
    
    // Test removing too many elements
    EXPECT_THROW(buffer.removeHead(2), std::out_of_range);
    
    // Test removing zero elements
    buffer.removeHead(0);
    EXPECT_EQ(buffer.size(), 1);
}

// Test removing elements from the tail
TEST_F(CircularBufferTest, RemoveTail) {
    CircularBuffer<int, 5> buffer;
    
    buffer.add(10);
    buffer.add(20);
    buffer.add(30);
    buffer.add(40);
    
    buffer.removeTail(1);
    EXPECT_EQ(buffer.size(), 3);
    EXPECT_EQ(buffer.get(0), 20);
    EXPECT_EQ(buffer.get(1), 30);
    EXPECT_EQ(buffer.get(2), 40);
    
    buffer.removeTail(2);
    EXPECT_EQ(buffer.size(), 1);
    EXPECT_EQ(buffer.get(0), 40);
    
    // Test removing too many elements
    EXPECT_THROW(buffer.removeTail(2), std::out_of_range);
    
    // Test removing zero elements
    buffer.removeTail(0);
    EXPECT_EQ(buffer.size(), 1);
}

// Test buffer with different data types
TEST_F(CircularBufferTest, DifferentTypes) {
    CircularBuffer<std::string, 3> strBuffer;
    
    strBuffer.add("Hello");
    strBuffer.add("World");
    
    EXPECT_EQ(strBuffer.get(0), "Hello");
    EXPECT_EQ(strBuffer.get(1), "World");
    
    CircularBuffer<double, 2> doubleBuffer;
    doubleBuffer.add(3.14);
    doubleBuffer.add(2.71);
    
    EXPECT_DOUBLE_EQ(doubleBuffer.get(0), 3.14);
    EXPECT_DOUBLE_EQ(doubleBuffer.get(1), 2.71);
}

// Test overwrite flag
TEST_F(CircularBufferTest, OverwriteFlag) {
    CircularBuffer<int, 3> buffer;
    
    buffer.add(10);
    buffer.add(20);
    buffer.add(30);
    
    // Buffer is full, try to add with overwrite=false
    EXPECT_FALSE(buffer.add(40, false));
    
    // The buffer should remain unchanged
    EXPECT_EQ(buffer.get(0), 10);
    EXPECT_EQ(buffer.get(1), 20);
    EXPECT_EQ(buffer.get(2), 30);
    
    // Now try with overwrite=true
    EXPECT_TRUE(buffer.add(40, true));
    
    // The oldest element should be overwritten
    EXPECT_EQ(buffer.get(0), 20);
    EXPECT_EQ(buffer.get(1), 30);
    EXPECT_EQ(buffer.get(2), 40);
}

// Test complex circular scenarios
TEST_F(CircularBufferTest, ComplexCircularScenarios) {
    CircularBuffer<int, 5> buffer;
    
    // Fill the buffer
    for (int i = 0; i < 5; i++) {
        buffer.add(i);
    }
    
    // Remove from tail and head
    buffer.removeTail(2);
    buffer.removeHead(1);
    
    // Add new elements
    buffer.add(100);
    buffer.add(200);
    
    // Check the state
    EXPECT_EQ(buffer.size(), 4);
    EXPECT_EQ(buffer.get(0), 2);
    EXPECT_EQ(buffer.get(1), 3);
    EXPECT_EQ(buffer.get(2), 100);
    EXPECT_EQ(buffer.get(3), 200);
    
    // Fill it again to test wrapping
    buffer.add(300);
    buffer.add(400); // This should start overwriting
    
    EXPECT_EQ(buffer.size(), 5);
    EXPECT_EQ(buffer.get(0), 3);
    EXPECT_EQ(buffer.get(1), 100);
    EXPECT_EQ(buffer.get(2), 200);
    EXPECT_EQ(buffer.get(3), 300);
    EXPECT_EQ(buffer.get(4), 400);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
