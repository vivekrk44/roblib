/**
 * @brief Circular buffer implementation which uses a std::vector as the underlying container. It is templated on the type and size of the buffer.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>


template<typename T, std::size_t SIZE>
class CircularBuffer
{
  public:
  
    /**
     * @brief Default constructor for the CircularBuffer class.
     *
     * Initializes the circular buffer with a fixed capacity, setting up the internal
     * buffer and initializing the head, tail, size, and capacity to their default values.
     */
    CircularBuffer() : _buffer(SIZE),
                       _head(0), 
                       _tail(0), 
                       _size(0), 
                       _capacity(SIZE),
                       _full(false) { }
    
    /**
     * @brief Destructor for the CircularBuffer class.
     *
     * This destructor clears the internal buffer and releases any allocated memory
     * by shrinking the buffer's capacity to zero.
     */
    ~CircularBuffer()
    {
      _buffer.clear();
      _buffer.shrink_to_fit();
    }

    
    /**
     * @brief Adds an element to the circular buffer.
     *
     * This function inserts the given element at the current head position of the buffer.
     * It adjusts the head and tail pointers as necessary and updates the size of the buffer.
     * If the buffer is full, the tail pointer is incremented to overwrite the oldest element.
     *
     * @param item The element to add to the buffer.
     * @param overwrite Flag to move the tail if the buffer is full. Default is true.
     * @return bool True if the element was added successfully, false if the buffer is full and overwrite is false.
     */
    bool add(const T& item, bool overwrite = true)
    {
      if (_full && !overwrite) 
      {
        return false;
      }

      // Insert the item at the head of the buffer
      _buffer[_head] = item;
      
      // Update the head position
      _head = (_head + 1) % _capacity;
      
      // If buffer was not full, increment size
      if (!_full)
      {
        _size++;
        // Check if buffer is now full
        if (_size == _capacity)
        {
          _full = true;
        }
      } 
      else 
      {
        // If buffer was full and overwriting, move tail
        _tail = (_tail + 1) % _capacity;
      }
      
      return true;
    }

    /**
     * @brief The get function, this function returns the element at the given index
     * @param index The index of the element to return
     * @return The element at the given index
     */
    T get(std::size_t index) const
    {
      // !< If the index is out of bounds, throw an exception
      if(index >= _size)
      {
        throw std::out_of_range("Index out of range");
      }

      // !< Return the element at the given index
      return _buffer[(_tail + index) % _capacity];
    }

    /**
     * @brief The set function, this function sets the element at the given index to the given value
     * @param index The index of the element to set
     * @param item The value to set the element to
     */
    void set(std::size_t index, const T& item)
    {
      // !< If the index is out of bounds, throw an exception
      if(index >= _size)
      {
        throw std::out_of_range("Index out of range");
      }
      // !< Return the element at the given index
     
      _buffer[(_tail + index) % _capacity] = item;
    }

    /**
     * @brief Removes the last n elements from the buffer by moving the head back n elements
     * @param n The number of elements to remove
     */
    void removeHead(std::size_t n = 1)
    {
      if (n == 0)
      {
        return;
      }
      // !< If the number of elements to remove is greater than the size of the buffer, throw an exception
      if(n > _size)
      {
        throw std::out_of_range("Cannot remove more elements than the size of the buffer");
      }
      
      if (n <= _head) 
      {
        _head -= n;
      } else 
      {
        _head = _capacity - (n - _head);
      }
      
      _size -= n;
      _full = false;
    }

    /**
     * @brief Removes the first n elements from the buffer by moving the tail forward n elements
     * @param n The number of elements to remove
     */
    void removeTail(std::size_t n = 1)
    {
      if (n == 0)
      {
        return;
      }
      // !< If the number of elements to remove is greater than the size of the buffer, throw an exception
      if(n > _size)
      {
        throw std::out_of_range("Cannot remove more elements than the size of the buffer");
      }

      // !< Move the tail forward n elements
      _tail = n ? (_tail + n) % _capacity : 0;
      _size = n ? std::max(_size - n, 0UL) : 0;
      _full = false;
    }

    /**
     * @brief The size function, this function returns the size of the buffer
     * @return The size of the buffer
     */
    std::size_t size() const { return _size; }
    
    /**
     * @brief The capacity function, this function returns the capacity of the buffer
     * @return The capacity of the buffer
     */
    std::size_t capacity() const { return SIZE; }

    /**
     * @brief The empty function, this function returns true if the buffer is empty
     * @return True if the buffer is empty
     */
    bool empty() const { return _size == 0; }

    /**
     * @brief The full function, this function returns true if the buffer is full
     * @return True if the buffer is full
     */
    bool full() const { return _full; }

  private:
    std::vector<T> _buffer; // !< The underlying buffer allocated on the heap
                                 
    std::size_t _head;           // !< The index of the head of the buffer
    std::size_t _tail;           // !< The index of the tail of the buffer
    std::size_t _size;           // !< The number of elements in the buffer
    std::size_t _capacity;       // !< The capacity of the buffer
                                
    bool _full;                  // !< Flag that indicates if the buffer is full

};
