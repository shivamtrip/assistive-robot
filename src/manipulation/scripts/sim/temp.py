import heapq

elements = [(1, 'a'), (2, 'b'), (3, 'c')]

# Convert the list of elements to a heap
heap = []
heapq.heapify(elements)

# Check if the element 'a' is in the heap
element_in_heap = (1, 'a') in elements

# Print the result
print(element_in_heap)
