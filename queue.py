# Queue implementation in Python
class Queue:

    def __init__(self, max_len):
        self.queue = []
        self.max_len = max_len

    # Add an element
    def enqueue(self, item):
        if self.size() == self.max_len:
            self.dequeue()
        self.queue.append(item)

    # Remove an element
    def dequeue(self):
        if len(self.queue) < 1:
            return None
        return self.queue.pop(0)

    def peek(self, index):
        try:
            return self.queue[index]
        except:
            return None

    # Display  the queue
    def display(self):
        print(self.queue)

    def size(self):
        return len(self.queue)