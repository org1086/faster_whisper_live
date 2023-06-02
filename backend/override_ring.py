class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self, size_max):
        self.max = size_max
        self.data = []
        self.cur = 0

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max
        def extend(self, xs):
            """ Extend multiple elements to the ring. """

            # WARN: update current position beforehand
            # then trace back to update the last update data
            old_pos = self.cur
            self.cur = (self.cur + len(xs)) % self.max

            # update data to the ring
            if len(xs) >= self.max:
                update_data = xs[-self.max:]
                self.data[self.cur:] = update_data[:(self.max-self.cur)]
                self.data[:self.cur] = update_data[(self.max-self.cur):]
            else:
                if old_pos + len(xs) <= self.max:
                    self.data[old_pos:old_pos + len(xs)] = xs
                else:
                    self.data[old_pos:] = xs[:self.max - old_pos]
                    self.data[:old_pos + len(xs) - self.max] = xs[self.max - old_pos:]

        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:]+self.data[:self.cur]

    def append(self,x):
        """ Append an element at the end of the buffer. """
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            self.__class__ = self.__Full
        else: 
            self.cur +=1
    
    def extend(self, xs):
        """ Extend multiple elements to the buffer. """
        if self.cur + len(xs) <= self.max:
            self.data.extend(xs)
            if len(self.data) == self.max:
                self.cur = 0
                self.__class__ = self.__Full
            else:
                self.cur += len(xs)
        else:
            # TRICK: add 0s to fill the buffer
            self.data.extend([0.0]*(self.max - self.cur))

            # switch this class to __Full
            self.__class__ = self.__Full
            
            # call extend method of the __Full class
            self.extend(xs)

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

def print_test(my_ring):
    print('----------------------------------------------')
    print(f'cursor: {my_ring.cur}')
    print(f'data array len: {len(my_ring.data)}')
    print(f'buffer get len: {len(my_ring.get())}')

if __name__ == "__main__":
    my_ring = RingBuffer(160000)    

    # start = time.time()
    init_sample = 147456
    my_ring.extend(range(0,init_sample))
    print_test(my_ring)

    new_size = 24576
    print('----------------------------------------------')
    print(f'new audio size: {new_size}')

    for i in range(1000):
        my_ring.extend(range(init_sample + i*new_size,147456 + (i+1)*new_size))
        print_test(my_ring)

    # end = time.time()
    # print(f'total time execution: {end - start} secs.')