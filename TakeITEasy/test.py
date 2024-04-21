from thread import Thread
import time 
k = 0
def count():
    for i in range(1000000):
        k += 1

my_thread = Thread(target = count)

def current_milli_time():
    return round(time.time() * 1000)

timer = current_milli_time()
for i in range(1000000):
    k += 1
print('manual',current_milli_time()-timer)
print(k)
k = 0
timer = current_milli_time()
my_thread.start()
print('thread',current_milli_time()-timer)
time.sleep(3)
print(k)