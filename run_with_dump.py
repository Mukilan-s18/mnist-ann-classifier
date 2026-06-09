import faulthandler
import signal
import time
import threading

faulthandler.enable()
faulthandler.register(signal.SIGALRM, all_threads=True)

def alarm_handler():
    time.sleep(10)
    import os
    os.kill(os.getpid(), signal.SIGALRM)

t = threading.Thread(target=alarm_handler, daemon=True)
t.start()

import src.mnist_ann_classifier
src.mnist_ann_classifier.main()
