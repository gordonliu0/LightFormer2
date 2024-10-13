import sys
import time
import concurrent.futures
from enum import Enum

class AnimationSequence(Enum):
    DEFAULT = "|/-\\"
    CROSS = "x+"
    CIRCLE = "◐◓◑◒"

default = "|/-\\"

def animated_loading(chars):
    'Threading animation. Define the animated spinner using the chars parameter.'
    for char in chars:
        sys.stdout.write('\r'+'Working '+char)
        time.sleep(.1)
        sys.stdout.flush()

# Run a function with loading animation.
def run_with_animation(f, args=(), kwargs={}, animation_chars=AnimationSequence.DEFAULT, name='process'):
    'Animate a function f. '
    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=name) as executor:
        print(f"Starting process '{name}'")
        start_time = time.time()
        future = executor.submit(f, *args, **kwargs)
        while not future.done():
            animated_loading(animation_chars)
        sys.stdout.write('\r' + ' ' * 20 + '\r')
        sys.stdout.flush()
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Finished process '{name}' in {int(hours):02d}h:{int(minutes):02d}m:{seconds:05.2f}s.")
        return future.result()
