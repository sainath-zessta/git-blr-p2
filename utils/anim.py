import time
from .color import COLORS
import sys

animation = [
    "[        ]",
    "[=       ]",
    "[===     ]",
    "[====    ]",
    "[=====   ]",
    "[======  ]",
    "[======= ]",
    "[========]",
    "[ =======]",
    "[  ======]",
    "[   =====]",
    "[    ====]",
    "[     ===]",
    "[      ==]",
    "[       =]",
    "[        ]",
    "[        ]",
]


def load_animation(limit):
    notcomplete = True

    i = 0

    while notcomplete:
        print(animation[i % len(animation)], end="\r")
        time.sleep(0.1)
        i += 1
        if i == limit:
            sys.stdout.flush()
            sys.stdout.write("\r          ")

            notcomplete = False


# here is the animation
def animate(limit):
    i = 0
    notcomplete = True

    while notcomplete:
        i += 1

        sys.stdout.write("\rRunning Detection |")
        time.sleep(0.1)
        sys.stdout.write("\rRunning Detection /")
        time.sleep(0.1)
        sys.stdout.write("\rRunning Detection -")
        time.sleep(0.1)
        sys.stdout.write("\rRunning Detection \\")
        time.sleep(0.1)

        if i == limit:
            notcomplete = False
        sys.stdout.flush()
    # sys.stdout.write("\rCompleted! Saving Results to JSON\n exited              ")
    # sys.stdout.flush()
