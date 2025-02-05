import pygetwindow as gw
import time
import keyboard

if __name__ == "__main__":
    try:
        while True:
            windows = gw.getWindowsWithTitle("FYPDataset")

            for w in windows:
                try:
                    w.activate()
                except:
                    w.minimize()
                    w.maximize()
                time.sleep(30)  # Pause to observe the switch
    except KeyboardInterrupt:
        print("\nProgram terminated.")
