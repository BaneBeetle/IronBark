# Operation Iron Bark
# BaneBeetle

import time, signal, os
from pidog import Pidog
# from preset_actions import bark  # keep if you have it

DANGER_DISTANCE = 15

def main():
    dog = None
    try:
        dog = Pidog()
        dog.do_action('stand', speed=80)
        dog.wait_all_done()
        time.sleep(0.5)

        stand = dog.legs_angle_calculation([[0, 80], [0, 80], [30, 75], [30, 75]])

        def patrol():
            d = round(dog.read_distance(), 2)  # this reads a shared value set by the child process
            print(f"distance: {d} cm", end="", flush=True)
            if 0 < d < DANGER_DISTANCE:
                print("\033[0;31m DANGER !\033[m")
                dog.body_stop()
                head_yaw = dog.head_current_angles[0]
                dog.rgb_strip.set_mode('bark', 'red', bps=2)
                dog.tail_move([[0]], speed=80)
                dog.legs_move([stand], speed=70)
                dog.wait_all_done()
                time.sleep(0.5)
                # bark(dog, [head_yaw, 0, 0])  # keep if available
                while True:
                    d = round(dog.read_distance(), 2)
                    if d >= DANGER_DISTANCE or d <= 0:
                        break
                    print(f"distance: {d} cm \033[0;31m DANGER !\033[m")
                    time.sleep(0.08)  # don't hammer at 10ms
            else:
                print("")
                dog.rgb_strip.set_mode('breath', 'white', bps=0.5)
                dog.do_action('forward', step_count=2, speed=98)
                dog.do_action('shake_head', step_count=1, speed=80)
                dog.do_action('wag_tail', step_count=5, speed=99)

        while True:
            patrol()
            time.sleep(0.08)  # friendlier polling

    except KeyboardInterrupt:
        print("\nCtrl+C — cleaning up…")

    finally:
        if dog is not None:
            # --- ensure the child process actually dies and releases GPIO ---
            try:
                sp = getattr(dog, "sensory_process", None)
                if sp is not None and sp.is_alive():
                    sp.terminate()
                    sp.join(timeout=1.5)
                    if sp.is_alive():
                        # escalate if needed
                        try:
                            os.kill(sp.pid, signal.SIGKILL)
                        except Exception:
                            pass
                # now do the library's close exactly once
                dog.close()
            except SystemExit:
                # some versions call sys.exit(0) inside close(); that's fine
                pass

if __name__ == "__main__":
    main()
