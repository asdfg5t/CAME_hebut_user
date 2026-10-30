"""Each child owns the GPU lock even if its PyCharm parent is forcibly stopped."""
import runpy
import sys
from .ide_workflow import acquire_gpu_lock
from .pause_control import TrainingPaused, pause_if_requested


def main():
    arguments = sys.argv[1:]
    if arguments and arguments[0] == '--':
        arguments = arguments[1:]
    lock = acquire_gpu_lock()
    try:
        pause_if_requested()
        if arguments[:1] == ['-m']:
            sys.argv = [arguments[1], *arguments[2:]]
            runpy.run_module(arguments[1], run_name='__main__')
        else:
            sys.argv = arguments
            runpy.run_path(arguments[0], run_name='__main__')
    except TrainingPaused as error:
        print('SAFE_PAUSE=' + str(error), flush=True)
        raise SystemExit(75)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
