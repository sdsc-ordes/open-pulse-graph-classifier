import sys


def test_print_sys_path():
    print("\nPYTHONPATH used by pytest:")
    for p in sys.path:
        print("  ", p)
