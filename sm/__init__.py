import os
import sys

# macOS malloc keeps freed large blocks per process. numpy's temporaries change size with every clip length, so that cache
# only grows: 25 GB within a minute of measuring, from arrays that peak at 4 GB. malloc reads this setting at process start.
if sys.platform == "darwin" and os.environ.get("MallocLargeCache") != "0" and sys.orig_argv[1:2] == ["-m"] and sys.orig_argv[2].startswith("sm."):
    os.environ["MallocLargeCache"] = "0"
    os.execv(sys.executable, sys.orig_argv)
