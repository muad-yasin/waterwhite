"""The writer of Check 4's scratch store: it recovers the position-mode map of a shard
from its registered tokenizer input and writes it, through the store's write call.

It is run as a process of its own, because the rows of Check 4 kill it. The arguments
are the store file, the shard's name, the input's name, a JSON object with the
tokenizer settings of the spec, and the number of rows of the map table after which the
process kills itself (-1: it is not killed). No tokenizer is given, so the map is the
one position mode derives with no verification run (docs/interfaces.md section 5.1); the
rows of Check 4 ask what the store holds after the write, not what the map says.

The kill is the store's own hook for a test that stops a writer in the middle of a
transaction (``Store.set_row_hook``): the process is killed with SIGKILL as the row it
names is handed to SQLite, so nothing after it runs, no rollback and no close.
"""

import json
import os
import signal
import sys

from waterwhite import tokenmap
from waterwhite.store import Store

store_file, shard, input_name, settings, kill_at = sys.argv[1:6]
store = Store.open(store_file)
if int(kill_at) >= 0:
    def hook(table, written):
        if table == "map_row" and written == int(kill_at):
            os.kill(os.getpid(), signal.SIGKILL)

    store.set_row_hook(hook)
tokenmap.recover_map(store, shard=shard, input=input_name,
                     settings=tokenmap.TokenizerSettings(**json.loads(settings)))
store.close()
print("finished")
