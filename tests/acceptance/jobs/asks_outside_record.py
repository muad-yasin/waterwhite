"""The job of the harness row of Check 5 (c05.harness.outside-read), run confined.

It reads every record it was handed, then asks for one record that was not
handed, and says what came back. The arguments are the instance id and the
locator of that record. The job's one output is the word "refused", or
"delivered:" and the text it was given; it states no record as used."""

import sys

import waterwhite_job

with waterwhite_job.connect() as job:
    for handed in job.handed:
        job.read(handed)
    try:
        text = job.read(waterwhite_job.Ref(sys.argv[1], sys.argv[2]))
        answer = "delivered:" + text
    except waterwhite_job.Refused:
        answer = "refused"
    job.emit(answer, used=[])
