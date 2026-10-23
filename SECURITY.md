# Security policy

Report vulnerabilities and **privacy leaks** privately, through GitHub's "Report a vulnerability"
button on the repository's Security tab. A privacy leak is any case where vision-toolkit shows,
writes, uploads or publishes what it should have hidden: a face left visible by `anonymize`, the
audience left in view below the redaction line, a person left in a board snapshot, an image that
leaves the machine without a review. Do not open a public issue and do not attach real frames:
describe the case in words or reproduce it with synthetic data.

Only the latest version is supported.

Out of scope: vulnerabilities of OpenCV, FFmpeg or MediaPipe themselves (report them to those
projects), and the false doubts described as fragile points in
[docs/usecases/anonymize.md](docs/usecases/anonymize.md#status).
