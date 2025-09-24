Filesystem                      Size  Used Avail Use% Mounted on
/dev/root                       116G   88G   28G  77% /
tmpfs                           7.8G     0  7.8G   0% /dev/shm
tmpfs                           3.1G  968K  3.1G   1% /run
tmpfs                           5.0M     0  5.0M   0% /run/lock
efivarfs                        128K  4.1K  119K   4% /sys/firmware/efi/efivars
/dev/nvme1n1p16                 881M  165M  655M  21% /boot
/dev/nvme1n1p15                 105M  6.2M   99M   6% /boot/efi
/dev/mapper/vg.01-lv_ephemeral  115G   24K  109G   1% /opt/dlami/nvme
tmpfs                           1.6G   12K  1.6G   1% /run/user/1000


## time

- start: 14:00
- read, setup


## draft report
Write a draft for the final repo, assume invented results
give the report in markdown

using uv for packages

## P1

- [ ] profiler use
- [ ] profile baseline
- [ ] analysis of bottlenecks
- [ ] optimize bottlenecks
  - [ ] one at a time
    - [ ] test correctness
    - [ ] test performance
    - [ ] document changes
- [ ] compare results


profile using length different inputs - if it growths a lot with input length, then it is a bottleneck, might be the decoder, it depends on the length of the output sequence, then we can do KV cache