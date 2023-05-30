# Robust Collaborative Learning with Linear Gradient Overhead

### Software dependencies

Python 3.7.3 has been used to run our scripts.

Besides the standard libraries associated with Python 3.7.3, our scripts have been tested with<sup>2</sup>:

| Library     | Version    |
| ----------- | ---------- |
| numpy       | 1.19.1     |
| torch       | 1.6.0      |
| torchvision | 0.7.0      |
| pandas      | 1.1.0      |
| matplotlib  | 3.0.2      |
| PIL         | 7.2.0      |
| requests    | 2.21.0     |
| urllib3     | 1.24.1     |
| chardet     | 3.0.4      |
| certifi     | 2018.08.24 |
| idna        | 2.6        |
| six         | 1.15.0     |
| pytz        | 2020.1     |
| dateutil    | 2.6.1      |
| pyparsing   | 2.2.0      |
| cycler      | 0.10.0     |
| kiwisolver  | 1.0.1      |
| cffi        | 1.13.2     |

<sup>2</sup><sub>this list is automatically generated (see `get_loaded_dependencies()` in `tools/misc.py`).
Some dependencies depend on others, while others are optional (e.g., only used to process the results and produce the plots).</sub>

We list below the OS on which our scripts have been tested:
* Debian 10 (GNU/Linux 4.19.171-2 x86_64)
* Ubuntu 18.04.5 LTS (GNU/Linux 4.15.0-128-generic x86_64)

### Hardware dependencies

Although our experiments are time-agnostic, we list below the hardware components used:
* 1 Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz
* 2 Nvidia GeForce GTX 1080 Ti
* 64 GB of RAM

### Command
All our results can be reproduced using the following two commands.
The first command is used to reproduce all results on MNIST.
The second command reproduces all results on CIFAR-10.
In the root directory:
```
$ python3 reproduce.py
```

```
$ python3 reproduce_cifar.py
```

Please be aware these scripts require non-negligible disk space.

Depending on the hardware, instructing the script to launch several runs per available GPU may reduce the total runtime.
For instance, to push up to 4 concurrent runs per GPU:
```
$ python3 reproduce.py --supercharge 4
```
On our hardware, reproducing all our results (from scratch) with both commands takes approximately 5 days.
