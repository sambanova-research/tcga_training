"""
Copyright 2023 SambaNova Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import torch


def worker_init(worker_id):
    """
    Initialization routine to be executed at startup by each data loader.
    This should be passed to the data loader via the worker_init_fn parameter.
    """
    # limit the number of threads spawned by torch pthreadpool_create to avoid
    # contention in the data loader workers.
    # the default is the number of cores in the system
    torch.set_num_threads(1)
