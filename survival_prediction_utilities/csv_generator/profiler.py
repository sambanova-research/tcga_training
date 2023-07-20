
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


import math
import time
from collections import defaultdict


def significant_figures(func):
    """ Decorator which rounds the output of a function to some number of significant figures """
    def round_func(*args, **kwargs):
        sig_figs = kwargs.get('sig_figs', None)
        kwargs.pop('sig_figs', None)
        num = func(*args, **kwargs)
        if sig_figs is None:
            return num
        if num == 0:
            return num
        return round(num, sig_figs - int(math.floor(math.log10(abs(num)))) - 1)

    return round_func


class Profiler:
    """ Profiler used to estimate the remaining time until completion of my data processing scripts """
    def __init__(self):
        self.start_times = {}
        # TODO: delete this data structure, huge waste of space
        self.event_times = defaultdict(list)
        self.sum_metrics = defaultdict(float)
        self.num_metrics = defaultdict(int)
        self.metrics = {}
        self.sum_times = defaultdict(int)

    def start_event(self, event_name: str):
        """ Starts timing an event """
        start_time = time.time()
        self.start_times[event_name] = start_time

    def end_event(self, event_name: str):
        """ Stops timing an event """
        end_time = time.time()
        time_elapsed = end_time - self.start_times[event_name]
        # TODO: delete this data structure, huge waste of space
        self.event_times[event_name].append(time_elapsed)
        self.sum_times[event_name] += time_elapsed

    def track_metric_avg(self, metric_name: str, value: float):
        """ Updates the average of a specified metric """
        self.sum_metrics[metric_name] += value
        self.num_metrics[metric_name] += 1

    def track_metric(self, metric_name: str, value: float):
        """ Store a metric """
        self.metrics[metric_name] = value

    def update_metric(self, metric_name: str, func: callable):
        """ Update a metric """
        self.metrics[metric_name] = func(self.metrics[metric_name])

    def get_metric(self, metric_name: str):
        """ Get metric value """
        return self.metrics[metric_name]

    @significant_figures
    def get_metric_avg(self, metric_name: str) -> float:
        """ Get average of a metric that's being tracked """
        if metric_name not in self.num_metrics:
            return 0
        return self.sum_metrics[metric_name] / self.num_metrics[metric_name]

    @significant_figures
    def get_time(self, event_name: str) -> float:
        """ Get time of an event """
        if event_name not in self.sum_times:
            return 0
        return self.sum_times[event_name] / len(self.event_times[event_name])

    @significant_figures
    def expected_time(self, event_name: str, num_iters: int):
        """ Get the expected time an event will take given how many iterations of that event must be done
        """
        return num_iters * self.get_time(event_name)
