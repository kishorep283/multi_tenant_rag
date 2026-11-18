
import time
import uuid


def generate_request_id():
    return str(uuid.uuid4())


class StageTimer:
    def __init__(self):
        self.stages = {}

    def start(self):
        return time.perf_counter()

    def stop(self, stage_name, start_time):
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stages[stage_name] = round(elapsed_ms, 2)
        return self.stages[stage_name]
