# Internal Process Monitor Start
# Internal Metacognitive Introspection 
# Nov6 self aware base preparation.
```python
     import psutil
     import time
     from collections import deque

     class InternalProcessMonitor:
         def __init__(self, max_history_size=100):
             self.cpu_usage_history = deque(maxlen=max_history_size)
             self.memory_usage_history = deque(maxlen=max_history_size)
             self.task_queue_length_history = deque(maxlen=max_history_size)
             self.knowledge_base_updates_history = deque(maxlen=max_history_size)
             self.model_training_time_history = deque(maxlen=max_history_size)
             self.model_inference_time_history = deque(maxlen=max_history_size)

         def monitor_cpu_usage(self):
             self.cpu_usage_history.append(psutil.cpu_percent())

         def monitor_memory_usage(self):
             self.memory_usage_history.append(psutil.virtual_memory().percent)

         def monitor_task_queue_length(self, queue_size):
             self.task_queue_length_history.append(queue_size)

         def monitor_knowledge_base_updates(self, num_updates):
             self.knowledge_base_updates_history.append(num_updates)

         def monitor_model_training_time(self, training_time):
             self.model_training_time_history.append(training_time)

         def monitor_model_inference_time(self, inference_time):
             self.model_inference_time_history.append(inference_time)

         def get_historical_data(self):
             return {
                 "cpu_usage": list(self.cpu_usage_history),
                 "memory_usage": list(self.memory_usage_history),
                 "task_queue_length": list(self.task_queue_length_history),
                 "knowledge_base_updates": list(self.knowledge_base_updates_history),
                 "model_training_time": list(self.model_training_time_history),
                 "model_inference_time": list(self.model_inference_time_history)
             }
     ```
# Internal Process Monitor end.
