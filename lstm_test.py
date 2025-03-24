import numpy as np
import onnxruntime as ort

class ONNXAgent:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        # Initialize memory with 3 dimensions [num_layers, batch_size, hidden_size]
        self.memory = np.zeros((1, 1, 256), dtype=np.float32)  # Changed to 3D tensor

    def get_action(self, observation):
        # Prepare inputs with correct dimensions
        inputs = {
            'obs_0': np.expand_dims(observation, 0).astype(np.float32),  # Shape [1, 6]
            'recurrent_in': self.memory,  # Now shape [1, 1, 256]
        }

        # Execute inference
        outputs = self.session.run(None, inputs)

        # Update memory - ensure output has same 3D structure
        self.memory = outputs[5]  # Verify index matches your model's output

        return outputs[5][0][0][0:5]  # Return action

# Usage remains the same
agent = ONNXAgent("model_lstm.onnx")
obs = np.random.randn(6)  # Input shape [6]
action = agent.get_action(obs)
print("Action:", action)