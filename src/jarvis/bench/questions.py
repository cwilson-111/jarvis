"""Fixed request set for the end-to-end latency benchmark.

Short, everyday voice-assistant requests. Each is spoken as "Hey Jarvis, <q>"
by a rotating set of Kokoro voices that are *not* the assistant's own voice.
"""

QUESTIONS: tuple[str, ...] = (
    "What is the capital of Australia?",
    "How many minutes are in a day?",
    "Give me a quick tip for falling asleep faster.",
    "What's the difference between weather and climate?",
    "Why is the sky blue?",
    "How do I boil an egg?",
    "Who wrote Pride and Prejudice?",
    "What is photosynthesis?",
    "Can you suggest a name for a golden retriever?",
    "How far away is the moon?",
    "What should I cook for dinner tonight?",
    "Explain what a black hole is.",
    "How many ounces are in a pound?",
    "What's a good way to stay focused while studying?",
    "Tell me a fun fact about octopuses.",
    "What year did the first person walk on the moon?",
    "How do vaccines work?",
    "What's the tallest mountain in the world?",
    "Recommend a classic science fiction book.",
    "How can I make my coffee less bitter?",
    "What is machine learning?",
    "How long should I steep green tea?",
    "What causes thunder?",
    "Give me a two sentence summary of Romeo and Juliet.",
    "How do I convert Celsius to Fahrenheit?",
    "What's the largest ocean on Earth?",
    "Why do cats purr?",
    "What is the speed of light?",
    "Suggest a short stretching routine for my back.",
    "What does a project manager do?",
)

SPEAKER_VOICES: tuple[str, ...] = ("af_heart", "am_adam", "bf_emma", "bm_george", "af_bella")

WAKE_PREFIX = "Hey Jarvis, "
