import json
from make_dataset import make_search_example, make_extraction_example
import random


def test_training_examples_have_conversational_prompt_completion():
    rng = random.Random(10)
    for example in (make_search_example(rng, 0), make_search_example(rng, 1),
                    make_extraction_example(rng, 0), make_extraction_example(rng, 1)):
        assert example["prompt"][0]["role"] == "system"
        assert example["prompt"][1]["role"] == "user"
        assert example["completion"][0]["role"] == "assistant"
        obj = json.loads(example["completion"][0]["content"])
        assert isinstance(obj, dict)
