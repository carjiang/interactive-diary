from openai import OpenAI


def test_llm():
    client = OpenAI()

    response = client.responses.create(
        model="gpt-5",
        instructions="You will follow instructions to a tee.",
        input="Repeat the following back to me, but reverse the word order: 'hello beautiful world'",
    )

    words = response.output_text.strip().rstrip(".").lower().split()
    assert words == ["world", "beautiful", "hello"], (
        f"Expected reversed word order, got: {response.output_text!r}"
    )
