from openai import OpenAI


def test_llm():
    client = OpenAI()

    response = client.responses.create(
        model="gpt-5",
        instructions="You will follow instructions to a tee.",
        input="Repeat the following back to me, but reverse the word order: 'hello beautiful world'",
    )

    assert response.output_text == "world beautiful hello"
