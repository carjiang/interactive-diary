# Run Docker Test
```bash
docker build -t mikono/id .
docker run your_username/your_docker_image_respository \
    python -m pytest tests/test_whisper.py
```

## Run LLM Test
1. First, set up a `.env` file containing the following
```bash
OPENAI_API_KEY=<insert_your_openai_key>
```
1. Once you've built the docker image, run:
```bash
docker run --env-file .env mikono/id \
    python -m pytest tests/test_llm.py
```

## Run Live Transcription
1. Build docker image with name `your_username/your_docker_image_respository`.
1. Set up a virtual environment and install packages at `host_requirements.txt` with `pip install -r host_requirements.txt`.
1. Run trascriptions script
```bash
python host_tests/live_transcription.py your_username/your_docker_image_repository
```
