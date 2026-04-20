# interactive-diary
CS 4701 project for Michael, Armaan, Carly


# Run the program
```bash
# Build and start
docker compose up --build

# Open a shell inside the container
docker compose exec app bash

# Run command in container
docker compose exec -T app [command]
```

1. First run `docker compose up --build`

2. Then in another terminal, install requirements from host_requirements.txt
and run `python interface/mvp.py`

When you want to save your image, push image to your Docker Hub with `docker push your_username/your_docker_image_repository`

# Run Docker Test
```bash
docker build -t mikono/id .
docker run your_username/your_docker_image_respository \
    python -m pytest tests/test_whisper.py
```

# Generate Synthetic Diary Dataset
Modify the location and number of entries generated in `data/gen_diary.py`
```bash
docker run --env-file .env -v $(pwd):/app -w /app your_username/your_docker_image_respository  \
python -u data/gen_diary.py
```

# Load Data Set
```bash
docker run mikono/id \
python -c "from data.schema_spec import load_dataset; load_dataset('data/synth_diary.jsonl', strict=False)"
```

# Run LLM Test
1. First, set up a `.env` file containing the following
```bash
OPENAI_API_KEY=<insert_your_openai_key>
```
1. Once you've built the docker image, run:
```bash
docker run --env-file .env mikono/id \
    python -m pytest tests/test_llm.py
```

# Run Live Transcription
1. Build docker image with name `your_username/your_docker_image_respository`.
1. Set up a virtual environment and install packages at `host_requirements.txt` with `pip install -r host_requirements.txt`.
1. Run trascriptions script
```bash
python host_tests/live_transcription.py your_username/your_docker_image_repository
```

# Whipser
Refer to [OpenAI's Whisper](https://github.com/openai/whisper?tab=readme-ov-file)

