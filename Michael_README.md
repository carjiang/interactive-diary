# Set up Docker
1. Run Docker on your computer
1. Build docker image `docker build -t your_username/your_docker_image_repository .`
1. Run docker image followed by command. E.g. if you want to run a test: 
```bash
docker run your_username/your_docker_image_respository \
    python python_file_you_want_to_run_in_container.py
```
1. When you want to save your image, push image to your Docker Hub with `docker push your_username/your_docker_image_repository`

# Run Docker Test
```bash
docker run your_username/your_docker_image_respository \
    python -m pytest tests/test_whisper.py
```

# Generate Synthetic Diary Dataset
Modify the location and number of entries generated in `data/gen_diary.py`
```bash
docker run --env-file .env -v $(pwd):/app -w /app mikono/id  \
    python data/gen_diary.py
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

