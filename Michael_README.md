# Set up Docker
1. Run Docker on your computer
1. Build docker image `docker build -t your_username/your_docker_image_repository .`
1. Run docker image followed by command. E.g. if you want to run a test: 
```bash
docker run your_username/your_docker_image_respository \
    python -m pytest tests/your_test_file_name
```
1. When you want to save your iamge, push image to your Docker Hub with `docker push your_username/your_docker_image_repository`

# Whipser
Refer to [OpenAI's Whisper](https://github.com/openai/whisper?tab=readme-ov-file)