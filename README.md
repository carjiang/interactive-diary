# interactive-diary
CS 4701 project for Michael, Armaan, Carly

Hi welcome to Interactive Diary!

## Set Up
1. Create `.env` file in the root directory of this project
2. Paste `OPENAI_API_KEY=` followed by the API key we send to you privately. There is $3 allocated on this key. Please do not share this API key or use it for any other purposes. If you run out of credits please let us know so we can give more!
3. Install [Docker Desktop](https://docs.docker.com/desktop/), and update to the latest version
1. Start Docker Desktop in order to start the Docker daemon
4. Run `docker compose up -build`
4. Stop the container with CTRL+C or `docker compose down` in a separate terminal
1. Create and set up a python [virtual environment](https://docs.python.org/3/library/venv.html)
1. On the virtual environment, install packages via `pip install -r user_requirements.txt`

## Run Diary Program
1. Start Docker Desktop in background
2. Run `docker compose up -d`
3. Activate virtual environment
4. Run `python interface/mvp.py`
5. Enjoy!