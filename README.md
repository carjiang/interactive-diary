# interactive-diary
CS 4701 project for Michael, Armaan, Carly

Hi welcome to Interactive Diary! This is an interactive diary which utilizes thought tracing to extract a user's beliefs, retrieval over counselling conversations dataset, and calls to Chat GPT. **Please let us know ASAP if you have problems with set up and running the diary.**

## Set Up
### Virtual Environment
1. Create and set up a python [virtual environment](https://docs.python.org/3/library/venv.html)
1. On the virtual environment, install packages via `pip install -r requirements.txt`

### Environment File
1. Create `.env` file in the root directory of this project
2. Paste `OPENAI_API_KEY=` followed by the API key we send to you privately. Please do not share this API key or use it for any other purposes. If you run out of credits please let us know so we can give more!

### Model Weights
1. Download and unzip models weights. It should be a structured as `thought_trace/extractor/checkpoints/best_v3`
1. Move `best_v3` to the same directory in the project repositor: `thought_trace/extractor/checkpoints/best_v3`

### Docker
3. Install [Docker Desktop](https://docs.docker.com/desktop/), and update to the latest version
1. Start Docker Desktop in order to start the Docker daemon
4. Run `docker compose up --build -d` (Note: this may take several minutes...)
4. Stop the container with `docker compose down`


## Run Diary Program
1. Start Docker Desktop in background
2. Run `docker compose up -d`
3. Activate virtual environment
4. Run `python mvp.py` from the project root
5. After you're done, stop the docker container with `docker compose down`
5. Review interactive diary conversations in `session_logs/`
5. Enjoy!


### Usage Guidelines:
- Use first-person
- Feel free to use our program to journal or diary about anything. We do NOT store or have access to any logs.
- Your diary session can be as short or long as you want: you can choose when to end the session (by entering "no" when it asks if you want to continue), or whether to continue conversing. All logs are stored locally on your own computer.
- Fill out survey. Keep track of ablation key for submission to the survey.
- If you cannot get audio to work, apply a text tag `python mvp.py --text`
- Don't abuse use of your API key; only use it here and don't use the diary for hours

### Security Concerns:
- Do not share your OpenAI API key
- Do not commit/upload the docker image



