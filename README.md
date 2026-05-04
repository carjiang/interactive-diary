# interactive-diary
CS 4701 project for Michael, Armaan, Carly

Hi welcome to Interactive Diary! This is an interactive diary which utilizes thought tracing to extract a user's beliefs, retrieval over counselling conversations dataset, and calls to Chat GPT.

## Set Up
### Virtual Environment
1. Create and set up a python [virtual environment](https://docs.python.org/3/library/venv.html)
1. On the virtual environment, install packages via `pip install -r requirements.txt`

### Environment File
1. Create `.env` file in the root directory of this project
2. Paste `OPENAI_API_KEY=` followed by the API key we send to you privately. There is $3 allocated on this key. Please do not share this API key or use it for any other purposes. If you run out of credits please let us know so we can give more!

### Model Weights
1. Download and unzip models weights. It should be a folder called `checkpoints/`
1. Move `checkpoints/` to `thought_trace/extractor/checkpoints/`

### Docker
3. Install [Docker Desktop](https://docs.docker.com/desktop/), and update to the latest version
1. Start Docker Desktop in order to start the Docker daemon
4. Run `docker compose up --build -d` (Note: this may take several minutes...)
4. Stop the container with `docker compose down`


## Run Diary Program
1. Start Docker Desktop in background
2. Run `docker compose up -d`
3. Activate virtual environment
4. Run `python interface/mvp.py` from the project root
5. After you're done, stop the docker container with `docker compose down`
5. Enjoy!


### Usage Guidelines:
- Use first-person
- Once entries are recorded, they can be edited
- Feel free to use our program to journal or diary about anything. We do NOT store or have access to any logs.
- If you cannot get audio to work, apply a text tag `python interface/mvp.py --text`
- Interactive Diary responses may take multi

### Further:
- Do not share your OpenAI API key
- Do not commit/upload the docker image
- Do not share things that are too personal; all 



