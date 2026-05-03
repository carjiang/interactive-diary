# Run the program
```bash
# Build and start
docker compose up --build

# Build and start and detache from terminal
docker compose up --build -d

# Stop all images
docker compose down 

# Open a shell inside the container
docker compose exec app bash

# Run command in container
docker compose exec -T app [command]
```
