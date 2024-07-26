# Makefile to manage Docker container for bird identifier project

# Build the Docker image
build:
	docker-compose build

# Start the Docker container
up:
	docker-compose up

# Stop the Docker container
down:
	docker-compose down

# View logs of the running container
logs:
	docker-compose logs -f

# Restart the Docker container
restart: down up

# Remove all stopped containers and unused images
clean:
	docker system prune -af

# Rebuild the Docker image and restart the container
rebuild: clean build restart
