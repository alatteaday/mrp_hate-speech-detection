# Set the 5 variables below as you want 
IMAGE_NAME = mrp-image
IMAGE_TAG = 0.1
CONTAINER_NAME = mrp-container
CONTAINER_PORT = 1234
NVIDIA_VISIBLE_DEVICES = all  # 'all' OR the numbers of available gpus such as '1,2,3'

SH := /bin/bash
WD := /home/mrp

docker-build:
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

docker-run:
	docker run -it -d --restart always -v $(shell pwd):${WD} -p ${CONTAINER_PORT}:${CONTAINER_PORT} --name ${CONTAINER_NAME} --ipc=host --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} ${IMAGE_NAME}:${IMAGE_TAG} ${SH}

docker-exec:
	docker exec -it -w ${WD} ${CONTAINER_NAME} ${SH}

docker-start:
	docker start ${CONTAINER_NAME}

docker-stop:
	docker stop ${CONTAINER_NAME}

docker-rm:
	docker rm ${CONTAINER_NAME}

docker-rmi:
	docker rmi ${IMAGE_NAME}:${IMAGE_TAG}