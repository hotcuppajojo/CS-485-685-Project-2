# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Create writable directories for Matplotlib cache and Gradio flagging
RUN mkdir -p /tmp/matplotlib && \
    mkdir -p /usr/src/app/flagged && \
    chmod -R 777 /tmp/matplotlib && \
    chmod -R 777 /usr/src/app/flagged

# Set environment variable to specify the Matplotlib configuration directory
ENV MPLCONFIGDIR=/tmp/matplotlib

# Set environment variable to specify the Gradio flagging directory
ENV FLAGGING_DIR=/usr/src/app/flagged

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the Gradio app will run on
EXPOSE 7860

# Run the app.py script when the container launches
CMD ["python", "app.py"]