# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables for Matplotlib and Gradio
ENV MPLCONFIGDIR=/usr/src/app/matplotlib
ENV FLAGGING_DIR=/usr/src/app/flagged

# Set the working directory in the container
WORKDIR /usr/src/app

# Create directories and set permissions
RUN mkdir -p /usr/src/app/matplotlib /usr/src/app/flagged && \
    chmod -R 777 /usr/src/app/matplotlib /usr.src/app/flagged

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the Gradio app will run on
EXPOSE 7860

# Run the app.py script when the container launches
CMD ["python", "app.py"]