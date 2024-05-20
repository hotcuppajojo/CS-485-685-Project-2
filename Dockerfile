# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Create a non-root user
RUN useradd -m myuser

# Set environment variables for Matplotlib and Gradio
ENV MPLCONFIGDIR=/home/myuser/matplotlib
ENV FLAGGING_DIR=/home/myuser/flagged

# Set the working directory in the container
WORKDIR /home/myuser/app

# Create directories and set permissions
RUN mkdir -p /usr/src/app/matplotlib /usr/src/app/flagged && \
    chmod -R 777 /usr/src/app/matplotlib /usr/src/app/flagged

# Copy the current directory contents into the container at /home/myuser/app
COPY . .

# Change ownership of the app directory
RUN chown -R myuser:myuser /home/myuser/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Switch to the non-root user
USER myuser

# Expose the port that the Gradio app will run on
EXPOSE 7860

# Run the app.py script when the container launches
CMD ["python", "app.py"]