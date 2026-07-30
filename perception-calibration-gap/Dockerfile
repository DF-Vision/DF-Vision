FROM osrf/ros:humble-desktop-full

# Install Gazebo Harmonic (adjust if a different Gazebo version is required)
RUN apt-get update && apt-get install -y \
    curl gnupg lsb-release mesa-utils \
    && curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
       | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
    && apt-get update \
    && apt-get install -y gz-harmonic \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project into the image (self-contained snapshot)
COPY . /workspace
WORKDIR /workspace

# Build the ROS package(s) at image-build time
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --packages-select so101_description"

# Persist environment setup for every future shell in this container
RUN { \
    echo "source /opt/ros/humble/setup.bash"; \
    echo "source /workspace/install/setup.bash"; \
    echo "export GZ_SIM_RESOURCE_PATH=/workspace/gazebo:\$GZ_SIM_RESOURCE_PATH"; \
    } >> /root/.bashrc

CMD ["bash"]
