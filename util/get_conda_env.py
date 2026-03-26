import os

def get_conda_environment():
    """
    Tries to retrieve the name of the current conda environment.

    This function checks for environment variables set by conda upon activation.
    It first checks 'CONDA_DEFAULT_ENV', which conda sets to the name of the
    active environment. As a fallback, it checks 'CONDA_PREFIX', which holds
    the path to the active environment, and extracts the name from there.

    Returns:
        str: The name of the current conda environment if it can be found.
        None: If neither environment variable is set, indicating that the
              user is likely not in a conda environment.
    """
    # 'CONDA_DEFAULT_ENV' is the most reliable and direct way to get the env name.
    env_name = os.environ.get('CONDA_DEFAULT_ENV')
    if env_name:
        return env_name

    # As a fallback, we can derive the name from the 'CONDA_PREFIX' path.
    # The environment name is the last part of the path.
    env_path = os.environ.get('CONDA_PREFIX')
    if env_path:
        return os.path.basename(env_path)
    
    # If neither are set, we are likely not in a conda environment.
    return None

# --- Example Usage ---
if __name__ == '__main__':
    # Call the function to get the current environment name
    current_env = get_conda_environment()

    # Check the result and print a message to the user
    if current_env:
        print(f"✅ You are in the conda environment: '{current_env}'")
    else:
        print("❌ Could not determine the current conda environment.")
        print("   This script is likely not running in an active conda environment.")

