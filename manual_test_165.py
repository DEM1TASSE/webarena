# My manual test for task 165

import json
import os
import re
import subprocess
import time

SLEEP = 1
# set the URLs of each website, we use the demo sites as an example
os.environ[
    "SHOPPING"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
os.environ[
    "SHOPPING_ADMIN"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
os.environ[
    "REDDIT"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
os.environ[
    "GITLAB"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
os.environ[
    "MAP"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
os.environ[
    "WIKIPEDIA"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
os.environ[
    "HOMEPAGE"
] = "PASS"  # The home page is not currently hosted in the demo site
print("Done setting up URLs")

# # First, run `python scripts/generate_test_data.py` to generate the config files
# p = subprocess.run(
#     ["python", "scripts/generate_test_data.py"], capture_output=True
# )

# # It will generate individual config file for each test example in config_files
# assert os.path.exists("config_files/0.json")

# # Make sure the URLs in the config files are replaced properly
# with open("config_files/0.json", "r") as f:
#     config = json.load(f)
#     assert os.environ["SHOPPING_ADMIN"] in config["start_url"], (
#         os.environ["SHOPPING_ADMIN"],
#         config["start_url"],
#     )

# print("Done generating config files with the correct URLs")

# run bash prepare.sh to save all account cookies, this only needs to be done once
# subprocess.run(["bash", "prepare.sh"])
# print("Done saving account cookies")

from agent.utils import Trajectory

# Init an environment
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    ScriptBrowserEnv,
    StateInfo,
    action2str,
    create_id_based_action,
    create_stop_action,
)
from evaluation_harness.evaluators import evaluator_router

# Init the environment
env = ScriptBrowserEnv(
    headless=False,
    slow_mo=100,
    observation_type="accessibility_tree",
    current_viewport_only=True,
    viewport_size={"width": 1280, "height": 720},
)

config_file = "config_files/165.json"
# maintain a trajectory
trajectory: Trajectory = []

# set the environment for the current example
obs, info = env.reset(options={"config_file": config_file})
# observation 1
actree_obs = obs["text"]

print("observation 1", actree_obs)

# action 1 scroll down
action = create_id_based_action(f"scroll [down]")
trajectory.append(action)

# observation 2
print("action 1:", action)
obs, _, terminated, _, info = env.step(action)
actree_obs = obs["text"]
print("observation 2")
print(actree_obs)
time.sleep(SLEEP)

state_info = {"observation": obs, "info": info}
trajectory.append(state_info)

# action 2 click the tab of reviews
match = re.search(r"\[(\d+)\] tab 'Reviews \(4\)'", actree_obs)
action = create_id_based_action(f"click [{match.group(1)}]")
trajectory.append(action)
print("action 2", action)

# observation 3
obs, _, terminated, _, info = env.step(action)
actree_obs = obs["text"]
print("observation 3")
print(actree_obs)
time.sleep(SLEEP)

state_info = {"observation": obs, "info": info}
trajectory.append(state_info)

# action 3 click the tabpanel of reviews
match = re.search(r"\[(\d+)\] tabpanel 'Reviews \(4\)'", actree_obs)
action = create_id_based_action(f"click [{match.group(1)}]")
trajectory.append(action)
print("action 3", action)

# observation 4
obs, _, terminated, _, info = env.step(action)
actree_obs = obs["text"]
print("observation 4")
print(actree_obs)
time.sleep(SLEEP)

state_info = {"observation": obs, "info": info}
trajectory.append(state_info)

# add useful information to memory

# action 4 scroll down to see the full review
action = create_id_based_action(f"scroll [down]")
trajectory.append(action)

# observation 5
obs, _, terminated, _, info = env.step(action)
actree_obs = obs["text"]
print("observation 5")
print(actree_obs)
time.sleep(SLEEP)

state_info = {"observation": obs, "info": info}
trajectory.append(state_info)

# add useful information to memory
# memory_reasoning

# action 5 based on the memory, conduct the result
# action_reasoning
action = create_stop_action("The 39 was too small. I am afraid the 40 will be too big. I was very sad when the shoe rubbed up against my baby toe. I had to return them because I knew in time it would tear up my feet. The problem is that the strap is made of some really stiff leather and is painful to my heel. The front is also uncomfortably tight. The Dansko's were similar (not as bad) and loosened up over time.")
trajectory.append(action)

evaluator = evaluator_router(config_file)
score = evaluator(
    trajectory=trajectory,
    config_file=config_file,
    page=env.page,
    client=env.get_page_client(env.page),
)

print("score:", score)
