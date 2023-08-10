# My manual test for task 164
'''
Some idea:
How to decide when to scroll down?
Maybe it's better to scroll down to see the full page at first?
You need to click the tab to change tab
Also need to click the tabpanel to see the specific content
'''

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

config_file = "config_files/164.json"
# maintain a trajectory
trajectory: Trajectory = []

# set the environment for the current example
obs, info = env.reset(options={"config_file": config_file})
# observation 1
actree_obs = obs["text"]

# action 1 click review link
match = re.search(r"\[(\d+)\] link '3\\xa0 Reviews '", actree_obs)
action = create_id_based_action(f"goto [{match.group(1)}]")
trajectory.append(action)

# observation 2
# Nothing happens
print(f"click [{match.group(1)}]")
print("action 1:", action)
obs, _, terminated, _, info = env.step(action)
actree_obs = obs["text"]
print("observation 2")
print(actree_obs)
time.sleep(SLEEP)

state_info = {"observation": obs, "info": info}
trajectory.append(state_info)

# action 2 scroll down
action = create_id_based_action(f"scroll [down]")
trajectory.append(action)

# observation 3
# See the tab of reviews
print("action 2:", action)
obs, _, terminated, _, info = env.step(action)
actree_obs = obs["text"]
print("observation 3")
print(actree_obs)
time.sleep(SLEEP)

state_info = {"observation": obs, "info": info}
trajectory.append(state_info)

# action 3 click review link
match = re.search(r"\[(\d+)\] tab 'Reviews \(3\)'", actree_obs)
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

# action 4 click tab
# change tabpanel to see the content of reviews
match = re.search(r"\[(\d+)\] tabpanel 'Reviews \(3\)'", actree_obs)
action = create_id_based_action(f"click [{match.group(1)}]")
trajectory.append(action)
print("action 3", action)

# observation 5
obs, _, terminated, _, info = env.step(action)
actree_obs = obs["text"]
print("observation 5")
print(actree_obs)
time.sleep(SLEEP)

state_info = {"observation": obs, "info": info}
trajectory.append(state_info)

trajectory.append(create_stop_action("DryUneven color"))

evaluator = evaluator_router(config_file)
score = evaluator(
    trajectory=trajectory,
    config_file=config_file,
    page=env.page,
    client=env.get_page_client(env.page),
)

print("score:", score)
